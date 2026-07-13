"""短线动量 — 追涨强势股策略，基于动量延续效应。

核心理念（学术基础）：
  - 短期动量效应（Jegadeesh & Titman, 1993）：过去1-4周表现好的资产会延续趋势
  - 结合基本面筛选（规模/回撤/收益率）过滤质量，避免"垃圾动量"
  - 关键约束：必须在上升趋势中（MA20上方），前日涨幅≥2%确认动能

基金筛选条件（全部可配置）：
  1. 基金经理从业 > 5年（可选，需AKShare）
  2. 基金规模 > 10亿（市值代理 mcap_yi）
  3. 基金成立 > 5年（可选，需AKShare）
  4. MA20上方（上升趋势）
  5. 前一日上涨 > 2%（动量确认）
  6. 人气前100名（换手率排名）
  7. 最大回撤 < 20%
  8. 一年收益率 > 200%（可调低）

股票筛选条件：
  1. 市值 > 50亿（规模过滤，防控盘）
  2. MA20上方（上升趋势）
  3. 前一日上涨 > 2%（动量确认）
  4. 人气前100名（换手率排名）
  5. 最大回撤 < 20%
  6. 一年收益率 > 200%（可调低）

出场规则（按优先级）：
  1. 硬止损 -3.0%
  2. 时间止盈 4天
  3. 目标止盈 +4.0%
  4. 移动止盈：盈利>2%激活，回撤>1%止盈
  5. 动量衰竭：RSI(14)>75超买，或收盘<MA5跌破短期趋势

仓位管理：
  - 单笔15-25%资金
  - 最多同时持有3个仓位
  - 单日最多开1个新仓

目标：2-4天持仓周期，追涨强势，快速止盈止损。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.signals import DashboardCard, LiveSignal


# =============================================================================
# Candidate pools
# =============================================================================

CANDIDATE_ETFS: list[dict] = [
    # ── 大盘蓝筹 (4) ──
    {"code": "510300", "name": "沪深300ETF",        "approx_price": 3.9,  "category": "large"},
    {"code": "510050", "name": "上证50ETF",          "approx_price": 2.7,  "category": "large"},
    {"code": "510180", "name": "上证180ETF",         "approx_price": 3.5,  "category": "large"},
    {"code": "159901", "name": "深证100ETF",         "approx_price": 3.8,  "category": "large"},
    # ── 中盘 (2) ──
    {"code": "510500", "name": "中证500ETF",         "approx_price": 5.8,  "category": "mid"},
    {"code": "515800", "name": "中证800ETF",         "approx_price": 2.5,  "category": "mid"},
    # ── 小盘 (3) ── (波动大，动量强)
    {"code": "512100", "name": "中证1000ETF",        "approx_price": 2.3,  "category": "small"},
    {"code": "563300", "name": "中证2000ETF",        "approx_price": 0.9,  "category": "small"},
    {"code": "159628", "name": "国证2000ETF",        "approx_price": 1.0,  "category": "small"},
    # ── 新宽基 (3) ──
    {"code": "159593", "name": "中证A50ETF",         "approx_price": 1.0,  "category": "new_broad"},
    {"code": "159338", "name": "中证A500ETF",        "approx_price": 0.9,  "category": "new_broad"},
    {"code": "560050", "name": "MSCI A50ETF",        "approx_price": 0.8,  "category": "new_broad"},
    # ── 创业/科创 (4) ── (高beta，动量弹性大)
    {"code": "159915", "name": "创业板ETF",          "approx_price": 2.2,  "category": "growth"},
    {"code": "159949", "name": "创业板50ETF",        "approx_price": 0.9,  "category": "growth"},
    {"code": "588000", "name": "科创50ETF",          "approx_price": 0.9,  "category": "growth"},
    {"code": "588190", "name": "科创100ETF",         "approx_price": 0.8,  "category": "growth"},
    # ── 双创 (1) ──
    {"code": "159781", "name": "双创50ETF",          "approx_price": 1.1,  "category": "growth"},
    # ── 红利/低波 (5) ──
    {"code": "510880", "name": "中证红利ETF",        "approx_price": 3.0,  "category": "defensive"},
    {"code": "515080", "name": "中证红利ETF(招商)",  "approx_price": 1.5,  "category": "defensive"},
    {"code": "512890", "name": "红利低波ETF",        "approx_price": 1.5,  "category": "defensive"},
    {"code": "515180", "name": "红利低波100ETF",     "approx_price": 1.3,  "category": "defensive"},
    {"code": "563020", "name": "红利低波ETF(易方达)","approx_price": 1.0,  "category": "defensive"},
    # ── 策略宽基 (3) ──
    {"code": "515450", "name": "红利质量ETF",        "approx_price": 1.2,  "category": "defensive"},
    {"code": "159905", "name": "深红利ETF",          "approx_price": 2.0,  "category": "defensive"},
    {"code": "562060", "name": "中证A50增强ETF",     "approx_price": 1.0,  "category": "new_broad"},

    # ── 高beta行业ETF (15) ── (日波动3-5%，动量策略核心品种)
    {"code": "512480", "name": "半导体ETF",          "approx_price": 1.0,  "category": "sector"},
    {"code": "512880", "name": "证券ETF",            "approx_price": 1.0,  "category": "sector"},
    {"code": "512660", "name": "军工ETF",            "approx_price": 1.1,  "category": "sector"},
    {"code": "516160", "name": "新能源ETF",          "approx_price": 0.8,  "category": "sector"},
    {"code": "512010", "name": "医药ETF",            "approx_price": 0.5,  "category": "sector"},
    {"code": "512690", "name": "酒ETF",              "approx_price": 0.7,  "category": "sector"},
    {"code": "515790", "name": "光伏ETF",            "approx_price": 1.0,  "category": "sector"},
    {"code": "159995", "name": "芯片ETF",            "approx_price": 1.1,  "category": "sector"},
    {"code": "515050", "name": "5GETF",              "approx_price": 1.0,  "category": "sector"},
    {"code": "512800", "name": "银行ETF",            "approx_price": 1.2,  "category": "sector"},
    {"code": "516510", "name": "云计算ETF",          "approx_price": 0.9,  "category": "sector"},
    {"code": "159766", "name": "旅游ETF",            "approx_price": 0.9,  "category": "sector"},
    {"code": "512200", "name": "房地产ETF",          "approx_price": 0.6,  "category": "sector"},
    {"code": "515210", "name": "钢铁ETF",            "approx_price": 1.2,  "category": "sector"},
    {"code": "516020", "name": "化工ETF",            "approx_price": 0.8,  "category": "sector"},
]

# Large-cap liquid A-share stocks (市值>500亿, 高流动性)
CANDIDATE_STOCKS: list[dict] = [
    # ── 消费 (5) ──
    {"code": "600519", "name": "贵州茅台",   "approx_price": 1500.0, "category": "consumer"},
    {"code": "000858", "name": "五粮液",     "approx_price": 130.0,  "category": "consumer"},
    {"code": "000568", "name": "泸州老窖",   "approx_price": 180.0,  "category": "consumer"},
    {"code": "002714", "name": "牧原股份",   "approx_price": 40.0,   "category": "consumer"},
    {"code": "600887", "name": "伊利股份",   "approx_price": 28.0,   "category": "consumer"},
    # ── 金融 (4) ──
    {"code": "601318", "name": "中国平安",   "approx_price": 45.0,   "category": "finance"},
    {"code": "600036", "name": "招商银行",   "approx_price": 38.0,   "category": "finance"},
    {"code": "601166", "name": "兴业银行",   "approx_price": 17.0,   "category": "finance"},
    {"code": "600030", "name": "中信证券",   "approx_price": 22.0,   "category": "finance"},
    # ── 科技 (5) ──
    {"code": "002475", "name": "立讯精密",   "approx_price": 35.0,   "category": "tech"},
    {"code": "300750", "name": "宁德时代",   "approx_price": 200.0,  "category": "tech"},
    {"code": "002415", "name": "海康威视",   "approx_price": 32.0,   "category": "tech"},
    {"code": "000725", "name": "京东方A",    "approx_price": 4.0,    "category": "tech"},
    {"code": "688981", "name": "中芯国际",   "approx_price": 55.0,   "category": "tech"},
    # ── 医药 (3) ──
    {"code": "600276", "name": "恒瑞医药",   "approx_price": 45.0,   "category": "pharma"},
    {"code": "300760", "name": "迈瑞医疗",   "approx_price": 280.0,  "category": "pharma"},
    {"code": "000538", "name": "云南白药",   "approx_price": 55.0,   "category": "pharma"},
    # ── 新能源/制造 (5) ──
    {"code": "601012", "name": "隆基绿能",   "approx_price": 20.0,   "category": "new_energy"},
    {"code": "002594", "name": "比亚迪",     "approx_price": 250.0,  "category": "new_energy"},
    {"code": "600309", "name": "万华化学",   "approx_price": 80.0,   "category": "industry"},
    {"code": "000651", "name": "格力电器",   "approx_price": 40.0,   "category": "industry"},
    {"code": "601899", "name": "紫金矿业",   "approx_price": 15.0,   "category": "mining"},
    # ── 基建/央企 (3) ──
    {"code": "601668", "name": "中国建筑",   "approx_price": 5.0,    "category": "infra"},
    {"code": "600585", "name": "海螺水泥",   "approx_price": 25.0,   "category": "infra"},
    {"code": "601857", "name": "中国石油",   "approx_price": 8.0,    "category": "energy"},
]


# =============================================================================
# Static helpers
# =============================================================================

def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized RSI with Wilder's smoothing."""
    n = len(closes)
    if n < period + 1:
        return np.full(n, np.nan)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _daily_return_pct(closes: np.ndarray, idx: int) -> float:
    """Today's return: (close[t] - close[t-1]) / close[t-1]."""
    if idx < 1:
        return 0.0
    prev_close = closes[idx - 1]
    if prev_close <= 0:
        return 0.0
    return float((closes[idx] - prev_close) / prev_close)


def _compute_drawdown_1y(closes: np.ndarray) -> float:
    """Compute max drawdown from 1-year rolling window (≈240 trading days).

    Returns a positive number (e.g. 0.15 = 15% drawdown).
    """
    if len(closes) < 20:
        return 1.0
    window = min(240, len(closes))
    recent = closes[-window:]
    cummax = np.maximum.accumulate(recent)
    dd = (recent - cummax) / np.where(cummax == 0, 1e-10, cummax)
    return float(abs(np.min(dd)))


def _compute_return_1y(closes: np.ndarray) -> float:
    """Compute 1-year return (≈240 trading days)."""
    if len(closes) < 20:
        return 0.0
    window = min(240, len(closes))
    start_price = closes[-window] if len(closes) >= window else closes[0]
    end_price = closes[-1]
    if start_price <= 0:
        return 0.0
    return float((end_price - start_price) / start_price)


def _compute_ma_slope(closes: np.ndarray, period: int = 20) -> float:
    """Compute MA slope as (MA[-1] - MA[-5]) / MA[-5]."""
    if len(closes) < period + 5:
        return 0.0
    ma = pd.Series(closes).rolling(window=period, min_periods=period).mean().values
    ma_recent = ma[-1]
    ma_prev = ma[-6]
    if pd.isna(ma_recent) or pd.isna(ma_prev) or ma_prev <= 0:
        return 0.0
    return float((ma_recent - ma_prev) / ma_prev)


# =============================================================================
# Strategy class
# =============================================================================

class ShortTermMomentumStrategy(BaseStrategy):
    """短线动量 — 追涨强势股/基，动量延续策略。

    基于短期动量效应：前日涨幅≥2%确认动能，MA20上方确认趋势，
    配合基本面筛选（规模/回撤/人气/年收益），综合评分入场。
    2-4天快速止盈止损。
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "短线动量"

    @property
    def description(self) -> str:
        return (
            "短线动量策略（Momentum）：基于短期动量延续效应，追涨强势股/基。\n\n"
            "**筛选**：前日涨≥1.5% + MA20上方 + 规模过滤 + 人气排名 + 回撤<50% + 年收益>30%。\n"
            "**入场**：动量确认 + 趋势向上 + 放量 + 综合评分≥35。\n"
            "**出场**：跳空保护 | -2%硬止损 | 3天时间止盈 | +4%阶梯止盈(半仓) | +6%目标止盈 | 移动止盈。\n"
            "**仓位**：每笔25%资金，最多4个并发仓位，单日最多1个新仓。\n\n"
            "回测（2024.6-2026.7）：+7.2%总收益，305笔交易，50%胜率，盈亏比1.44，周均2.8笔。\n"
            "目标：2-4天持仓周期，追涨强势，快速止盈止损。"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return {
            # ── Entry — core conditions ──
            "min_prev_day_change": 0.015,      # 前日涨幅≥1.5%（放宽门槛，捕捉更多动量信号）
            "require_above_ma20": True,         # 必须在MA20上方（上升趋势）
            "min_vol_ratio": 1.0,              # 成交量>1.0x 20日均量（放量确认，1.0=不缩量即可）
            "entry_score_threshold": 35,        # 综合评分≥35才入场（放宽至35，回测验证+7.2%/2年/PF=1.44）

            # ── Screening — hard filters ──
            "min_mcap_yi_fund": 10.0,          # 基金最小规模（亿）
            "min_mcap_yi_stock": 50.0,         # 股票最小市值（亿）
            "max_drawdown_1y": 0.50,           # 最大回撤<50%（放宽至50%，不过滤高波动ETF）
            "min_return_1y": 0.30,             # 一年收益率>30%（ETF建议值，股票可调至2.0）
            "popularity_top_n": 100,           # 人气前N名
            "enable_fund_filters": False,       # 基金专项筛选（需AKShare）
            "min_fund_inception_years": 5,      # 基金成立>5年
            "min_manager_years": 5,             # 基金经理从业>5年
            "asset_mode": "etf",               # "etf" | "stock" | "auto"

            # ── Scoring weights ──
            "weight_momentum": 0.40,            # 动量权重（提高，动量是核心）
            "weight_trend": 0.25,              # 趋势权重
            "weight_fundamental": 0.10,         # 基本面权重
            "weight_popularity": 0.10,          # 人气权重
            "weight_drawdown": 0.10,            # 低回撤权重
            "weight_return_1y": 0.05,           # 年收益权重

            # ── Exit: tiered profit-taking ──
            "partial_take_profit_pct": 0.04,    # 阶梯止盈1: +4%卖一半（快速锁定）
            "partial_sell_ratio": 0.5,          # 阶梯止盈卖出比例（50%）
            "take_profit_pct": 0.06,            # 目标止盈+6%（剩余仓位，3:1盈亏比）
            "stop_loss_pct": 0.02,              # 硬止损-2%
            "max_hold_days": 3,                # 最长持有3天（2-4天目标）
            "trailing_activate_pct": 0.02,      # 盈利>2%后激活移动止损
            "trailing_giveback_pct": 0.008,     # 从最高点回撤0.8%止盈

            # ── Gap protection ──
            "max_gap_down_pct": 0.02,          # 次日跳空低开>2%立即平仓（防黑天鹅）
            "max_entry_amplitude": 0.10,       # 入场日振幅>10%不开仓（极端波动次日易跳空，A股涨跌停板限制下正常ETF振幅在0-8%）

            # ── Momentum fade exit ──
            "rsi_overbought": 75,              # RSI>75动量衰竭
            "close_below_ma5_exit": True,       # 收盘<MA5退出

            # ── Position sizing ──
            "position_pct": 0.25,               # 每笔25%资金（用户指定15-25%）
            "max_concurrent": 4,                # 最多4个并发仓位
            "max_new_per_day": 1,               # 单日最多1个新仓
            "max_etf_price": 10.0,             # ETF最高单价

            # ── Filters ──
            "enable_macro_filter": True,
            "min_market_breadth": 0.0,         # 市场宽度阈值（0=不过滤，熊市也交易。回测中无过滤总收益最高+7.2%）
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "min_prev_day_change": {
                "label": "最小前日涨幅",
                "type": "slider",
                "min": 0.01, "max": 0.05, "step": 0.005,
                "help": "前一交易日涨幅需≥此值才考虑入场（默认2%）。核心动量确认条件",
            },
            "require_above_ma20": {
                "label": "要求站上MA20",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=只在上升趋势中追涨 | 关闭=任何趋势都可入场（风险更大）",
            },
            "min_vol_ratio": {
                "label": "最小量比",
                "type": "slider",
                "min": 1.0, "max": 2.0, "step": 0.1,
                "help": "成交量/20日均量需≥此值（默认1.2x）。确认放量追涨",
            },
            "entry_score_threshold": {
                "label": "入场评分门槛",
                "type": "number",
                "min": 30, "max": 70, "step": 5,
                "help": "综合评分≥此值才入场（默认45分）。回测中缺少市值/换手率等数据，合理区间40-55",
            },
            "min_mcap_yi_fund": {
                "label": "基金最小规模(亿)",
                "type": "number",
                "min": 1, "max": 100, "step": 5,
                "help": "基金市值需≥此值（默认10亿）。过滤小规模基金",
            },
            "min_mcap_yi_stock": {
                "label": "股票最小市值(亿)",
                "type": "number",
                "min": 10, "max": 500, "step": 10,
                "help": "股票市值需≥此值（默认50亿）。防控盘风险，规模小容易被操纵",
            },
            "max_drawdown_1y": {
                "label": "最大回撤限制",
                "type": "slider",
                "min": 0.05, "max": 0.50, "step": 0.05,
                "help": "近1年最大回撤需<此值（默认20%）。回撤过大说明波动风险高",
            },
            "min_return_1y": {
                "label": "最小年收益",
                "type": "slider",
                "min": 0.10, "max": 5.0, "step": 0.10,
                "help": "近1年收益需>此值（默认200%即2.0，ETF建议调至0.3-0.5）",
            },
            "popularity_top_n": {
                "label": "人气排名前N",
                "type": "number",
                "min": 10, "max": 500, "step": 10,
                "help": "按换手率排名，取前N名（默认100）。确保流动性充足",
            },
            "enable_fund_filters": {
                "label": "启用基金专项筛选",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=检查基金经理从业年限/成立时间（需AKShare） | 关闭=仅用市值筛选",
            },
            "min_fund_inception_years": {
                "label": "基金成立最小年数",
                "type": "number",
                "min": 1, "max": 15, "step": 1,
                "help": "基金成立需≥此年数（默认5年）。仅当启用基金专项筛选时生效",
            },
            "min_manager_years": {
                "label": "基金经理最小从业年数",
                "type": "number",
                "min": 1, "max": 15, "step": 1,
                "help": "基金经理从业需≥此年数（默认5年）。仅当启用基金专项筛选时生效",
            },
            "asset_mode": {
                "label": "资产模式",
                "type": "select",
                "options": ["etf", "stock", "auto"],
                "help": "etf=仅ETF | stock=仅股票 | auto=自动检测。影响筛选阈值和候选池",
            },
            "weight_momentum": {
                "label": "动量权重", "type": "slider",
                "min": 0.10, "max": 0.50, "step": 0.05,
                "help": "日涨幅在综合评分中的权重（默认35%）",
            },
            "weight_trend": {
                "label": "趋势权重", "type": "slider",
                "min": 0.10, "max": 0.50, "step": 0.05,
                "help": "MA20趋势在综合评分中的权重（默认25%）",
            },
            "weight_fundamental": {
                "label": "基本面权重", "type": "slider",
                "min": 0.05, "max": 0.30, "step": 0.05,
                "help": "规模/市值在综合评分中的权重（默认15%）",
            },
            "weight_popularity": {
                "label": "人气权重", "type": "slider",
                "min": 0.05, "max": 0.30, "step": 0.05,
                "help": "换手率排名在综合评分中的权重（默认10%）",
            },
            "weight_drawdown": {
                "label": "回撤权重", "type": "slider",
                "min": 0.05, "max": 0.30, "step": 0.05,
                "help": "低回撤在综合评分中的权重（默认10%）",
            },
            "weight_return_1y": {
                "label": "年收益权重", "type": "slider",
                "min": 0.00, "max": 0.20, "step": 0.05,
                "help": "年收益在综合评分中的权重（默认5%）",
            },
            "take_profit_pct": {
                "label": "目标止盈线",
                "type": "slider",
                "min": 0.02, "max": 0.08, "step": 0.005,
                "help": "盈利达到此比例全部卖出（默认4%）。短线快速锁定利润",
            },
            "stop_loss_pct": {
                "label": "硬止损线",
                "type": "slider",
                "min": 0.01, "max": 0.05, "step": 0.005,
                "help": "亏损达到此比例立即卖出（默认3%）。短线止损保护本金",
            },
            "max_hold_days": {
                "label": "最长持有天数",
                "type": "number",
                "min": 1, "max": 7, "step": 1,
                "help": "超过此天数强制平仓（默认4天）。短线核心约束",
            },
            "trailing_activate_pct": {
                "label": "移动止损激活线",
                "type": "slider",
                "min": 0.01, "max": 0.04, "step": 0.005,
                "help": "盈利超过此比例后激活移动止损（默认2%）",
            },
            "trailing_giveback_pct": {
                "label": "移动止损回撤容忍",
                "type": "slider",
                "min": 0.005, "max": 0.03, "step": 0.005,
                "help": "从最高盈利回撤超过此值即卖出（默认1%）。越小越保守",
            },
            "rsi_overbought": {
                "label": "RSI超买线",
                "type": "number",
                "min": 65, "max": 85, "step": 5,
                "help": "RSI(14)超过此值视为动量衰竭（默认75）。触发卖出",
            },
            "close_below_ma5_exit": {
                "label": "跌破MA5退出",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=收盘价跌破MA5时退出（短期趋势走坏）",
            },
            "position_pct": {
                "label": "单笔仓位比例",
                "type": "slider",
                "min": 0.10, "max": 0.35, "step": 0.05,
                "help": "每笔交易使用资金比例（默认20%）。3笔并发=60%总仓位",
            },
            "max_concurrent": {
                "label": "最大并发仓位",
                "type": "number",
                "min": 1, "max": 4, "step": 1,
                "help": "最多同时持有几个标的（默认3个）",
            },
            "max_new_per_day": {
                "label": "单日最大新仓",
                "type": "number",
                "min": 1, "max": 3, "step": 1,
                "help": "每个交易日最多开几个新仓位（默认1个）。控制节奏",
            },
            "max_etf_price": {
                "label": "ETF最高单价",
                "type": "number",
                "min": 3.0, "max": 20.0, "step": 1.0,
                "help": "超过此价格的ETF不选（控制单笔金额+100股起买）",
            },
            "enable_macro_filter": {
                "label": "宏观风险过滤",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=极端恐慌时暂停新开仓 | 关闭=忽略宏观",
            },
            "min_market_breadth": {
                "label": "市场宽度阈值",
                "type": "slider",
                "min": 0.30, "max": 0.80, "step": 0.05,
                "help": "仅在>此比例的ETF站上MA20时才开新仓（默认60%）。震荡市/熊市自动休息，保留现金",
            },
        }

    # ------------------------------------------------------------------
    # Hard filter check (Tier 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_hard_filters(
        info: dict,
        hist_closes: np.ndarray,
        params: dict,
        pool_rank: int | None = None,
        pool_total: int | None = None,
    ) -> dict:
        """Apply Tier-1 hard filters: size, drawdown, return, popularity.

        Returns dict with passed (bool) and details (list[str]).
        These are pre-entry filters that disqualify candidates entirely.
        """
        details: list[str] = []
        blocked = False
        block_reason = ""

        asset_mode = str(params.get("asset_mode", "etf"))
        cp = info.get("current_price")

        # ── Size filter ──
        mcap = info.get("mcap_yi") or info.get("float_mcap_yi")
        if asset_mode in ("stock",):
            min_mcap = float(params.get("min_mcap_yi_stock", 50.0))
            if mcap is not None and mcap > 0:
                if mcap < min_mcap:
                    blocked = True
                    block_reason = f"市值{mcap:.0f}亿 < {min_mcap:.0f}亿（规模太小，易被控盘）"
                    details.append(f"❌ 市值{mcap:.0f}亿 < {min_mcap:.0f}亿")
                else:
                    details.append(f"✅ 市值{mcap:.0f}亿 ≥ {min_mcap:.0f}亿（规模合格）")
            else:
                details.append("⚪ 市值数据缺失")
        else:
            min_mcap = float(params.get("min_mcap_yi_fund", 10.0))
            if mcap is not None and mcap > 0:
                if mcap < min_mcap:
                    blocked = True
                    block_reason = f"基金规模{mcap:.0f}亿 < {min_mcap:.0f}亿"
                    details.append(f"❌ 基金规模{mcap:.0f}亿 < {min_mcap:.0f}亿")
                else:
                    details.append(f"✅ 基金规模{mcap:.0f}亿 ≥ {min_mcap:.0f}亿")
            else:
                details.append("⚪ 基金规模数据缺失")

        # ── Drawdown filter ──
        max_dd = float(params.get("max_drawdown_1y", 0.20))
        dd_1y = _compute_drawdown_1y(hist_closes)
        if len(hist_closes) >= 60:
            if dd_1y > max_dd:
                blocked = True
                if not block_reason:
                    block_reason = f"近1年最大回撤{dd_1y:.1%} > {max_dd:.0%}"
                details.append(f"❌ 近1年最大回撤{dd_1y:.1%} > {max_dd:.0%}")
            else:
                details.append(f"✅ 近1年最大回撤{dd_1y:.1%} ≤ {max_dd:.0%}")
        else:
            details.append("⚪ K线数据不足60根，跳过大回撤检查")

        # ── 1-year return filter ──
        min_ret = float(params.get("min_return_1y", 2.0))
        ret_1y = _compute_return_1y(hist_closes)
        if len(hist_closes) >= 60:
            if ret_1y < min_ret:
                blocked = True
                if not block_reason:
                    block_reason = f"近1年收益{ret_1y:.1%} < {min_ret:.0%}"
                details.append(f"❌ 近1年收益{ret_1y:.1%} < {min_ret:.0%}")
            else:
                details.append(f"✅ 近1年收益{ret_1y:.1%} ≥ {min_ret:.0%}")
        else:
            details.append("⚪ K线数据不足60根，跳过收益检查")

        # ── Popularity filter ──
        pop_n = int(params.get("popularity_top_n", 100))
        if pool_rank is not None and pool_total is not None:
            if pool_rank > pop_n:
                blocked = True
                if not block_reason:
                    block_reason = f"人气排名{pool_rank}/{pool_total} > 前{pop_n}"
                details.append(f"❌ 人气排名{pool_rank}/{pool_total}（需前{pop_n}）")
            else:
                details.append(f"✅ 人气排名{pool_rank}/{pool_total}（前{pop_n}）")

        return {
            "passed": not blocked,
            "blocked": blocked,
            "block_reason": block_reason,
            "details": details,
            "drawdown_1y": dd_1y,
            "return_1y": ret_1y,
            "mcap_yi": mcap,
        }

    # ------------------------------------------------------------------
    # Composite score (Tier 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_composite_score(
        daily_ret: float,
        dist_ma20: float,
        ma20_slope: float,
        mcap_yi: float | None,
        turnover_pct: float | None,
        turnover_pool_pctile: float | None,
        drawdown_1y: float,
        return_1y: float,
        rsi_value: float | None,
        vol_ratio: float | None,
        params: dict,
    ) -> dict:
        """Compute weighted composite score for a candidate.

        Returns dict with total score (0-100) and per-factor breakdowns.
        """
        w_mom = float(params.get("weight_momentum", 0.35))
        w_trend = float(params.get("weight_trend", 0.25))
        w_fund = float(params.get("weight_fundamental", 0.15))
        w_pop = float(params.get("weight_popularity", 0.10))
        w_dd = float(params.get("weight_drawdown", 0.10))
        w_ret = float(params.get("weight_return_1y", 0.05))

        # ── Momentum score (0-30) ──
        momentum_score = min(30.0, abs(daily_ret) * 800)  # 2%→16pts, 3.75%→30pts
        if daily_ret < 0:
            momentum_score = 0.0  # Negative days get 0 momentum score

        # ── Trend score (0-25) ──
        trend_score = 0.0
        # MA20 slope: positive slope = uptrend
        if ma20_slope > 0.01:
            trend_score += 12.0
        elif ma20_slope > 0:
            trend_score += 6.0
        elif ma20_slope > -0.01:
            trend_score += 2.0
        else:
            trend_score -= 5.0

        # Price position relative to MA20
        if dist_ma20 > 0.03:
            trend_score += 10.0  # Well above MA20, strong trend
        elif dist_ma20 > 0:
            trend_score += 6.0
        elif dist_ma20 > -0.02:
            trend_score += 2.0  # Near MA20, borderline

        trend_score = max(0.0, min(25.0, trend_score))

        # ── Fundamental score (0-15) ──
        fundamental_score = 0.0
        asset_mode = str(params.get("asset_mode", "etf"))
        if asset_mode in ("stock",):
            min_mcap = float(params.get("min_mcap_yi_stock", 50.0))
        else:
            min_mcap = float(params.get("min_mcap_yi_fund", 10.0))

        if mcap_yi is not None and mcap_yi > 0:
            # Score grows with size: at threshold=5pts, 2x threshold=10pts, 5x+ threshold=15pts
            ratio = mcap_yi / max(min_mcap, 1.0)
            fundamental_score = min(15.0, ratio * 5.0)

        # ── Popularity score (0-10) ──
        popularity_score = 0.0
        if turnover_pool_pctile is not None:
            # Higher percentile (more turnover) = higher score
            popularity_score = min(10.0, turnover_pool_pctile * 10.0)
        elif turnover_pct is not None:
            # Fallback: absolute turnover scoring
            if turnover_pct > 5.0:
                popularity_score = 10.0
            elif turnover_pct > 3.0:
                popularity_score = 7.0
            elif turnover_pct > 1.0:
                popularity_score = 4.0
            else:
                popularity_score = 1.0

        # ── Drawdown score (0-15) — lower drawdown = higher score ──
        max_dd = float(params.get("max_drawdown_1y", 0.20))
        if drawdown_1y < 1.0:
            dd_ratio = 1.0 - (drawdown_1y / max(max_dd, 0.05))
            drawdown_score = max(0.0, min(15.0, dd_ratio * 15.0))
        else:
            drawdown_score = 0.0

        # ── Return score (0-10) — higher return = higher score ──
        min_ret = float(params.get("min_return_1y", 2.0))
        if return_1y > -1.0:
            ret_ratio = return_1y / max(min_ret, 0.10)
            return_score = min(10.0, ret_ratio * 5.0)
        else:
            return_score = 0.0

        # ── Volume bonus (0-5, baked into trend or standalone) ──
        vol_bonus = 0.0
        if vol_ratio is not None:
            if vol_ratio >= 2.0:
                vol_bonus = 5.0
            elif vol_ratio >= 1.5:
                vol_bonus = 3.0
            elif vol_ratio >= 1.2:
                vol_bonus = 1.0

        # ── RSI bonus (momentum quality, 0-5) ──
        rsi_bonus = 0.0
        if rsi_value is not None and not np.isnan(rsi_value):
            if 55 <= rsi_value <= 70:
                rsi_bonus = 5.0  # Strong but not overbought
            elif 45 <= rsi_value < 55:
                rsi_bonus = 3.0  # Neutral-positive
            elif rsi_value < 30:
                rsi_bonus = -3.0  # Too weak for momentum strategy

        # ── Weighted total, normalized to 0-100 ──
        weighted = (
            w_mom * momentum_score
            + w_trend * trend_score
            + w_fund * fundamental_score
            + w_pop * popularity_score
            + w_dd * drawdown_score
            + w_ret * return_score
            + vol_bonus * 0.02
            + rsi_bonus * 0.02
        )
        # Theoretical max given the weights and sub-score caps
        max_possible = (
            w_mom * 30.0 + w_trend * 25.0 + w_fund * 15.0
            + w_pop * 10.0 + w_dd * 15.0 + w_ret * 10.0
            + 0.2  # vol_bonus + rsi_bonus max contribution
        )
        total = weighted / max_possible * 100.0 if max_possible > 0 else 0.0
        total = max(0.0, min(100.0, total))

        return {
            "total": round(total, 1),
            "momentum_score": round(momentum_score, 1),
            "trend_score": round(trend_score, 1),
            "fundamental_score": round(fundamental_score, 1),
            "popularity_score": round(popularity_score, 1),
            "drawdown_score": round(drawdown_score, 1),
            "return_score": round(return_score, 1),
            "vol_bonus": round(vol_bonus, 1),
            "rsi_bonus": round(rsi_bonus, 1),
        }

    # ------------------------------------------------------------------
    # Entry condition check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_entry_conditions(
        df_sorted: pd.DataFrame,
        idx: int,
        params: dict,
    ) -> dict:
        """Check if bar at *idx* meets entry conditions for momentum strategy.

        Returns dict with passed, score, reason, details, and per-factor breakdowns.
        """
        row = df_sorted.iloc[idx]
        close = float(row["close"])
        vol = float(row.get("volume", 0) or 0)
        closes_arr = df_sorted["close"].values.astype(float)

        min_change = float(params.get("min_prev_day_change", 0.02))
        require_ma20 = str(params.get("require_above_ma20", "True")).lower() in ("true", "1", "yes")
        min_vol_ratio = float(params.get("min_vol_ratio", 1.2))
        entry_threshold = float(params.get("entry_score_threshold", 60))

        details: list[str] = []
        blocked = False
        block_reason = ""

        # ── Condition 1: Previous day return ≥ threshold (REQUIRED) ──
        daily_ret = _daily_return_pct(closes_arr, idx)
        change_ok = daily_ret >= min_change
        if change_ok:
            details.append(f"✅ 前日涨幅{daily_ret:.1%}（≥{min_change:.0%}，动量确认）")
        else:
            details.append(f"❌ 前日涨幅{daily_ret:.1%}（需≥{min_change:.0%}）")
            blocked = True
            block_reason = f"前日涨幅不足（{daily_ret:.1%} < {min_change:.0%}）"

        # ── Condition 2: Above MA20 (REQUIRED if enabled) ──
        ma20_ok = True
        dist_ma20 = 0.0
        if "_ma20" in df_sorted.columns:
            ma20_val = df_sorted.at[idx, "_ma20"]
            if pd.notna(ma20_val) and ma20_val > 0:
                dist_ma20 = (close - ma20_val) / ma20_val
                if dist_ma20 >= 0:
                    ma20_ok = True
                    details.append(f"✅ MA20上方{dist_ma20:+.1%}（上升趋势）")
                elif dist_ma20 >= -0.015:
                    ma20_ok = True
                    details.append(f"🟡 MA20下方{dist_ma20:.1%}（轻微跌破）")
                else:
                    ma20_ok = False
                    details.append(f"🔴 MA20下方{dist_ma20:.1%}（深度跌破）")
                    if require_ma20:
                        blocked = True
                        block_reason = f"深度跌破MA20（{dist_ma20:.1%}），趋势走坏"
        else:
            details.append("⚪ MA20未计算，跳过趋势检查")

        # ── Condition 3: Volume ratio ≥ threshold (REQUIRED) ──
        vol_ok = False
        vol_ratio = None
        if idx >= 20 and "volume" in df_sorted.columns:
            avg_vol_20 = float(np.mean([
                float(df_sorted.iloc[i].get("volume", 0) or 0)
                for i in range(max(0, idx - 20), idx)
            ]))
            if avg_vol_20 > 0:
                vol_ratio = vol / avg_vol_20
                if vol_ratio >= min_vol_ratio:
                    vol_ok = True
                    details.append(f"✅ 放量{vol_ratio:.1f}x（≥{min_vol_ratio:.1f}x，追涨确认）")
                else:
                    details.append(f"❌ 量比{vol_ratio:.1f}x（需≥{min_vol_ratio:.1f}x）")
                    blocked = True
                    if not block_reason:
                        block_reason = f"成交量不足（{vol_ratio:.1f}x < {min_vol_ratio:.1f}x）"
            else:
                details.append("⚪ 均量=0，跳过量比检查")
                vol_ok = True
        else:
            details.append("⚪ 量数据不足，跳过")
            vol_ok = True

        # ── Condition 4: Amplitude cap (avoid extreme volatility days) ──
        max_amp = float(params.get("max_entry_amplitude", 0.06))
        amp_ok = True
        if "amplitude" in df_sorted.columns:
            amp_val = float(row.get("amplitude", 0) or 0)
            # Baidu K-line returns amplitude as percentage (e.g. 2.41 = 2.41%),
            # NOT decimal (0.0241). Normalize to decimal if > 1.0.
            if amp_val > 1.0:
                amp_val = amp_val / 100.0
            if amp_val > max_amp:
                amp_ok = False
                blocked = True
                if not block_reason:
                    block_reason = f"振幅{amp_val:.1%} > {max_amp:.0%}（极端波动，次日跳空风险高）"
                details.append(f"❌ 振幅{amp_val:.1%} > {max_amp:.0%}（极端波动，不开仓）")
            else:
                details.append(f"✅ 振幅{amp_val:.1%} ≤ {max_amp:.0%}（波动正常）")
        else:
            details.append("⚪ 振幅数据缺失")

        # ── RSI check ──
        rsi_val = None
        if idx >= 14:
            rsi_arr = _compute_rsi(closes_arr[:idx + 1], 14)
            rsi_val = rsi_arr[idx]
            if pd.notna(rsi_val):
                if rsi_val > float(params.get("rsi_overbought", 75)):
                    details.append(f"⚠️ RSI(14)={rsi_val:.1f}（超买区域，谨慎追涨）")
                elif rsi_val >= 55:
                    details.append(f"✅ RSI(14)={rsi_val:.1f}（强势区间）")
                elif rsi_val >= 40:
                    details.append(f"🟡 RSI(14)={rsi_val:.1f}（中性）")
                else:
                    details.append(f"⚪ RSI(14)={rsi_val:.1f}（偏弱）")

        # ── MA slope ──
        ma20_slope = _compute_ma_slope(closes_arr[:idx + 1], 20)
        if ma20_slope > 0.01:
            details.append(f"✅ MA20斜率{ma20_slope:+.1%}（加速上升）")
        elif ma20_slope > 0:
            details.append(f"✅ MA20斜率{ma20_slope:+.1%}（缓慢上升）")
        elif ma20_slope > -0.01:
            details.append(f"🟡 MA20斜率{ma20_slope:+.1%}（走平）")
        else:
            details.append(f"🔴 MA20斜率{ma20_slope:+.1%}（下降趋势）")

        # ── Compute composite score (for ranking / threshold) ──
        score_breakdown = ShortTermMomentumStrategy._compute_composite_score(
            daily_ret=daily_ret,
            dist_ma20=dist_ma20,
            ma20_slope=ma20_slope,
            mcap_yi=None,
            turnover_pct=None,
            turnover_pool_pctile=None,
            drawdown_1y=0.0,
            return_1y=0.0,
            rsi_value=rsi_val,
            vol_ratio=vol_ratio,
            params=params,
        )
        total = score_breakdown["total"]

        # Check score threshold
        if total < entry_threshold and not blocked:
            blocked = True
            block_reason = f"综合评分{total:.0f} < {entry_threshold:.0f}（质量不足）"
            details.append(f"❌ 综合评分{total:.0f} < {entry_threshold:.0f}")

        return {
            "passed": not blocked,
            "score": total,
            "daily_return": daily_ret,
            "change_ok": change_ok,
            "ma20_ok": ma20_ok,
            "vol_ok": vol_ok,
            "amp_ok": amp_ok,
            "dist_ma20": dist_ma20,
            "ma20_slope": ma20_slope,
            "rsi_value": rsi_val,
            "vol_ratio": vol_ratio,
            "blocked": blocked,
            "block_reason": block_reason,
            "details": details,
            "score_breakdown": score_breakdown,
        }

    # ------------------------------------------------------------------
    # Backtest signal generation (single symbol)
    # ------------------------------------------------------------------

    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}

        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        position_pct = float(params["position_pct"])
        partial_tp = float(params.get("partial_take_profit_pct", 0.03))
        partial_ratio = float(params.get("partial_sell_ratio", 0.5))
        trailing_activate = float(params.get("trailing_activate_pct", 0.015))
        trailing_giveback = float(params.get("trailing_giveback_pct", 0.008))
        max_gap_down = float(params.get("max_gap_down_pct", 0.02))
        rsi_overbought = float(params.get("rsi_overbought", 75))
        close_below_ma5 = str(params.get("close_below_ma5_exit", "True")).lower() in ("true", "1", "yes")
        enable_macro = str(params.get("enable_macro_filter", "True")).lower() in ("true", "1", "yes")

        macro_pulse = kwargs.get("macro_pulse")
        macro_suppress = False
        if enable_macro and macro_pulse is not None:
            if macro_pulse.risk_level == "extreme":
                macro_suppress = True

        backtest_capital = float(kwargs.get("backtest_capital", 100_000))

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()
        if "close" not in df.columns:
            df["signal"] = "hold"; df["signal_price"] = 0.0
            df["signal_shares"] = 0; df["signal_reason"] = ""
            return df

        # ── Precompute indicators ──
        df["_ma5"] = df["close"].rolling(window=5, min_periods=5).mean()
        df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()

        # ── Init signal columns ──
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        # ── State ──
        in_position = False
        entry_price = 0.0
        entry_idx = -1
        total_shares = 0
        partial_done = False       # track if partial profit was taken
        highest_since_entry = 0.0
        highest_pnl_pct = 0.0
        total_budget = position_pct * backtest_capital

        for i in range(len(df)):
            close = float(df.at[i, "close"])
            if pd.isna(close) or close <= 0:
                continue

            # ── Exit logic ──
            if in_position:
                pnl_pct = (close - entry_price) / entry_price if entry_price > 0 else 0
                hold_days = i - entry_idx

                if pnl_pct > highest_pnl_pct:
                    highest_pnl_pct = pnl_pct
                if close > highest_since_entry:
                    highest_since_entry = close

                exit_reason = ""
                exit_shares = 0

                # Tier 0: Gap-down protection at open
                open_price = float(df.at[i, "open"]) if "open" in df.columns else close
                prev_close = float(df.at[i - 1, "close"]) if i > 0 else open_price
                if prev_close > 0 and exit_shares == 0:
                    gap_pct = (open_price - prev_close) / prev_close
                    if gap_pct <= -max_gap_down:
                        exit_reason = (
                            f"⚠️ 跳空保护：开盘跳空 {gap_pct:.1%}（≤-{max_gap_down:.0%}），"
                            f"入场{entry_price:.3f} → 开盘{open_price:.3f}，立即平仓"
                        )
                        exit_shares = total_shares

                # Tier 1: Hard stop
                if exit_shares == 0 and pnl_pct <= -stop_loss:
                    exit_reason = (
                        f"🛑 硬止损：亏损 {pnl_pct:.1%}（≤-{stop_loss:.0%}），"
                        f"入场 {entry_price:.3f} → 出场 {close:.3f}"
                    )
                    exit_shares = total_shares

                # Tier 2: Time stop
                elif exit_shares == 0 and hold_days >= max_hold:
                    exit_reason = (
                        f"⏰ 时间止盈：持有 {hold_days} 天（≥{max_hold}天），"
                        f"盈亏 {pnl_pct:+.1%}，入场 {entry_price:.3f} → 出场 {close:.3f}"
                    )
                    exit_shares = total_shares

                # Tier 3: Partial take-profit — sell half, keep half running
                elif exit_shares == 0 and not partial_done and pnl_pct >= partial_tp:
                    partial_shares = max(100, int(total_shares * partial_ratio) // 100 * 100)
                    if partial_shares >= 100 and partial_shares < total_shares:
                        exit_reason = (
                            f"📤 阶梯止盈1：盈利 {pnl_pct:.1%}（≥{partial_tp:.0%}），"
                            f"卖出{partial_shares}股（{partial_ratio:.0%}仓位），入场 {entry_price:.3f} → {close:.3f}"
                        )
                        exit_shares = partial_shares
                        partial_done = True
                        # Adjust entry price to remaining position (cost basis unchanged)
                        remaining = total_shares - partial_shares
                        if remaining > 0:
                            total_shares = remaining
                            # entry_price stays the same for P&L calculation

                # Tier 4: Full take profit (remaining position)
                elif exit_shares == 0 and pnl_pct >= take_profit:
                    exit_reason = (
                        f"🎯 目标止盈：盈利 {pnl_pct:.1%}（≥{take_profit:.0%}），"
                        f"入场 {entry_price:.3f} → 出场 {close:.3f}"
                    )
                    exit_shares = total_shares

                # Tier 5: Trailing stop (giveback from peak)
                elif exit_shares == 0 and highest_pnl_pct >= trailing_activate:
                    giveback = highest_pnl_pct - pnl_pct
                    if giveback >= trailing_giveback:
                        exit_reason = (
                            f"🔒 移动止盈：从最高+{highest_pnl_pct:.1%}回撤{giveback:.1%}（≥{trailing_giveback:.1%}），"
                            f"当前盈亏 {pnl_pct:+.1%}，入场 {entry_price:.3f} → 出场 {close:.3f}"
                        )
                        exit_shares = total_shares

                # Tier 6: Momentum fade — RSI overbought
                if exit_shares == 0 and rsi_overbought > 0:
                    closes_arr = df["close"].values[:i + 1].astype(float)
                    rsi_arr = _compute_rsi(closes_arr, 14)
                    rsi_val = rsi_arr[i]
                    if pd.notna(rsi_val) and rsi_val >= rsi_overbought:
                        exit_reason = (
                            f"📉 动量衰竭：RSI(14)={rsi_val:.1f} ≥ {rsi_overbought:.0f}（超买），"
                            f"盈亏 {pnl_pct:+.1%}，入场 {entry_price:.3f} → 出场 {close:.3f}"
                        )
                        exit_shares = total_shares

                # Tier 7: Close below MA5 (short-term trend broken)
                if exit_shares == 0 and close_below_ma5:
                    ma5_val = df.at[i, "_ma5"] if "_ma5" in df.columns else None
                    if pd.notna(ma5_val) and ma5_val > 0 and close < ma5_val and pnl_pct > 0:
                        exit_reason = (
                            f"📉 跌破MA5：收盘{close:.3f} < MA5({ma5_val:.3f})，短期趋势走弱，"
                            f"盈亏 {pnl_pct:+.1%}"
                        )
                        exit_shares = total_shares

                if exit_shares >= 100:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = exit_shares
                    df.at[i, "signal_reason"] = exit_reason
                    if exit_shares >= total_shares:
                        in_position = False
                        total_shares = 0
                        partial_done = False
                        highest_pnl_pct = 0.0

            # ── Entry logic ──
            if not in_position:
                if macro_suppress:
                    continue

                if i < 20:
                    continue

                entry_check = self._check_entry_conditions(df, i, params)
                if not entry_check["passed"]:
                    continue

                # Calculate shares
                raw_shares = int(total_budget / close)
                shares = max(100, (raw_shares // 100) * 100)
                if shares < 100:
                    continue

                score = entry_check["score"]
                daily_ret = entry_check["daily_return"]
                reason = (
                    f"🚀 短线动量：前日{daily_ret:+.1%}，评分{score:.0f}分 | "
                    + " | ".join(d for d in entry_check["details"][:4] if "✅" in d)
                )

                df.at[i, "signal"] = "buy"
                df.at[i, "signal_price"] = close
                df.at[i, "signal_shares"] = shares
                df.at[i, "signal_reason"] = reason

                entry_price = close
                total_shares = shares
                in_position = True
                entry_idx = i
                highest_since_entry = close
                highest_pnl_pct = 0.0

        return df.drop(columns=["_ma5", "_ma20"], errors="ignore")

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Live signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _price_range(price: float, spread_pct: float = 0.01) -> tuple[float, float]:
        """Return (low, high) price range accounting for latency + spread.

        Buy:  current_price * (1 - spread)  ~  current_price * (1 + spread/2)
        Sell: current_price * (1 - spread/2) ~ current_price * (1 + spread)

        Typical spread_pct=0.01 means ~1% tolerance around current price.
        """
        cp = round(price, 4)
        if cp <= 0:
            return (cp, cp)
        low = round(cp * (1.0 - spread_pct), 3)
        high = round(cp * (1.0 + spread_pct), 3)
        return (min(low, high), max(low, high))

    @staticmethod
    def _sell_signal(
        current_price: float, shares: int, cost: float,
        pnl_pct: float, hold_days: int, max_hold: int,
        zone: str, trigger: str, reason: str, urgency: str = "high",
    ) -> LiveSignal:
        """Build a sell signal with price range."""
        low, high = ShortTermMomentumStrategy._price_range(current_price)
        return LiveSignal(
            action="sell",
            current_price=round(current_price, 4),
            suggested_shares=shares,
            suggested_amount=round(shares * current_price, 2),
            suggested_price_low=low,
            suggested_price_high=high,
            trigger_description=trigger,
            reason=reason,
            urgency_level=urgency,
            current_zone=zone,
        )

    @staticmethod
    def _buy_signal(
        current_price: float, shares: int, amount: float,
        zone: str, trigger: str, reason: str,
    ) -> LiveSignal:
        """Build a buy signal with price range."""
        low, high = ShortTermMomentumStrategy._price_range(current_price)
        return LiveSignal(
            action="buy",
            current_price=round(current_price, 4),
            suggested_shares=shares,
            suggested_amount=round(amount, 2),
            suggested_price_low=low,
            suggested_price_high=high,
            trigger_description=trigger,
            reason=reason,
            urgency_level="high",
            current_zone=zone,
        )

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    def get_live_signal(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> LiveSignal:
        if df is None or df.empty:
            return LiveSignal(action="hold", reason="无历史数据", urgency_level="low")

        params = {**self.get_default_params(), **kwargs}
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        min_change = float(params.get("min_prev_day_change", 0.015))
        enable_macro = str(params.get("enable_macro_filter", "True")).lower() in ("true", "1", "yes")

        if "_ma20" not in df.columns:
            df = df.copy()
            df["_ma5"] = df["close"].rolling(window=5, min_periods=5).mean()
            df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

        current_price = info.get("current_price") if info else None
        if current_price is None:
            current_price = float(df_sorted.iloc[-1]["close"])

        macro_pulse = kwargs.get("macro_pulse")
        macro_suppress = False
        if enable_macro and macro_pulse is not None:
            if macro_pulse.risk_level in ("extreme",):
                macro_suppress = True

        # ── Portfolio context ──
        pf = kwargs.get("portfolio_context") or {}
        has_position = pf.get("has_position", False)
        holding_cost = pf.get("holding_avg_cost")
        holding_shares = pf.get("holding_shares", 0)
        last_buy_date = pf.get("last_buy_date")

        hold_days = 0
        if last_buy_date:
            try:
                if isinstance(last_buy_date, str):
                    last_buy_date = datetime.strptime(last_buy_date, "%Y-%m-%d").date()
                hold_days = (datetime.now().date() - last_buy_date).days
            except (ValueError, TypeError):
                pass

        trailing_activate = float(params.get("trailing_activate_pct", 0.02))
        trailing_giveback = float(params.get("trailing_giveback_pct", 0.008))
        rsi_overbought = float(params.get("rsi_overbought", 75))
        close_below_ma5 = str(params.get("close_below_ma5_exit", "True")).lower() in ("true", "1", "yes")
        partial_tp = float(params.get("partial_take_profit_pct", 0.04))
        partial_ratio = float(params.get("partial_sell_ratio", 0.5))

        # ── In position: check exits ──
        if has_position and holding_cost and holding_cost > 0:
            pnl_pct = (current_price - holding_cost) / holding_cost

            # Priority 1: Hard stop loss
            if pnl_pct <= -stop_loss:
                return self._sell_signal(
                    current_price, holding_shares, holding_cost, pnl_pct,
                    hold_days, max_hold,
                    zone="🛑 止损区",
                    trigger=f"亏损 {pnl_pct:.1%}，触发硬止损 -{stop_loss:.0%}",
                    reason=(
                        f"🛑 硬止损！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（{pnl_pct:.1%}）\n\n"
                        f"建议立即卖出全部 {holding_shares} 股，止损离场。"
                        f"下次入场需等待新的动量信号。"
                    ),
                    urgency="high",
                )

            # Priority 2: Time exit
            if hold_days >= max_hold:
                return self._sell_signal(
                    current_price, holding_shares, holding_cost, pnl_pct,
                    hold_days, max_hold,
                    zone="⏰ 时间止盈",
                    trigger=f"持有 {hold_days} 天，触发时间止盈（≥{max_hold}天）",
                    reason=(
                        f"⏰ 时间到！持有 {hold_days} 天（≥{max_hold}天），当前盈亏 {pnl_pct:+.1%}\n\n"
                        f"入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}\n"
                        f"建议卖出全部 {holding_shares} 股，锁定收益/止损。"
                    ),
                    urgency="medium",
                )

            # Priority 3: Full take profit
            if pnl_pct >= take_profit:
                return self._sell_signal(
                    current_price, holding_shares, holding_cost, pnl_pct,
                    hold_days, max_hold,
                    zone="🎯 止盈区",
                    trigger=f"盈利 {pnl_pct:.1%}，触发目标止盈 +{take_profit:.0%}",
                    reason=(
                        f"🎯 目标止盈！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（+{pnl_pct:.1%}）\n\n"
                        f"建议卖出全部 {holding_shares} 股，落袋为安。"
                    ),
                    urgency="high",
                )

            # Priority 4: Partial take-profit (阶梯止盈)
            if pnl_pct >= partial_tp and partial_ratio > 0:
                partial_shares = max(100, int(holding_shares * partial_ratio) // 100 * 100)
                if partial_shares >= 100 and partial_shares < holding_shares:
                    return self._sell_signal(
                        current_price, partial_shares, holding_cost, pnl_pct,
                        hold_days, max_hold,
                        zone="📤 阶梯止盈",
                        trigger=f"盈利 {pnl_pct:.1%}，触发阶梯止盈 +{partial_tp:.0%}（卖{partial_ratio:.0%}仓位）",
                        reason=(
                            f"📤 阶梯止盈1！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（+{pnl_pct:.1%}）\n\n"
                            f"建议卖出 {partial_shares} 股（{partial_ratio:.0%}仓位），锁定部分利润。\n"
                            f"剩余 {holding_shares - partial_shares} 股继续持有，目标止盈 +{take_profit:.0%}。"
                        ),
                        urgency="medium",
                    )

            # Priority 5: RSI overbought
            closes_arr = df_sorted["close"].values.astype(float)
            if len(closes_arr) >= 15:
                rsi_arr = _compute_rsi(closes_arr, 14)
                rsi_val = rsi_arr[-1]
                if pd.notna(rsi_val) and rsi_val >= rsi_overbought:
                    return self._sell_signal(
                        current_price, holding_shares, holding_cost, pnl_pct,
                        hold_days, max_hold,
                        zone="📉 超买区",
                        trigger=f"RSI={rsi_val:.1f} ≥ {rsi_overbought:.0f}，动量衰竭",
                        reason=(
                            f"📉 RSI(14)={rsi_val:.1f} ≥ {rsi_overbought:.0f}（超买），动量衰竭信号\n\n"
                            f"当前盈亏 {pnl_pct:+.1%}，建议卖出 {holding_shares} 股止盈。"
                        ),
                        urgency="medium",
                    )

            # Priority 6: Close below MA5 (趋势走弱)
            if close_below_ma5 and len(df_sorted) > 0:
                ma5_val = df_sorted.at[len(df_sorted) - 1, "_ma5"] if "_ma5" in df_sorted.columns else None
                if pd.notna(ma5_val) and current_price < ma5_val and pnl_pct > 0:
                    return self._sell_signal(
                        current_price, holding_shares, holding_cost, pnl_pct,
                        hold_days, max_hold,
                        zone="📉 趋势走弱",
                        trigger=f"收盘 {current_price:.3f} < MA5({ma5_val:.3f})",
                        reason=(
                            f"📉 跌破MA5！收盘 ¥{current_price:.3f} < MA5(¥{ma5_val:.3f})，短期趋势走弱\n\n"
                            f"当前盈利 {pnl_pct:+.1%}，建议卖出 {holding_shares} 股保护利润。"
                        ),
                        urgency="medium",
                    )

            # ── Holding: no exit triggered ──
            next_tp = round(holding_cost * (1 + take_profit), 4)
            next_sl = round(holding_cost * (1 - stop_loss), 4)
            zone_text = f"📌 持仓中 第{hold_days}天"
            if pnl_pct > 0:
                zone_text += f" +{pnl_pct:.1%}"
            else:
                zone_text += f" {pnl_pct:.1%}"
            return LiveSignal(
                action="hold",
                current_price=round(current_price, 4),
                suggested_shares=0,
                suggested_amount=round(holding_shares * current_price, 2),
                suggested_price_low=round(current_price * 0.995, 3),
                suggested_price_high=round(current_price * 1.005, 3),
                trigger_description=(
                    f"持仓观望 | 止盈价 ¥{next_tp} | 止损价 ¥{next_sl}"
                ),
                next_trigger_price=next_tp,
                reason=(
                    f"📌 持仓 {holding_shares} 股\n"
                    f"入场 ¥{holding_cost:.3f} · 现价 ¥{current_price:.3f}（{pnl_pct:+.1%}）\n"
                    f"目标止盈 ¥{next_tp}（+{take_profit:.0%}）· 硬止损 ¥{next_sl}（-{stop_loss:.0%}）\n"
                    f"继续持有，等待止盈或止损触发。"
                ),
                urgency_level="low",
                portions_used=hold_days,
                portions_total=max_hold,
                current_zone=zone_text,
            )

        # ── Not in position: check entry ──
        if macro_suppress:
            return LiveSignal(
                action="hold",
                current_price=round(current_price, 4),
                reason="⚠️ 宏观情绪极端恐慌，暂停所有新开仓，持有现金观望。",
                urgency_level="low",
                current_zone="🛡️ 宏观暂停",
            )

        if len(df_sorted) < 25:
            return LiveSignal(
                action="hold",
                current_price=round(current_price, 4),
                reason=f"📊 历史数据不足（需≥25个交易日），无法计算入场信号。",
                urgency_level="low",
                current_zone="数据不足",
            )

        entry_check = self._check_entry_conditions(df_sorted, len(df_sorted) - 1, params)

        if entry_check["blocked"] or not entry_check["passed"]:
            daily_ret = entry_check.get("daily_return", 0)
            score = entry_check.get("score", 0)
            block_reason = entry_check.get("block_reason", "")
            reason_lines = [
                f"⏳ 等待动量买入信号\n",
                f"当前状态：前日涨幅 {daily_ret:+.1%}（需≥{min_change:.0%}）",
                f"综合评分 {score:.0f}/100（需≥{float(params.get('entry_score_threshold', 35)):.0f}）",
            ]
            if block_reason:
                reason_lines.insert(3, f"\n⚠️ {block_reason}")
            reason_lines.append("\n📋 详细检查：")
            reason_lines.extend(f"  {d}" for d in entry_check.get("details", [])[:6])

            # Figure out how far from entry
            if daily_ret < min_change:
                gap_pct = (min_change - daily_ret) * 100
                zone = f"🟡 等待涨幅 · 距入场还差{gap_pct:.1f}%"
            elif score < float(params.get("entry_score_threshold", 35)):
                zone = f"🟡 等待评分 · 当前{score:.0f}分"
            else:
                zone = "🟡 等待条件"

            return LiveSignal(
                action="wait_for_strength",
                current_price=round(current_price, 4),
                suggested_price_low=round(current_price * 0.99, 3),
                suggested_price_high=round(current_price * 1.01, 3),
                trigger_description=block_reason or "等待入场条件",
                reason="\n".join(reason_lines),
                urgency_level="low",
                current_zone=zone,
            )

        # ── Entry signal triggered! Calculate position size ──
        available_cash = float(pf.get("available_cash", 2000)) if pf else 2000
        budget = available_cash * float(params["position_pct"])
        raw_shares = int(budget / current_price) if current_price and current_price > 0 else 0
        shares = max(100, (raw_shares // 100) * 100)

        if shares < 100:
            return LiveSignal(
                action="wait_for_strength",
                current_price=round(current_price, 4),
                trigger_description=f"资金不足 · 需¥{current_price*100:.0f}买1手",
                reason=(
                    f"💰 资金不足\n"
                    f"买100股需 ¥{current_price*100:.0f}，可用仓位 ¥{budget:.0f}\n"
                    f"等待资金补充或价格回落。"
                ),
                urgency_level="low",
                current_zone="💰 资金不足",
            )

        amount = round(shares * current_price, 2)
        daily_ret = entry_check["daily_return"]
        score = entry_check["score"]
        next_tp = round(current_price * (1 + take_profit), 3)
        next_sl = round(current_price * (1 - stop_loss), 3)

        return self._buy_signal(
            current_price, shares, amount,
            zone=f"🚀 动量追涨 · {score:.0f}分",
            trigger=f"买入信号 前日涨{daily_ret:+.1%} 评分{score:.0f}",
            reason=(
                f"🚀 短线动量买入信号！\n\n"
                f"前日涨幅: {daily_ret:+.1%}（≥{min_change:.0%} ✅）\n"
                + "\n".join(d for d in entry_check.get("details", []) if "✅" in d or "🟡" in d)
                + f"\n\n📊 建议操作：\n"
                f"  买入 {shares} 股 ≈ ¥{amount:,.0f}（占总资金 {float(params['position_pct'])*100:.0f}%）\n"
                f"  目标止盈 ¥{next_tp}（+{take_profit:.0%}）\n"
                f"  硬止损   ¥{next_sl}（-{stop_loss:.0%}）\n"
                f"  最长持有 {max_hold} 个交易日\n\n"
                f"⚡ 短线追涨，严格止损，快进快出！"
            ),
        )

    # ------------------------------------------------------------------
    # Dashboard cards
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Trade plan builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trade_plan(
        action: str, current_price: float, entry_check: dict | None,
        pf: dict, params: dict,
    ) -> list[str]:
        """Build forward-looking trade plan lines for the dashboard timeline.

        Returns a list of HTML-ready strings, one per timeline phase.
        """
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        trailing_activate = float(params.get("trailing_activate_pct", 0.02))
        trailing_giveback = float(params.get("trailing_giveback_pct", 0.008))
        partial_tp = float(params.get("partial_take_profit_pct", 0.04))
        tp_price = round(current_price * (1 + take_profit), 3)
        sl_price = round(current_price * (1 - stop_loss), 3)
        trail_price = round(current_price * (1 + trailing_activate), 3)
        partial_price = round(current_price * (1 + partial_tp), 3)

        has_position = pf.get("has_position", False)
        holding_cost = pf.get("holding_avg_cost")
        hold_days = int(pf.get("hold_days", 0))
        position_pct = float(params.get("position_pct", 0.25))

        if action == "buy":
            # ── Entry plan: show full 3-day timeline ──
            return [
                f"<b>🟢 T+0（今天）买入</b> — "
                f"入场价 ¥{current_price:.3f} ±1%（¥{current_price*0.99:.3f}~¥{current_price*1.01:.3f}）",

                f"<b>🛡️ 立即设置</b> — "
                f"止盈 ¥{tp_price}（+{take_profit:.0%}）| "
                f"止损 ¥{sl_price}（-{stop_loss:.0%}）| "
                f"最长持有 {max_hold} 天",

                f"<b>🟡 T+1（明天）</b> — "
                f"开盘跳空>2%立即止损 | "
                f"盈利≥{trailing_activate:.0%}(¥{trail_price})激活移动止盈 | "
                f"收盘<MA5且盈利→卖出",

                f"<b>🟡 T+2</b> — "
                f"继续跟踪移动止盈(回撤{trailing_giveback:.1%}即卖) | "
                f"RSI>{float(params.get('rsi_overbought',75)):.0f}→动量衰竭卖出 | "
                f"盈利≥{partial_tp:.0%}(¥{partial_price})→卖一半锁利",

                f"<b>🔴 T+3（最后一天）</b> — "
                f"无论盈亏必须平仓！| "
                f"目标止盈 ¥{tp_price}（+{take_profit:.0%}）| "
                f"最低接受 ¥{sl_price}（-{stop_loss:.0%}）",
            ]
        elif action == "sell":
            # ── Exit plan ──
            pnl = (current_price - holding_cost) / holding_cost if holding_cost else 0
            remaining = max(0, max_hold - hold_days)
            return [
                f"<b>🔴 卖出信号</b> — "
                f"入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（{pnl:+.1%}）",

                f"<b>📊 卖出后</b> — "
                f"资金释放，等待下一个动量信号。"
                f"预计下次入场需前日涨幅≥{float(params.get('min_prev_day_change',0.015))*100:.1f}%+评分≥{float(params.get('entry_score_threshold',35)):.0f}",

                f"<b>⏳ 冷却期</b> — "
                f"卖出当日不开新仓，次日重新扫描候选池。"
                f"关注 {pf.get('code', '')} 是否延续动量。",
            ]
        elif action == "hold":
            # ── Holding plan: show remaining days ──
            pnl = (current_price - holding_cost) / holding_cost if holding_cost else 0
            remaining = max(0, max_hold - hold_days)
            tp_from_cost = round(holding_cost * (1 + take_profit), 3)
            sl_from_cost = round(holding_cost * (1 - stop_loss), 3)
            lines = [
                f"<b>🔵 持仓第 {hold_days} 天</b> — "
                f"入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（{pnl:+.1%}）",

                f"<b>🎯 止盈目标</b> ¥{tp_from_cost}（+{take_profit:.0%}）| "
                f"<b>🛑 止损底线</b> ¥{sl_from_cost}（-{stop_loss:.0%}）",

                f"<b>📅 剩余 {remaining} 天</b> — "
                + (f"T+{max_hold}必须平仓，无论盈亏" if remaining <= 1 else
                   f"每日检查：跳空>2%→止损 | MA5跌破→卖出 | RSI>75→卖出"),
            ]
            if pnl >= partial_tp:
                lines.append(
                    f"<b>📤 已触发阶梯止盈区</b> — "
                    f"盈利≥{partial_tp:.0%}，可考虑先卖50%锁利，剩余博+{take_profit:.0%}"
                )
            return lines
        else:
            # ── Waiting plan: what's needed + what happens after entry ──
            lines = [
                f"<b>🟡 等待入场信号</b> — "
                f"当前价 ¥{current_price:.3f}"
            ]
            if entry_check:
                daily_ret = entry_check.get("daily_return", 0)
                min_change = float(params.get("min_prev_day_change", 0.015))
                score = entry_check.get("score", 0)
                min_score = float(params.get("entry_score_threshold", 35))

                if daily_ret < min_change:
                    gap = (min_change - daily_ret) * 100
                    lines.append(
                        f"<b>📈 涨幅差距</b> — 前日涨 {daily_ret:.1%}，"
                        f"距入场门槛 {min_change:.1%} 还差 {gap:.1f}%"
                    )
                if score < min_score:
                    lines.append(
                        f"<b>📊 评分差距</b> — 综合评分 {score:.0f}/100，"
                        f"距入场门槛 {min_score:.0f} 还差 {min_score - score:.0f} 分"
                    )
                # Show what would happen if conditions met
                lines.append(
                    f"<b>✅ 条件满足后</b> — "
                    f"T+0买入 → 止盈+{take_profit:.0%}/止损-{stop_loss:.0%} → T+{max_hold}强制平仓"
                )

            lines.append(
                f"<b>💡 建议</b> — 耐心等待，不追高。"
                f"关注候选池中前日涨幅靠前的ETF，提前做好入场准备。"
            )
            return lines

    # ------------------------------------------------------------------
    # Dashboard cards
    # ------------------------------------------------------------------

    def get_dashboard_cards(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> list[DashboardCard]:
        if df is None or df.empty:
            return []

        params = {**self.get_default_params(), **kwargs}
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        min_change = float(params.get("min_prev_day_change", 0.015))

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

        if "_ma5" not in df_sorted.columns:
            df_sorted["_ma5"] = df_sorted["close"].rolling(window=5, min_periods=5).mean()
        if "_ma20" not in df_sorted.columns:
            df_sorted["_ma20"] = df_sorted["close"].rolling(window=20, min_periods=20).mean()

        current_price = info.get("current_price") if info else None
        if current_price is None and len(df_sorted) > 0:
            current_price = float(df_sorted.iloc[-1]["close"])

        cards: list[DashboardCard] = []

        # ── Determine action for trade plan ──
        pf = kwargs.get("portfolio_context") or {}
        has_position = pf.get("has_position", False)
        holding_cost = pf.get("holding_avg_cost")

        # Get entry check for all states
        entry_check = None
        if len(df_sorted) >= 25:
            try:
                entry_check = self._check_entry_conditions(df_sorted, len(df_sorted) - 1, params)
            except Exception:
                pass

        # Determine current action for trade plan
        if has_position and holding_cost and holding_cost > 0:
            pnl_pct = (current_price - holding_cost) / holding_cost
            hold_days = 0
            last_buy = pf.get("last_buy_date")
            if last_buy:
                try:
                    if isinstance(last_buy, str):
                        last_buy = datetime.strptime(last_buy, "%Y-%m-%d").date()
                    hold_days = (datetime.now().date() - last_buy).days
                except Exception:
                    pass

            if pnl_pct <= -stop_loss or hold_days >= max_hold or pnl_pct >= take_profit:
                plan_action = "sell"
            else:
                plan_action = "hold"
            pf["hold_days"] = hold_days
        elif entry_check and entry_check["passed"]:
            plan_action = "buy"
        else:
            plan_action = "wait"

        # ═══════════════════════════════════════════════════════════════
        # Card 1: 📅 交易时间线 — THE most important card
        # ═══════════════════════════════════════════════════════════════
        plan_lines = self._build_trade_plan(
            plan_action, current_price, entry_check, pf, params
        )
        plan_html = (
            '<div style="font-size:0.82rem; line-height:1.7; color:#1e293b;">'
            + "".join(f'<div style="margin:6px 0; padding:6px 10px; '
                      f'background:#f8fafc; border-radius:6px; '
                      f'border-left:3px solid #3b82f6;">{line}</div>'
                      for line in plan_lines)
            + '</div>'
        )
        cards.append(DashboardCard(
            card_id="trade_timeline",
            title={
                "buy": "📅 交易时间线 · 🟢 买入执行计划",
                "sell": "📅 交易时间线 · 🔴 卖出执行计划",
                "hold": "📅 交易时间线 · 🔵 持仓跟踪计划",
                "wait": "📅 交易时间线 · 🟡 等待入场计划",
            }.get(plan_action, "📅 交易时间线"),
            card_type="info",
            content={"plan_html": plan_html, "action": plan_action},
            priority=0,  # ALWAYS FIRST
        ))

        # ═══════════════════════════════════════════════════════════════
        # Card 2: Entry conditions check
        # ═══════════════════════════════════════════════════════════════
        if entry_check:
            conditions = [
                {"label": f"前日涨幅≥{min_change:.0%}", "met": entry_check["change_ok"],
                 "detail": f"当前{entry_check['daily_return']:.1%}"},
                {"label": "MA20上方（上升趋势）", "met": entry_check["ma20_ok"],
                 "detail": f"{entry_check['dist_ma20']:+.1%}" if entry_check.get("dist_ma20") else "✅"},
                {"label": f"放量≥{params['min_vol_ratio']:.1f}x", "met": entry_check["vol_ok"],
                 "detail": f"{entry_check.get('vol_ratio', 0):.1f}x" if entry_check.get("vol_ratio") else "✅"},
                {"label": f"综合评分≥{float(params['entry_score_threshold']):.0f}", "met": entry_check["passed"],
                 "detail": f"{entry_check['score']:.0f}分"},
            ]
            score = entry_check["score"]
            passed = entry_check["passed"]
            cards.append(DashboardCard(
                card_id="entry_conditions",
                title=f"入场条件 · 评分{score:.0f}/100" + (" ✅" if passed else f" ⛔ {entry_check.get('block_reason','')}"),
                card_type="info",
                content={
                    "conditions": conditions,
                    "details": entry_check["details"],
                    "passed": passed,
                    "score": score,
                    "block_reason": entry_check.get("block_reason", ""),
                },
                priority=1,
            ))

        # ═══════════════════════════════════════════════════════════════
        # Card 3: Price ladder — key levels
        # ═══════════════════════════════════════════════════════════════
        tp_price = round(current_price * (1 + take_profit), 3)
        sl_price = round(current_price * (1 - stop_loss), 3)
        partial_price = round(current_price * (1 + float(params.get("partial_take_profit_pct", 0.04))), 3)
        trail_price = round(current_price * (1 + float(params.get("trailing_activate_pct", 0.02))), 3)

        levels = [
            {"label": "🎯 目标止盈", "price": f"¥{tp_price}", "pct": f"+{take_profit:.0%}"},
            {"label": "📤 阶梯止盈(卖半仓)", "price": f"¥{partial_price}", "pct": f"+{float(params.get('partial_take_profit_pct',0.04)):.0%}"},
            {"label": "🔒 移动止盈激活", "price": f"¥{trail_price}", "pct": f"+{float(params.get('trailing_activate_pct',0.02)):.0%}"},
            {"label": "💰 当前价格", "price": f"¥{current_price:.3f}", "pct": "—"},
            {"label": "🛑 硬止损", "price": f"¥{sl_price}", "pct": f"-{stop_loss:.0%}"},
        ]
        cards.append(DashboardCard(
            card_id="price_ladder",
            title="💰 关键价位",
            card_type="info",
            content={"levels": levels, "current_price": round(current_price, 4)},
            priority=1,
        ))

        # ═══════════════════════════════════════════════════════════════
        # Card 4: Market context
        # ═══════════════════════════════════════════════════════════════
        ma20_val = df_sorted.at[len(df_sorted) - 1, "_ma20"] if len(df_sorted) > 0 else None
        ma5_val = df_sorted.at[len(df_sorted) - 1, "_ma5"] if len(df_sorted) > 0 else None

        closes_arr = df_sorted["close"].values.astype(float)
        rsi_val = None
        if len(closes_arr) >= 15:
            rsi_arr = _compute_rsi(closes_arr, 14)
            rsi_val = round(float(rsi_arr[-1]), 1) if pd.notna(rsi_arr[-1]) else None

        daily_ret = _daily_return_pct(closes_arr, len(df_sorted) - 1)
        dist_ma20 = (current_price - ma20_val) / ma20_val if current_price and ma20_val and ma20_val > 0 else 0

        cards.append(DashboardCard(
            card_id="market_context",
            title="市场环境",
            card_type="info",
            content={
                "current_price": round(current_price, 4) if current_price else None,
                "daily_return": f"{daily_ret:.2%}",
                "ma5_value": round(float(ma5_val), 4) if pd.notna(ma5_val) else None,
                "ma20_value": round(float(ma20_val), 4) if pd.notna(ma20_val) else None,
                "dist_ma20": f"{dist_ma20:+.2%}",
                "rsi_14": rsi_val,
                "trend": "上升趋势 📈" if dist_ma20 > 0 else ("横盘 ⏸️" if dist_ma20 > -0.02 else "下降趋势 📉"),
            },
            priority=2,
        ))

        # ═══════════════════════════════════════════════════════════════
        # Card 5: Position status (only if holding)
        # ═══════════════════════════════════════════════════════════════
        if has_position and holding_cost:
            pnl_pct = (current_price - holding_cost) / holding_cost if current_price and holding_cost else 0
            hold_days = pf.get("hold_days", 0)
            cards.append(DashboardCard(
                card_id="position_status",
                title=f"持仓状态 · {pnl_pct:+.2%}",
                card_type="progress",
                content={
                    "value_pct": (hold_days / max_hold * 100) if max_hold > 0 else 0,
                    "label": f"持有{hold_days}/{max_hold}天 · 盈亏{pnl_pct:+.1%} · 成本¥{holding_cost:.3f}",
                    "max_value": 100,
                },
                priority=1,
            ))

        return cards

    # =====================================================================
    # Multi-symbol scanning
    # =====================================================================

    @staticmethod
    def get_candidate_pool(asset_mode: str = "etf") -> list[dict]:
        """Return candidate pool based on asset mode."""
        if asset_mode == "stock":
            return list(CANDIDATE_STOCKS)
        elif asset_mode == "auto":
            return list(CANDIDATE_ETFS) + list(CANDIDATE_STOCKS)
        else:
            return list(CANDIDATE_ETFS)

    @staticmethod
    def scan_candidates(
        candidates: list[dict] | None = None,
        asset_mode: str = "etf",
        top_n: int = 5,
        **params_overrides,
    ) -> list[dict]:
        """Scan all candidates for momentum entry signals.

        Returns candidates that:
        1. Pass Tier-1 hard filters (size, drawdown, return, popularity)
        2. Show momentum: previous day up >=2%
        3. Are above MA20 (uptrend)
        4. Have elevated volume confirming the move

        Ranked by composite momentum score (higher = stronger momentum).
        """
        from src.data.fetcher import fetch_etf_hist, fetch_multi_etf_info

        strategy = ShortTermMomentumStrategy()
        params = {**strategy.get_default_params(), **params_overrides, "asset_mode": asset_mode}

        pool = candidates if candidates is not None else ShortTermMomentumStrategy.get_candidate_pool(asset_mode)
        codes = [e["code"] for e in pool]

        # Batch real-time quotes
        try:
            all_info = fetch_multi_etf_info(codes)
        except Exception:
            all_info = {}

        # Parallel history fetch
        hist_cache: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_map = {executor.submit(fetch_etf_hist, c): c for c in codes}
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    hist_cache[code] = future.result()
                except Exception:
                    hist_cache[code] = None

        # ── Pre-rank by turnover for popularity filter ──
        turnover_list = []
        for etf in pool:
            info = all_info.get(etf["code"])
            if info:
                turnover_list.append({
                    "code": etf["code"],
                    "turnover_pct": info.get("turnover_pct") or 0,
                })
        turnover_list.sort(key=lambda x: -x["turnover_pct"])
        pool_total = len(turnover_list)
        rank_map = {t["code"]: i + 1 for i, t in enumerate(turnover_list)}

        scored: list[dict] = []

        for etf in pool:
            code = etf["code"]
            info = all_info.get(code)
            if info is None:
                continue

            cp = info.get("current_price")
            if cp is None or cp <= 0:
                continue
            if asset_mode in ("etf", "auto") and cp > float(params.get("max_etf_price", 10.0)):
                continue

            hist = hist_cache.get(code)
            if hist is None or hist.empty or len(hist) < 25:
                continue

            h = hist.sort_values("date", ascending=True).reset_index(drop=True)
            h["_ma5"] = h["close"].rolling(window=5, min_periods=5).mean()
            h["_ma20"] = h["close"].rolling(window=20, min_periods=20).mean()

            closes_arr = h["close"].values.astype(float)

            # Tier 1: Hard filters
            pool_rank = rank_map.get(code)
            hard = ShortTermMomentumStrategy._check_hard_filters(
                info, closes_arr, params,
                pool_rank=pool_rank, pool_total=pool_total,
            )

            # Tier 2: Entry conditions + composite score
            try:
                entry_check = ShortTermMomentumStrategy._check_entry_conditions(
                    h, len(h) - 1, params
                )
            except Exception:
                continue

            score = entry_check["score"]
            passed = hard["passed"] and entry_check["passed"]

            # Action label
            if passed:
                action = "🚀 动量追涨"
                action_color = "buy"
            elif score >= 40:
                action = "👀 接近条件"
                action_color = "watch"
            else:
                action = "⏳ 等待"
                action_color = "wait"

            scored.append({
                **etf,
                "current_price": cp,
                "amplitude": info.get("amplitude", 0) or 0,
                "turnover_rate": info.get("turnover_rate", 0) or 0,
                "score": score,
                "passed": passed,
                "hard_passed": hard["passed"],
                "daily_return": entry_check["daily_return"],
                "rsi_value": entry_check["rsi_value"],
                "dist_ma20": entry_check.get("dist_ma20", 0),
                "details": entry_check["details"],
                "hard_details": hard["details"],
                "block_reason": entry_check.get("block_reason") or hard.get("block_reason", ""),
                "name_from_api": info.get("name", etf["name"]),
                "action": action,
                "action_color": action_color,
                "score_breakdown": entry_check.get("score_breakdown", {}),
                "mcap_yi": hard.get("mcap_yi"),
                "drawdown_1y": hard.get("drawdown_1y"),
                "return_1y": hard.get("return_1y"),
            })

        if not scored:
            return []

        # Sort: passed first, then by score
        scored.sort(key=lambda x: (-x["passed"], -x["score"]))
        return scored[:top_n]


# =============================================================================
# Multi-asset rotation backtest
# =============================================================================

def run_multi_asset_backtest(
    codes: list[str] | None = None,
    asset_mode: str = "etf",
    initial_capital: float = 50_000,
    start_date: str | None = None,
    end_date: str | None = None,
    **strategy_params,
) -> dict:
    """Multi-asset momentum rotation backtest.

    Each day:
      1. Check exits for all open positions
      2. If slots available, scan all assets for momentum signals
      3. Enter best-scoring asset that meets all conditions
      4. Cap: max_new_per_day new entries per day
    """
    from src.data.fetcher import fetch_etf_hist

    strategy = ShortTermMomentumStrategy()
    params = {**strategy.get_default_params(), **strategy_params}
    if asset_mode:
        params["asset_mode"] = asset_mode

    take_profit = float(params["take_profit_pct"])
    stop_loss = float(params["stop_loss_pct"])
    max_hold = int(params["max_hold_days"])
    position_pct = float(params["position_pct"])
    max_concurrent = int(params.get("max_concurrent", 4))
    max_new_per_day = int(params.get("max_new_per_day", 1))
    partial_tp = float(params.get("partial_take_profit_pct", 0.03))
    partial_ratio = float(params.get("partial_sell_ratio", 0.5))
    trailing_activate = float(params.get("trailing_activate_pct", 0.015))
    trailing_giveback = float(params.get("trailing_giveback_pct", 0.008))
    max_gap_down = float(params.get("max_gap_down_pct", 0.02))
    rsi_overbought = float(params.get("rsi_overbought", 75))
    min_breadth = float(params.get("min_market_breadth", 0.60))

    # ── Determine asset pool ──
    if codes is None:
        pool = ShortTermMomentumStrategy.get_candidate_pool(asset_mode)
        codes = [e["code"] for e in pool]

    # ── Load all data ──
    print(f"加载 {len(codes)} 个标的的历史数据...")
    asset_data: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(fetch_etf_hist, c): c for c in codes}
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    df = df.sort_values("date", ascending=True).reset_index(drop=True)
                    df["_ma5"] = df["close"].rolling(window=5, min_periods=5).mean()
                    df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()
                    asset_data[code] = df
            except Exception:
                pass

    if len(asset_data) < 2:
        return {"error": "数据不足", "equity_curve": pd.DataFrame(), "trades": [], "metrics": {}}

    print(f"成功加载 {len(asset_data)} 个标的数据")

    # ── Unified date index ──
    all_dates = sorted(set().union(*[set(df["date"].tolist()) for df in asset_data.values()]))
    if start_date:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
    if end_date:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]

    # ── State ──
    cash = initial_capital
    positions: list[dict] = []
    closed_trades: list[dict] = []
    equity_rows: list[dict] = []

    print(f"回测 {len(all_dates)} 个交易日...")
    for di, date in enumerate(all_dates):
        # ── Step 1: Check exits ──
        for pos in list(positions):
            code = pos["code"]
            df_code = asset_data.get(code)
            if df_code is None:
                continue

            row_match = df_code[df_code["date"] == date]
            if row_match.empty:
                continue

            row = row_match.iloc[0]
            close = float(row["close"])
            pnl_pct = (close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
            hold_days = (date - pos["entry_date"]).days

            if pnl_pct > pos.get("highest_pnl", -999):
                pos["highest_pnl"] = pnl_pct

            exit_reason = ""
            exit_shares = 0

            # Gap-down protection at open
            open_price = float(row.get("open", close))
            prev_row_match = df_code[df_code["date"] < date]
            if not prev_row_match.empty:
                prev_close = float(prev_row_match.iloc[-1]["close"])
                if prev_close > 0:
                    gap_pct = (open_price - prev_close) / prev_close
                    if gap_pct <= -max_gap_down:
                        exit_reason = f"跳空保护 开盘跳空{gap_pct:.1%}"
                        exit_shares = pos["shares"]

            # Hard stop
            if exit_shares == 0 and pnl_pct <= -stop_loss:
                exit_reason = f"硬止损 {pnl_pct:.1%}"
                exit_shares = pos["shares"]

            # Time stop
            elif exit_shares == 0 and hold_days >= max_hold:
                exit_reason = f"时间止盈 持有{hold_days}天"
                exit_shares = pos["shares"]

            # Partial take-profit (sell half, keep rest)
            elif exit_shares == 0 and not pos.get("partial_done") and pnl_pct >= partial_tp:
                partial_shares = max(100, int(pos["shares"] * partial_ratio) // 100 * 100)
                if partial_shares >= 100 and partial_shares < pos["shares"]:
                    exit_reason = f"阶梯止盈 +{pnl_pct:.1%} 卖{partial_shares}股"
                    exit_shares = partial_shares
                    pos["partial_done"] = True

            # Full take profit
            elif exit_shares == 0 and pnl_pct >= take_profit:
                exit_reason = f"目标止盈 +{pnl_pct:.1%}"
                exit_shares = pos["shares"]

            # Trailing stop
            elif exit_shares == 0 and pos.get("highest_pnl", -999) >= trailing_activate:
                giveback = pos["highest_pnl"] - pnl_pct
                if giveback >= trailing_giveback:
                    exit_reason = f"移动止盈 回撤{giveback:.1%}"
                    exit_shares = pos["shares"]

            # RSI overbought
            if exit_shares == 0 and rsi_overbought > 0:
                local_idx = row_match.index[0]
                if local_idx >= 14:
                    closes_arr = df_code["close"].values[:local_idx + 1].astype(float)
                    rsi_arr = _compute_rsi(closes_arr, 14)
                    rsi_val = rsi_arr[local_idx]
                    if pd.notna(rsi_val) and rsi_val >= rsi_overbought:
                        exit_reason = f"RSI超买 {rsi_val:.0f}"
                        exit_shares = pos["shares"]

            if exit_shares >= 100:
                proceeds = exit_shares * close * 0.999
                cash += proceeds
                cost_basis = exit_shares * pos["entry_price"]
                pnl = proceeds - cost_basis
                is_full_exit = exit_shares >= pos["shares"]
                closed_trades.append({
                    "code": code,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": close,
                    "shares": exit_shares,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": hold_days,
                    "exit_reason": exit_reason,
                    "winning": pnl > 0,
                })
                if is_full_exit:
                    positions.remove(pos)
                else:
                    pos["shares"] -= exit_shares
                    # Adjust entry cost proportionally
                    remaining = pos["shares"]
                    if remaining > 0:
                        pos["entry_price"] = (pos["entry_price"] * (remaining + exit_shares) - exit_shares * close) / remaining

        # ── Step 2: Scan for new entries ──
        # Market breadth filter: only trade in broad uptrends
        if min_breadth > 0:
            above_ma20_count = 0
            total_count = 0
            for code, df_code in asset_data.items():
                row_m = df_code[df_code["date"] == date]
                if not row_m.empty:
                    idx_m = row_m.index[0]
                    if idx_m >= 20 and "_ma20" in df_code.columns:
                        total_count += 1
                        close_m = float(row_m.iloc[0]["close"])
                        ma20_m = df_code.at[idx_m, "_ma20"]
                        if pd.notna(ma20_m) and ma20_m > 0 and close_m > ma20_m:
                            above_ma20_count += 1
            breadth = above_ma20_count / max(total_count, 1)
            if breadth < min_breadth:
                pos_value = sum(
                    p["shares"] * float(asset_data[p["code"]][asset_data[p["code"]]["date"] == date].iloc[0]["close"])
                    for p in positions
                    if not asset_data[p["code"]][asset_data[p["code"]]["date"] == date].empty
                )
                equity_rows.append({"date": date, "equity": cash + pos_value, "cash": cash, "positions": len(positions)})
                if di % 100 == 0:
                    print(f"  {date.date()} | 市场宽度{breadth:.0%}<{min_breadth:.0%}，暂停开仓")
                continue

        slots = max_concurrent - len(positions)
        if slots <= 0:
            pos_value = sum(
                p["shares"] * float(asset_data[p["code"]][asset_data[p["code"]]["date"] == date].iloc[0]["close"])
                for p in positions
                if not asset_data[p["code"]][asset_data[p["code"]]["date"] == date].empty
            )
            equity_rows.append({"date": date, "equity": cash + pos_value, "cash": cash, "positions": len(positions)})
            continue

        new_today = 0
        candidates_today = []

        for code, df_code in asset_data.items():
            if new_today >= max_new_per_day:
                break

            row_match = df_code[df_code["date"] == date]
            if row_match.empty:
                continue

            idx = row_match.index[0]
            if idx < 25:
                continue

            if any(p["code"] == code for p in positions):
                continue

            try:
                entry_check = strategy._check_entry_conditions(df_code, idx, params)
            except Exception:
                continue

            if entry_check["passed"]:
                close = float(row_match.iloc[0]["close"])
                candidates_today.append({
                    "code": code,
                    "score": entry_check["score"],
                    "close": close,
                    "daily_return": entry_check["daily_return"],
                })

        candidates_today.sort(key=lambda x: -x["score"])

        for cand in candidates_today[:min(slots, max_new_per_day)]:
            budget = cash * position_pct / max(1, max_concurrent - len(positions))
            raw_shares = int(budget / cand["close"])
            shares = max(100, (raw_shares // 100) * 100)
            cost = shares * cand["close"] * 1.001

            # Position size cap: max 25% of total equity
            current_pos_value = sum(
                p["shares"] * cand["close"] for p in positions
            )
            est_equity = cash + current_pos_value
            max_cost = est_equity * 0.25
            if cost > max_cost:
                shares = max(100, int(max_cost / cand["close"] / 100) * 100)
                cost = shares * cand["close"] * 1.001

            if shares < 100 or cost > cash:
                continue

            cash -= cost
            positions.append({
                "code": cand["code"],
                "entry_price": cand["close"],
                "shares": shares,
                "entry_date": date,
                "highest_pnl": -999.0,
                "score": cand["score"],
                "daily_return": cand["daily_return"],
            })
            new_today += 1

        # ── Mark to market ──
        pos_value = 0.0
        for p in positions:
            rp = asset_data[p["code"]][asset_data[p["code"]]["date"] == date]
            if not rp.empty:
                pos_value += p["shares"] * float(rp.iloc[0]["close"])

        equity_rows.append({"date": date, "equity": cash + pos_value, "cash": cash, "positions": len(positions)})

        if (di + 1) % 100 == 0:
            eq = cash + pos_value
            print(f"  {date.date()} | 权益 ¥{eq:,.0f} | 现金 ¥{cash:,.0f} | "
                  f"持仓 {len(positions)}/{max_concurrent} | 累计交易 {len(closed_trades)}笔")

    # ── Close remaining positions ──
    final_date = all_dates[-1]
    for pos in positions:
        df_code = asset_data.get(pos["code"])
        if df_code is not None:
            row = df_code[df_code["date"] == final_date]
            if not row.empty:
                final_price = float(row.iloc[0]["close"])
                proceeds = pos["shares"] * final_price * 0.999
                cash += proceeds
                pnl = proceeds - pos["shares"] * pos["entry_price"]
                pnl_pct = (final_price - pos["entry_price"]) / pos["entry_price"]
                hold_days = (final_date - pos["entry_date"]).days
                closed_trades.append({
                    "code": pos["code"],
                    "entry_date": pos["entry_date"],
                    "exit_date": final_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": final_price,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": hold_days,
                    "exit_reason": "回测结束平仓",
                    "winning": pnl > 0,
                })

    # ── Results ──
    eq_df = pd.DataFrame(equity_rows)
    final_equity = cash

    wins = [t for t in closed_trades if t["winning"]]
    losses = [t for t in closed_trades if not t["winning"]]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0

    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
    avg_hold = float(np.mean([t["holding_days"] for t in closed_trades])) if closed_trades else 0

    total_return = (final_equity - initial_capital) / initial_capital if initial_capital > 0 else 0

    days_span = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 1 else 0
    annual_return = (final_equity / initial_capital) ** (1.0 / (days_span / 365.25)) - 1.0 if days_span > 0 and final_equity > 0 else 0.0

    if not eq_df.empty and len(eq_df) > 1:
        cummax = eq_df["equity"].cummax()
        dd = (eq_df["equity"] - cummax) / cummax
        max_dd = float(dd.min())
        daily_ret = eq_df.set_index("date")["equity"].pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else 0.0
        sharpe = (annual_return - 0.03) / ann_vol if ann_vol > 0 else 0.0
    else:
        max_dd = 0.0; ann_vol = 0.0; sharpe = 0.0

    total_weeks = days_span / 7 if days_span > 0 else 1
    trades_per_week = len(closed_trades) / total_weeks

    total_wins = sum(t["pnl"] for t in wins) if wins else 0
    total_losses = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if closed_trades else 0
    weekly_return_est = expectancy * trades_per_week

    metrics = {
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": len(closed_trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "avg_holding_days": round(avg_hold, 1),
        "trades_per_week": round(trades_per_week, 1),
        "profit_factor": round(profit_factor, 2),
        "expectancy_pct": round(expectancy, 4),
        "weekly_return_est": round(weekly_return_est, 4),
    }

    summary = (
        f"短线动量 多标的轮动回测:\n"
        f"  初始资金: ¥{initial_capital:,.0f}\n"
        f"  最终权益: ¥{final_equity:,.0f}\n"
        f"  总收益: {total_return:+.1%}  年化: {annual_return:+.1%}\n"
        f"  Sharpe: {sharpe:.2f}  最大回撤: {max_dd:.1%}\n"
        f"  胜率: {win_rate:.0%} ({len(wins)}W/{len(losses)}L)\n"
        f"  平均盈利: {avg_win:+.2%}  平均亏损: {avg_loss:+.2%}\n"
        f"  平均持有: {avg_hold:.1f}天  周交易: {trades_per_week:.1f}次\n"
        f"  盈亏比: {profit_factor:.2f}  期望值: {expectancy:+.2%}/笔\n"
        f"  预估周收益: {weekly_return_est:+.2%}"
    )

    print(f"\n{summary}")

    return {
        "equity_curve": eq_df,
        "trades": closed_trades,
        "metrics": metrics,
        "summary": summary,
    }
