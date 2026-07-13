"""超短波段 v2 — "恐慌反弹"策略，基于行为金融学的过度反应效应。

核心理念（学术基础）：
  - ETF宽基指数的单日大幅下跌（>1.5%）往往是过度反应，1-3天内会部分反弹
  - 行为金融学文献：Barberis, Shleifer & Vishny (1998); Daniel, Hirshleifer (1997)
  - 关键约束：必须在上升趋势中（MA20上方），确保是"回调"而非"崩盘"
  - 大成交量确认恐慌性质，增加反弹概率

入场条件（全部满足）：
  1. 单日跌幅 ≥ 1.5%（恐慌日，过度反应信号）
  2. 收盘价 > MA20（中期上升趋势，确保是回调不是崩盘）
  3. 成交量 > 1.2x 20日均量（恐慌抛售确认）
  4. 连跌天数 ≤ 3（不是持续性下跌）
  5. 近5日振幅 > 2%（有足够波动性，横盘ETF跳过）

加分项（提高优先级的因子，非必须）：
  - RSI(14) < 30：极端超卖，反弹力度更强 → +10分
  - 单日跌幅 > 2.5%：更大的过度反应 → +10分
  - 收盘价距MA10 < 1%：接近支撑位 → +5分

出场规则（按优先级）：
  1. 硬止损 -1.0%（市场证明你错了，立即退出）
  2. 时间止盈 2天（超短线核心：持仓不过夜2晚）
  3. 目标止盈 +1.5%（快速锁定利润，不贪心）
  4. 移动止损：盈利>1%后激活，回撤50%止盈
  5. 跳空保护：开盘价距昨收>2%立即平仓（防黑天鹅）

仓位管理：
  - 单笔15-25%资金（小仓位，高频交易）
  - 最多同时持有3个仓位（资金利用率45-75%）
  - 单日最多开1个新仓（控制节奏）

目标：周交易4-10次，胜率>75%，周收益3%，月收益6%
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
# Candidate ETF pool — 25 broad-market ETFs
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

    # ── 小盘/微盘 (3) ── (波动大，反弹力度强)
    {"code": "512100", "name": "中证1000ETF",        "approx_price": 2.3,  "category": "small"},
    {"code": "563300", "name": "中证2000ETF",        "approx_price": 0.9,  "category": "small"},
    {"code": "159628", "name": "国证2000ETF",        "approx_price": 1.0,  "category": "small"},

    # ── 新宽基 (3) ──
    {"code": "159593", "name": "中证A50ETF",         "approx_price": 1.0,  "category": "new_broad"},
    {"code": "159338", "name": "中证A500ETF",        "approx_price": 0.9,  "category": "new_broad"},
    {"code": "560050", "name": "MSCI A50ETF",        "approx_price": 0.8,  "category": "new_broad"},

    # ── 创业/科创 (4) ── (高beta，反弹弹性大)
    {"code": "159915", "name": "创业板ETF",          "approx_price": 2.2,  "category": "growth"},
    {"code": "159949", "name": "创业板50ETF",        "approx_price": 0.9,  "category": "growth"},
    {"code": "588000", "name": "科创50ETF",          "approx_price": 0.9,  "category": "growth"},
    {"code": "588190", "name": "科创100ETF",         "approx_price": 0.8,  "category": "growth"},

    # ── 双创 (1) ──
    {"code": "159781", "name": "双创50ETF",          "approx_price": 1.1,  "category": "growth"},

    # ── 红利/低波 (5) ── (防御型，波动小但稳定)
    {"code": "510880", "name": "中证红利ETF",        "approx_price": 3.0,  "category": "defensive"},
    {"code": "515080", "name": "中证红利ETF(招商)",  "approx_price": 1.5,  "category": "defensive"},
    {"code": "512890", "name": "红利低波ETF",        "approx_price": 1.5,  "category": "defensive"},
    {"code": "515180", "name": "红利低波100ETF",     "approx_price": 1.3,  "category": "defensive"},
    {"code": "563020", "name": "红利低波ETF(易方达)","approx_price": 1.0,  "category": "defensive"},

    # ── 策略宽基 (3) ──
    {"code": "515450", "name": "红利质量ETF",        "approx_price": 1.2,  "category": "defensive"},
    {"code": "159905", "name": "深红利ETF",          "approx_price": 2.0,  "category": "defensive"},
    {"code": "562060", "name": "中证A50增强ETF",     "approx_price": 1.0,  "category": "new_broad"},
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


def _daily_return_pct(opens: np.ndarray, closes: np.ndarray, idx: int) -> float:
    """Today's return: (close - prev_close) / prev_close."""
    if idx < 1:
        return 0.0
    prev_close = closes[idx - 1]
    if prev_close <= 0:
        return 0.0
    return float((closes[idx] - prev_close) / prev_close)


# =============================================================================
# Strategy class
# =============================================================================

class UltraShortStrategy(BaseStrategy):
    """超短波段 v2 — 恐慌反弹（Shock Bounce）。

    基于行为金融学的过度反应效应：ETF单日大幅下跌后会在1-2天内部分反弹。
    严格只在上升趋势（MA20上方）的回调中入场，确保是"回调"而非"崩盘"。
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "超短波段"

    @property
    def description(self) -> str:
        return (
            "恐慌反弹策略（Shock Bounce）：基于行为金融学过度反应效应。\n\n"
            "**入场**：单日跌幅≥1.5% + MA20上方（上升趋势） + 放量（恐慌确认） + 连跌≤3天。\n"
            "**出场**：-1.0%硬止损 | 2天时间止盈 | +1.5%目标止盈 | 移动保本止损。\n"
            "**仓位**：每笔15-25%资金，最多3个并发仓位，单日最多1个新仓。\n\n"
            "目标：周交易4-10次，胜率>75%，周收益3%，月收益6%。"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return {
            # Entry — core conditions
            "min_daily_drop_pct": 0.015,     # 单日跌幅≥1.5%（核心入场条件）
            "require_above_ma20": True,       # 必须在MA20上方（上升趋势）
            "min_vol_ratio": 1.2,            # 成交量>1.2x 20日均量
            "max_consec_down": 3,            # 最多连跌3天
            "min_amplitude_5d": 0.02,        # 近5日振幅>2%

            # Entry — bonus factors
            "rsi_bonus_threshold": 30,        # RSI(14)<30加分
            "big_drop_bonus_pct": 0.025,      # 跌幅>2.5%额外加分

            # Exit
            "take_profit_pct": 0.015,         # 目标止盈+1.5%
            "stop_loss_pct": 0.01,            # 硬止损-1.0%
            "max_hold_days": 2,              # 最长持有2天
            "trailing_activate_pct": 0.01,    # 盈利>1%后激活移动止损
            "trailing_giveback_pct": 0.005,   # 从最高点回撤0.5%止盈

            # Gap protection
            "max_gap_pct": 0.02,             # 开盘跳空>2%立即平仓

            # Position sizing
            "position_pct": 0.20,             # 每笔20%资金
            "max_concurrent": 3,              # 最多3个并发仓位
            "max_new_per_day": 1,             # 单日最多1个新仓
            "max_etf_price": 10.0,

            # Filters
            "enable_macro_filter": True,
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "min_daily_drop_pct": {
                "label": "最小单日跌幅",
                "type": "slider",
                "min": 0.01, "max": 0.04, "step": 0.005,
                "help": "单日跌幅需≥此值才考虑入场（默认1.5%）。核心过滤条件，越大信号越少但质量越高",
            },
            "require_above_ma20": {
                "label": "要求站上MA20",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=只在上升趋势中做反弹 | 关闭=任何趋势都可入场（风险更大）",
            },
            "min_vol_ratio": {
                "label": "最小量比",
                "type": "slider",
                "min": 1.0, "max": 2.0, "step": 0.1,
                "help": "成交量/20日均量需≥此值（默认1.2x）。确认恐慌抛售",
            },
            "max_consec_down": {
                "label": "最大连跌天数",
                "type": "number",
                "min": 1, "max": 5, "step": 1,
                "help": "连跌超过此天数不入场（默认3天）。避免接飞刀",
            },
            "min_amplitude_5d": {
                "label": "最小5日振幅",
                "type": "slider",
                "min": 0.01, "max": 0.05, "step": 0.005,
                "help": "近5日平均振幅需≥此值（默认2%）。过滤横盘ETF",
            },
            "rsi_bonus_threshold": {
                "label": "RSI加分线",
                "type": "number",
                "min": 20, "max": 40, "step": 5,
                "help": "RSI(14)低于此值额外加分（默认30）。超卖越严重反弹越强",
            },
            "big_drop_bonus_pct": {
                "label": "大跌加分线",
                "type": "slider",
                "min": 0.02, "max": 0.05, "step": 0.005,
                "help": "单日跌幅超过此值额外加分（默认2.5%）。跌幅越大反弹越强",
            },
            "take_profit_pct": {
                "label": "目标止盈线",
                "type": "slider",
                "min": 0.01, "max": 0.03, "step": 0.005,
                "help": "盈利达到此比例全部卖出（默认1.5%）。小目标更容易达成",
            },
            "stop_loss_pct": {
                "label": "硬止损线",
                "type": "slider",
                "min": 0.005, "max": 0.02, "step": 0.005,
                "help": "亏损达到此比例立即卖出（默认1.0%）。超短止损保护本金",
            },
            "max_hold_days": {
                "label": "最长持有天数",
                "type": "number",
                "min": 1, "max": 4, "step": 1,
                "help": "超过此天数强制平仓（默认2天）。超短线核心约束",
            },
            "trailing_activate_pct": {
                "label": "移动止损激活线",
                "type": "slider",
                "min": 0.005, "max": 0.02, "step": 0.005,
                "help": "盈利超过此比例后激活移动止损（默认1.0%）",
            },
            "trailing_giveback_pct": {
                "label": "移动止损回撤容忍",
                "type": "slider",
                "min": 0.002, "max": 0.01, "step": 0.001,
                "help": "从最高盈利回撤超过此值即卖出（默认0.5%）。越小越保守",
            },
            "max_gap_pct": {
                "label": "跳空保护线",
                "type": "slider",
                "min": 0.01, "max": 0.05, "step": 0.005,
                "help": "次日开盘价相对昨收跳空超过此值立即平仓（默认2.0%）。防黑天鹅",
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
                "help": "最多同时持有几个ETF（默认3个）",
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
        }

    # ------------------------------------------------------------------
    # Entry condition check (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_entry_conditions(
        df_sorted: pd.DataFrame,
        idx: int,
        params: dict,
    ) -> dict:
        """Check if bar at *idx* meets all entry conditions.

        Returns dict with:
          passed: bool — all required conditions met
          score: float — quality score (0-100, for ranking)
          reason: str — human-readable signal description
          details: list[str] — per-condition results
          gap_risk: float — estimated gap-down risk at next open
        """
        row = df_sorted.iloc[idx]
        close = float(row["close"])
        vol = float(row.get("volume", 0) or 0)

        min_drop = float(params.get("min_daily_drop_pct", 0.015))
        require_ma20 = str(params.get("require_above_ma20", "True")).lower() in ("true", "1", "yes")
        min_vol_ratio = float(params.get("min_vol_ratio", 1.2))
        max_consec = int(params.get("max_consec_down", 3))
        min_amp = float(params.get("min_amplitude_5d", 0.02))
        rsi_bonus = float(params.get("rsi_bonus_threshold", 30))
        big_drop_bonus = float(params.get("big_drop_bonus_pct", 0.025))

        details: list[str] = []
        blocked = False
        block_reason = ""

        # ── Condition 1: Single-day drop ≥ threshold (REQUIRED) ──
        daily_ret = _daily_return_pct(
            df_sorted["open"].values.astype(float),
            df_sorted["close"].values.astype(float),
            idx,
        )
        drop_ok = daily_ret <= -min_drop
        drop_score = 0.0
        if drop_ok:
            # Score based on drop magnitude: -1.5%→20pts, -2.5%→40pts, -3.5%→50pts
            drop_magnitude = abs(daily_ret)
            drop_score = min(50.0, drop_magnitude * 2000)  # 0.015→30, 0.025→50
            if drop_magnitude >= big_drop_bonus:
                details.append(f"✅ 单日暴跌{daily_ret:.1%}（≥{big_drop_bonus:.0%}加分） +{drop_score:.0f}")
            else:
                details.append(f"✅ 单日跌幅{daily_ret:.1%}（≥{min_drop:.0%}） +{drop_score:.0f}")
        else:
            details.append(f"❌ 单日跌幅{daily_ret:.1%}（需≥{min_drop:.0%}）")
            blocked = True
            block_reason = f"单日跌幅不足（{daily_ret:.1%} < {min_drop:.0%}）"

        # ── Condition 2: Above MA20 (REQUIRED if enabled) ──
        ma20_ok = True
        ma20_score = 0.0
        if "_ma20" in df_sorted.columns:
            ma20_val = df_sorted.at[idx, "_ma20"]
            if pd.notna(ma20_val) and ma20_val > 0:
                dist20 = (close - ma20_val) / ma20_val
                if dist20 >= 0:
                    ma20_ok = True
                    ma20_score = 15.0
                    details.append(f"✅ MA20上方{dist20:+.1%}（上升趋势） +{ma20_score:.0f}")
                elif dist20 >= -0.015:
                    ma20_ok = True  # Slightly below, still acceptable
                    ma20_score = 5.0
                    details.append(f"🟡 MA20下方{dist20:.1%}（轻微跌破） +{ma20_score:.0f}")
                else:
                    ma20_ok = False
                    ma20_score = -10.0
                    details.append(f"🔴 MA20下方{dist20:.1%}（深度跌破）")
                    if require_ma20:
                        blocked = True
                        block_reason = f"深度跌破MA20（{dist20:.1%}），趋势走坏"
        else:
            details.append("⚪ MA20未计算，跳过趋势检查")

        # ── Condition 3: Volume > threshold × 20-day avg (REQUIRED) ──
        vol_ok = False
        vol_score = 0.0
        if idx >= 20 and "volume" in df_sorted.columns:
            avg_vol_20 = float(np.mean([
                float(df_sorted.iloc[i].get("volume", 0) or 0)
                for i in range(max(0, idx - 20), idx)
            ]))
            if avg_vol_20 > 0:
                vol_ratio = vol / avg_vol_20
                if vol_ratio >= min_vol_ratio:
                    vol_ok = True
                    vol_score = min(15.0, vol_ratio * 10)
                    details.append(f"✅ 放量{vol_ratio:.1f}x（≥{min_vol_ratio:.1f}x，恐慌确认） +{vol_score:.0f}")
                else:
                    details.append(f"❌ 量比{vol_ratio:.1f}x（需≥{min_vol_ratio:.1f}x）")
                    blocked = True
                    if not block_reason:
                        block_reason = f"成交量不足（{vol_ratio:.1f}x < {min_vol_ratio:.1f}x）"
            else:
                details.append("⚪ 均量=0，跳过量比检查")
                vol_ok = True  # Pass if no data
        else:
            details.append("⚪ 量数据不足，跳过")
            vol_ok = True  # Pass if insufficient data

        # ── Condition 4: Consecutive down days ≤ max (REQUIRED) ──
        consec = 0
        if idx >= 1:
            for j in range(idx, max(idx - 10, -1), -1):
                if j >= 1:
                    prev_c = float(df_sorted.iloc[j - 1]["close"])
                    prev_o = float(df_sorted.iloc[j - 1]["open"])
                    if prev_c < prev_o:
                        consec += 1
                    else:
                        break

        consec_ok = consec <= max_consec
        consec_score = 0.0
        if consec_ok:
            if consec == 0:
                consec_score = 5.0
                details.append(f"🟡 首日下跌 +{consec_score:.0f}")
            elif consec <= 2:
                consec_score = 15.0
                details.append(f"✅ 连跌{consec}天（理想回调） +{consec_score:.0f}")
            else:
                consec_score = 5.0
                details.append(f"🟡 连跌{consec}天（接近上限） +{consec_score:.0f}")
        else:
            consec_score = -5.0
            details.append(f"🔴 连跌{consec}天（>{max_consec}天，趋势性下跌）")
            blocked = True
            block_reason = f"连跌{consec}天，回避趋势性下跌"

        # ── Condition 5: Amplitude > min (REQUIRED) ──
        amp_ok = False
        amp_score = 0.0
        if idx >= 4:
            amps = []
            for j in range(max(0, idx - 4), idx + 1):
                h = float(df_sorted.iloc[j].get("high", 0) or 0)
                l = float(df_sorted.iloc[j].get("low", 0) or 0)
                ref = float(df_sorted.iloc[j - 1].get("close", h) or h) if j > 0 else h
                if ref > 0:
                    amps.append((h - l) / ref)
            if amps:
                avg_amp = float(np.mean(amps))
                if avg_amp >= min_amp:
                    amp_ok = True
                    amp_score = min(10.0, avg_amp * 200)  # 2%→4pts, 3%→6pts, 5%→10pts
                    details.append(f"✅ 近5日振幅{avg_amp:.1%}（≥{min_amp:.0%}） +{amp_score:.0f}")
                else:
                    details.append(f"❌ 近5日振幅{avg_amp:.1%}（<{min_amp:.0%}，横盘ETF）")
                    blocked = True
                    if not block_reason:
                        block_reason = f"振幅不足（{avg_amp:.1%} < {min_amp:.0%}），横盘ETF不适合超短"
        else:
            details.append("⚪ 振幅数据不足，跳过")
            amp_ok = True

        # ── Bonus: RSI(14) oversold (NOT REQUIRED, bonus points) ──
        rsi_val = None
        rsi_score = 0.0
        if idx >= 14:
            closes_arr = df_sorted["close"].values[:idx + 1].astype(float)
            rsi_arr = _compute_rsi(closes_arr, 14)
            rsi_val = rsi_arr[idx]
            if pd.notna(rsi_val):
                if rsi_val < rsi_bonus:
                    rsi_score = 15.0  # Big bonus for extreme oversold
                    details.append(f"✅ RSI(14)={rsi_val:.1f} < {rsi_bonus}（极端超卖加分） +{rsi_score:.0f}")
                elif rsi_val < 35:
                    rsi_score = 8.0
                    details.append(f"🟡 RSI(14)={rsi_val:.1f}（超卖区域） +{rsi_score:.0f}")
                elif rsi_val < 45:
                    rsi_score = 3.0
                    details.append(f"⚪ RSI(14)={rsi_val:.1f}（中性偏低） +{rsi_score:.0f}")
                else:
                    details.append(f"⚪ RSI(14)={rsi_val:.1f}（中性） +0")
        else:
            details.append("⚪ RSI数据不足 +0")

        # ── Bonus: MA10 proximity (NOT REQUIRED) ──
        ma10_score = 0.0
        if "_ma" in df_sorted.columns:
            ma10_val = df_sorted.at[idx, "_ma"]
            if pd.notna(ma10_val) and ma10_val > 0:
                dist10 = abs(close - ma10_val) / ma10_val
                if dist10 < 0.01:
                    ma10_score = 10.0
                    details.append(f"✅ 距MA10仅{dist10:.1%}（接近支撑） +{ma10_score:.0f}")
                elif dist10 < 0.02:
                    ma10_score = 5.0
                    details.append(f"🟡 距MA10 {dist10:.1%} +{ma10_score:.0f}")
                else:
                    details.append(f"⚪ 距MA10 {dist10:.1%}（较远） +0")
        else:
            details.append("⚪ MA10未计算")

        # ── Gap risk estimate ──
        gap_risk = abs(daily_ret) * 1.5 if drop_ok else abs(daily_ret)  # After a big drop, gap risk is higher

        # ── Total score ──
        total = round(drop_score + ma20_score + vol_score + consec_score +
                      amp_score + rsi_score + ma10_score, 1)
        total = max(0.0, min(100.0, total))

        return {
            "passed": not blocked,
            "score": total,
            "daily_return": daily_ret,
            "drop_ok": drop_ok,
            "ma20_ok": ma20_ok,
            "vol_ok": vol_ok,
            "consec_ok": consec_ok,
            "amp_ok": amp_ok,
            "consec_days": consec,
            "rsi_value": rsi_val,
            "gap_risk": round(gap_risk, 4),
            "blocked": blocked,
            "block_reason": block_reason,
            "details": details,
            # Per-factor scores
            "drop_score": drop_score,
            "ma20_score": ma20_score,
            "vol_score": vol_score,
            "consec_score": consec_score,
            "amp_score": amp_score,
            "rsi_score": rsi_score,
            "ma10_score": ma10_score,
        }

    # ------------------------------------------------------------------
    # Backtest signal generation (single ETF)
    # ------------------------------------------------------------------

    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}

        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        position_pct = float(params["position_pct"])
        max_gap = float(params.get("max_gap_pct", 0.02))
        trailing_activate = float(params.get("trailing_activate_pct", 0.01))
        trailing_giveback = float(params.get("trailing_giveback_pct", 0.005))
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
        df["_ma"] = df["close"].rolling(window=10, min_periods=10).mean()
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

                # Tier 0: Gap protection (check open vs previous close)
                open_price = float(df.at[i, "open"]) if "open" in df.columns else close
                prev_close = float(df.at[i - 1, "close"]) if i > 0 else open_price
                if prev_close > 0:
                    gap_pct = (open_price - prev_close) / prev_close
                    if gap_pct <= -max_gap:
                        exit_reason = (
                            f"⚠️ 跳空保护：开盘跳空 {gap_pct:.1%}（≤-{max_gap:.0%}），"
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

                # Tier 3: Take profit
                elif exit_shares == 0 and pnl_pct >= take_profit:
                    exit_reason = (
                        f"🎯 目标止盈：盈利 {pnl_pct:.1%}（≥{take_profit:.0%}），"
                        f"入场 {entry_price:.3f} → 出场 {close:.3f}"
                    )
                    exit_shares = total_shares

                # Tier 4: Trailing stop (giveback from peak)
                elif exit_shares == 0 and highest_pnl_pct >= trailing_activate:
                    giveback = highest_pnl_pct - pnl_pct
                    if giveback >= trailing_giveback:
                        exit_reason = (
                            f"🔒 移动止盈：从最高+{highest_pnl_pct:.1%}回撤{giveback:.1%}（≥{trailing_giveback:.1%}），"
                            f"当前盈亏 {pnl_pct:+.1%}，入场 {entry_price:.3f} → 出场 {close:.3f}"
                        )
                        exit_shares = total_shares

                if exit_shares >= 100:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = exit_shares
                    df.at[i, "signal_reason"] = exit_reason
                    in_position = False
                    total_shares = 0
                    highest_pnl_pct = 0.0

            # ── Entry logic ──
            if not in_position:
                if macro_suppress:
                    continue

                # Need at least 20 bars of data for MA20
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
                    f"🎯 恐慌反弹：单日{daily_ret:.1%}，评分{score:.0f}分 | "
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

        return df.drop(columns=["_ma", "_ma20"], errors="ignore")

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
        enable_macro = str(params.get("enable_macro_filter", "True")).lower() in ("true", "1", "yes")

        # Ensure MAs
        if "_ma" not in df.columns:
            df = df.copy()
            df["_ma"] = df["close"].rolling(window=10, min_periods=10).mean()
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

        # ── In position: check exits ──
        if has_position and holding_cost and holding_cost > 0:
            pnl_pct = (current_price - holding_cost) / holding_cost

            if pnl_pct <= -stop_loss:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"亏损 {pnl_pct:.1%}，触发硬止损",
                    reason=f"🛑 硬止损！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（{pnl_pct:.1%}），建议立即卖出。",
                    urgency_level="high", current_zone="止损区",
                )

            if hold_days >= max_hold:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"持有 {hold_days} 天，触发时间止盈",
                    reason=f"⏰ 时间到！持有 {hold_days} 天（≥{max_hold}天），盈亏 {pnl_pct:+.1%}，建议卖出。",
                    urgency_level="medium", current_zone="时间止盈区",
                )

            if pnl_pct >= take_profit:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"盈利 {pnl_pct:.1%}，触发目标止盈",
                    reason=f"🎯 止盈！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（+{pnl_pct:.1%}），建议卖出。",
                    urgency_level="high", current_zone="止盈区",
                )

            next_tp = round(holding_cost * (1 + take_profit), 4)
            next_sl = round(holding_cost * (1 - stop_loss), 4)
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                trigger_description=f"持仓中 盈亏{pnl_pct:+.1%} 第{hold_days}天",
                next_trigger_price=next_tp,
                reason=f"📌 持仓中：入场 ¥{holding_cost:.3f}，现价 ¥{current_price:.3f}（{pnl_pct:+.1%}），止盈 ¥{next_tp}，止损 ¥{next_sl}",
                urgency_level="low",
                portions_used=hold_days, portions_total=max_hold,
                current_zone=f"持仓中 第{hold_days}天",
            )

        # ── Not in position: check entry ──
        if macro_suppress:
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                reason="⚠️ 宏观情绪极端恐慌，暂停所有新开仓。",
                urgency_level="low", current_zone="宏观暂停区",
            )

        if len(df_sorted) < 25:
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                reason=f"数据不足（需要至少25个交易日）",
                urgency_level="low",
            )

        entry_check = self._check_entry_conditions(df_sorted, len(df_sorted) - 1, params)

        if entry_check["blocked"]:
            return LiveSignal(
                action="wait_for_drop", current_price=round(current_price, 4),
                trigger_description=f"入场被阻止：{entry_check['block_reason']}",
                reason=f"⛔ {entry_check['block_reason']}\n\n" + "\n".join(entry_check["details"]),
                urgency_level="low", current_zone="等待条件",
            )

        if not entry_check["passed"]:
            return LiveSignal(
                action="wait_for_drop", current_price=round(current_price, 4),
                trigger_description=f"等待入场条件",
                reason=f"⏳ 等待恐慌日（单日跌幅≥{params['min_daily_drop_pct']:.0%}）或其他入场条件。\n\n" + "\n".join(entry_check["details"]),
                urgency_level="low", current_zone="等待条件",
            )

        # Calculate shares
        available_cash = float(pf.get("available_cash", 2000)) if pf else 2000
        budget = available_cash * float(params["position_pct"])
        raw_shares = int(budget / current_price) if current_price and current_price > 0 else 0
        shares = max(100, (raw_shares // 100) * 100)

        if shares < 100:
            return LiveSignal(
                action="wait_for_drop", current_price=round(current_price, 4),
                reason=f"资金不足（需¥{current_price*100:.0f}买100股，可用¥{budget:.0f}）",
                urgency_level="low", current_zone="等待资金",
            )

        amount = round(shares * current_price, 2)
        daily_ret = entry_check["daily_return"]
        return LiveSignal(
            action="buy", current_price=round(current_price, 4),
            suggested_shares=shares, suggested_amount=amount,
            trigger_description=f"恐慌反弹信号 日跌幅{daily_ret:.1%} 评分{entry_check['score']:.0f}",
            reason=(
                f"🎯 恐慌反弹买入信号！\n单日跌幅: {daily_ret:.1%}\n"
                + "\n".join(entry_check["details"])
                + f"\n\n建议买入 {shares} 股（≈¥{amount:.0f}），"
                + f"止盈 +{take_profit:.0%}，止损 -{stop_loss:.0%}，持有≤{max_hold}天。"
            ),
            urgency_level="high",
            current_zone=f"恐慌反弹 · 评分{entry_check['score']:.0f}",
        )

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
        min_drop = float(params.get("min_daily_drop_pct", 0.015))

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

        if "_ma" not in df_sorted.columns:
            df_sorted["_ma"] = df_sorted["close"].rolling(window=10, min_periods=10).mean()
        if "_ma20" not in df_sorted.columns:
            df_sorted["_ma20"] = df_sorted["close"].rolling(window=20, min_periods=20).mean()

        current_price = info.get("current_price") if info else None
        if current_price is None and len(df_sorted) > 0:
            current_price = float(df_sorted.iloc[-1]["close"])

        cards: list[DashboardCard] = []

        # ── Card 1: Entry conditions check ──
        if len(df_sorted) >= 25:
            try:
                entry_check = self._check_entry_conditions(df_sorted, len(df_sorted) - 1, params)
            except Exception:
                entry_check = None
        else:
            entry_check = None

        if entry_check:
            conditions = [
                {"label": f"单日跌幅≥{min_drop:.0%}", "met": entry_check["drop_ok"],
                 "detail": f"当前{entry_check['daily_return']:.1%}"},
                {"label": "MA20上方（上升趋势）", "met": entry_check["ma20_ok"],
                 "detail": "✅" if entry_check["ma20_ok"] else "❌"},
                {"label": f"放量≥{params['min_vol_ratio']:.1f}x", "met": entry_check["vol_ok"],
                 "detail": "✅" if entry_check["vol_ok"] else "❌"},
                {"label": f"连跌≤{int(params['max_consec_down'])}天", "met": entry_check["consec_ok"],
                 "detail": f"当前{entry_check['consec_days']}天"},
                {"label": f"振幅≥{float(params['min_amplitude_5d']):.0%}", "met": entry_check["amp_ok"],
                 "detail": "✅" if entry_check["amp_ok"] else "❌"},
            ]
            score = entry_check["score"]
            passed = entry_check["passed"]
            cards.append(DashboardCard(
                card_id="entry_conditions",
                title=f"入场条件 · 评分{score:.0f}/100" + (" ✅" if passed else f" ⛔{entry_check.get('block_reason','')}"),
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

        # ── Card 2: Exit rules ──
        cards.append(DashboardCard(
            card_id="exit_rules",
            title="出场规则",
            card_type="info",
            content={
                "rules": [
                    {"label": "🥇 跳空保护", "value": f"开盘跳空>{float(params.get('max_gap_pct',0.02)):.0%}", "priority": "最高"},
                    {"label": "🥈 硬止损", "value": f"-{stop_loss:.0%}", "priority": "最高"},
                    {"label": "🥉 时间止盈", "value": f"持有{max_hold}天", "priority": "高"},
                    {"label": "🎯 目标止盈", "value": f"+{take_profit:.0%}", "priority": "高"},
                    {"label": "🔒 移动止盈", "value": f"回撤{float(params.get('trailing_giveback_pct',0.005)):.1%}", "priority": "中"},
                ],
            },
            priority=1,
        ))

        # ── Card 3: Market context ──
        ma_val = df_sorted.at[len(df_sorted) - 1, "_ma"] if len(df_sorted) > 0 else None
        ma20_val = df_sorted.at[len(df_sorted) - 1, "_ma20"] if len(df_sorted) > 0 else None

        closes_arr = df_sorted["close"].values.astype(float)
        rsi_val = None
        if len(closes_arr) >= 15:
            rsi_arr = _compute_rsi(closes_arr, 14)
            rsi_val = round(float(rsi_arr[-1]), 1) if pd.notna(rsi_arr[-1]) else None

        daily_ret = _daily_return_pct(
            df_sorted["open"].values.astype(float) if "open" in df_sorted.columns else closes_arr,
            closes_arr, len(df_sorted) - 1
        )

        cards.append(DashboardCard(
            card_id="market_context",
            title="市场环境",
            card_type="info",
            content={
                "current_price": round(current_price, 4) if current_price else None,
                "daily_return": f"{daily_ret:.2%}",
                "ma10_value": round(float(ma_val), 4) if pd.notna(ma_val) else None,
                "ma20_value": round(float(ma20_val), 4) if pd.notna(ma20_val) else None,
                "rsi_14": rsi_val,
                "gap_risk": f"{entry_check.get('gap_risk', 0):.2%}" if entry_check else "N/A",
            },
            priority=2,
        ))

        # ── Card 4: Position status ──
        pf = kwargs.get("portfolio_context") or {}
        if pf.get("has_position"):
            pnl_pct = (current_price - pf["holding_avg_cost"]) / pf["holding_avg_cost"] if current_price and pf.get("holding_avg_cost") else 0
            hold_days = 0
            last_buy = pf.get("last_buy_date")
            if last_buy:
                try:
                    if isinstance(last_buy, str):
                        last_buy = datetime.strptime(last_buy, "%Y-%m-%d").date()
                    hold_days = (datetime.now().date() - last_buy).days
                except Exception:
                    pass

            cards.append(DashboardCard(
                card_id="position_status",
                title=f"持仓状态 · {pnl_pct:+.2%}",
                card_type="progress",
                content={
                    "value_pct": (hold_days / max_hold * 100) if max_hold > 0 else 0,
                    "label": f"持有{hold_days}/{max_hold}天 · 盈亏{pnl_pct:+.1%}",
                    "max_value": 100,
                },
                priority=1,
            ))

        return cards

    # =====================================================================
    # Multi-ETF scanning
    # =====================================================================

    @staticmethod
    def get_candidate_pool() -> list[dict]:
        return list(CANDIDATE_ETFS)

    @staticmethod
    def scan_all_etfs(
        candidates: list[dict] | None = None,
        max_price: float = 10.0,
        top_n: int = 5,
    ) -> list[dict]:
        """Scan all ETFs for shock bounce entry signals.

        Returns ETFs that:
        1. Had a ≥1.5% single-day drop today
        2. Are above MA20 (uptrend)
        3. Have elevated volume
        4. Have ≤3 consecutive down days

        Ranked by entry quality score (higher = better bounce probability).
        """
        from src.data.fetcher import fetch_etf_hist, fetch_multi_etf_info

        pool = candidates if candidates is not None else list(CANDIDATE_ETFS)
        codes = [e["code"] for e in pool]

        try:
            all_info = fetch_multi_etf_info(codes)
        except Exception:
            all_info = {}

        hist_cache: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_map = {executor.submit(fetch_etf_hist, c): c for c in codes}
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    hist_cache[code] = future.result()
                except Exception:
                    hist_cache[code] = None

        params = UltraShortStrategy().get_default_params()
        scored: list[dict] = []

        for etf in pool:
            code = etf["code"]
            info = all_info.get(code)
            if info is None:
                continue

            cp = info.get("current_price")
            if cp is None or cp <= 0 or cp > max_price:
                continue

            hist = hist_cache.get(code)
            if hist is None or hist.empty or len(hist) < 25:
                continue

            h = hist.sort_values("date", ascending=True).reset_index(drop=True)
            h["_ma"] = h["close"].rolling(window=10, min_periods=10).mean()
            h["_ma20"] = h["close"].rolling(window=20, min_periods=20).mean()

            try:
                entry_check = UltraShortStrategy._check_entry_conditions(h, len(h) - 1, params)
            except Exception:
                continue

            score = entry_check["score"]
            passed = entry_check["passed"]

            if passed:
                action = "🎯 恐慌反弹"
                action_color = "buy"
            elif score >= 25:
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
                "daily_return": entry_check["daily_return"],
                "rsi_value": entry_check["rsi_value"],
                "consec_down": entry_check["consec_days"],
                "details": entry_check["details"],
                "block_reason": entry_check.get("block_reason", ""),
                "name_from_api": info.get("name", etf["name"]),
                "action": action,
                "action_color": action_color,
                "drop_score": entry_check["drop_score"],
                "ma20_score": entry_check["ma20_score"],
                "vol_score": entry_check["vol_score"],
                "rsi_score": entry_check["rsi_score"],
            })

        if not scored:
            return []

        # Sort: passed first, then by score
        scored.sort(key=lambda x: (-x["passed"], -x["score"]))
        return scored[:top_n]


# =============================================================================
# Multi-ETF rotation backtest
# =============================================================================

def run_multi_etf_backtest(
    etf_codes: list[str] | None = None,
    initial_capital: float = 50_000,
    start_date: str | None = None,
    end_date: str | None = None,
    **strategy_params,
) -> dict:
    """Multi-ETF shock bounce rotation backtest.

    Each day:
      1. Check exits for all open positions
      2. If slots available (< max_concurrent), scan all ETFs for panic signals
      3. Enter best-scoring ETF that meets all conditions
      4. Cap: max_new_per_day new entries per day
    """
    from src.data.fetcher import fetch_etf_hist

    codes = etf_codes if etf_codes else [e["code"] for e in CANDIDATE_ETFS]
    strategy = UltraShortStrategy()
    params = {**strategy.get_default_params(), **strategy_params}

    take_profit = float(params["take_profit_pct"])
    stop_loss = float(params["stop_loss_pct"])
    max_hold = int(params["max_hold_days"])
    position_pct = float(params["position_pct"])
    max_concurrent = int(params.get("max_concurrent", 3))
    max_new_per_day = int(params.get("max_new_per_day", 1))
    max_gap = float(params.get("max_gap_pct", 0.02))
    trailing_activate = float(params.get("trailing_activate_pct", 0.01))
    trailing_giveback = float(params.get("trailing_giveback_pct", 0.005))

    # ── Load all ETF data ──
    print(f"加载 {len(codes)} 只ETF历史数据...")
    etf_data: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(fetch_etf_hist, c): c for c in codes}
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    df = df.sort_values("date", ascending=True).reset_index(drop=True)
                    df["_ma"] = df["close"].rolling(window=10, min_periods=10).mean()
                    df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()
                    etf_data[code] = df
            except Exception:
                pass

    if len(etf_data) < 2:
        return {"error": "数据不足", "equity_curve": pd.DataFrame(), "trades": [], "metrics": {}}

    print(f"成功加载 {len(etf_data)} 只ETF数据")

    # ── Unified date index ──
    all_dates = sorted(set().union(*[set(df["date"].tolist()) for df in etf_data.values()]))
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
            df_code = etf_data.get(code)
            if df_code is None:
                continue

            row_match = df_code[df_code["date"] == date]
            if row_match.empty:
                continue

            row = row_match.iloc[0]
            close = float(row["close"])
            open_price = float(row.get("open", close))
            pnl_pct = (close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
            hold_days = (date - pos["entry_date"]).days

            if pnl_pct > pos.get("highest_pnl", -999):
                pos["highest_pnl"] = pnl_pct

            exit_reason = ""
            exit_shares = 0

            # Gap protection
            prev_row = df_code[df_code["date"] < date]
            if not prev_row.empty:
                prev_close = float(prev_row.iloc[-1]["close"])
                if prev_close > 0:
                    gap_pct = (open_price - prev_close) / prev_close
                    if gap_pct <= -max_gap:
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

            # Take profit
            elif exit_shares == 0 and pnl_pct >= take_profit:
                exit_reason = f"目标止盈 +{pnl_pct:.1%}"
                exit_shares = pos["shares"]

            # Trailing stop
            elif exit_shares == 0 and pos.get("highest_pnl", -999) >= trailing_activate:
                giveback = pos["highest_pnl"] - pnl_pct
                if giveback >= trailing_giveback:
                    exit_reason = f"移动止盈 回撤{giveback:.1%}"
                    exit_shares = pos["shares"]

            if exit_shares >= 100:
                proceeds = exit_shares * close * 0.999
                cash += proceeds
                pnl = proceeds - pos["shares"] * pos["entry_price"]
                closed_trades.append({
                    "code": code,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": close,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": hold_days,
                    "exit_reason": exit_reason,
                    "winning": pnl > 0,
                })
                positions.remove(pos)

        # ── Step 2: Scan for new entries ──
        slots = max_concurrent - len(positions)
        if slots <= 0:
            pos_value = sum(
                p["shares"] * float(etf_data[p["code"]][etf_data[p["code"]]["date"] == date].iloc[0]["close"])
                for p in positions
                if not etf_data[p["code"]][etf_data[p["code"]]["date"] == date].empty
            )
            equity_rows.append({"date": date, "equity": cash + pos_value, "cash": cash, "positions": len(positions)})
            continue

        # Cap new entries per day
        new_today = 0
        candidates_today = []

        for code, df_code in etf_data.items():
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
            cost = shares * cand["close"] * 1.001  # 0.1% commission

            # Position size cap: max 25% of total equity
            est_equity = cash + sum(p["shares"] * cand["close"] for p in positions)
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
            rp = etf_data[p["code"]][etf_data[p["code"]]["date"] == date]
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
        df_code = etf_data.get(pos["code"])
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

    # Weekly return estimate
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
        f"超短波段 v2 恐慌反弹 多ETF轮动回测:\n"
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
