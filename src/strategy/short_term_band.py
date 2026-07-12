"""Enhanced Short-term Band Trading strategy — multi-factor entry, pyramid building, tiered exits.

Designed for small accounts (~2000 yuan) targeting ~3% weekly profit.
- Entry: 6-factor weighted scoring (0-100), ≥60 triggers buy
- Position: 3-layer pyramid (50%→30%→20%) for optimal cost averaging
- Exit: 5-tier system (hard stop / time stop / tiered profit / trailing stop / MA exits)
- ETF pool: ~35 candidates across broad/industry/defensive/cross-border/commodity
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
# Candidate ETF pool — ~35 ETFs, 5 categories, all affordable (< ¥15)
# =============================================================================

CANDIDATE_ETFS: list[dict] = [
    # ── 宽基指数 (Broad Market) — 8 ETFs ──
    {"code": "510300", "name": "沪深300ETF",        "approx_price": 3.9,  "category": "broad"},
    {"code": "510050", "name": "上证50ETF",          "approx_price": 2.7,  "category": "broad"},
    {"code": "510500", "name": "中证500ETF",          "approx_price": 5.8,  "category": "broad"},
    {"code": "159915", "name": "创业板ETF",           "approx_price": 2.2,  "category": "broad"},
    {"code": "588000", "name": "科创50ETF",           "approx_price": 0.9,  "category": "broad"},
    {"code": "512100", "name": "中证1000ETF",         "approx_price": 2.3,  "category": "broad"},
    {"code": "159949", "name": "创业板50ETF",         "approx_price": 0.9,  "category": "broad"},
    {"code": "510180", "name": "上证180ETF",          "approx_price": 3.5,  "category": "broad"},

    # ── 高波动行业 (High-volatility Industry) — 13 ETFs, core pool ──
    {"code": "512480", "name": "半导体ETF",           "approx_price": 1.2,  "category": "industry"},
    {"code": "512660", "name": "军工ETF",             "approx_price": 1.1,  "category": "industry"},
    {"code": "159766", "name": "新能源车ETF",         "approx_price": 1.0,  "category": "industry"},
    {"code": "516510", "name": "碳中和ETF",           "approx_price": 0.9,  "category": "industry"},
    {"code": "512980", "name": "传媒ETF",             "approx_price": 0.8,  "category": "industry"},
    {"code": "516880", "name": "光伏ETF(易方达)",     "approx_price": 0.7,  "category": "industry"},
    {"code": "516820", "name": "稀土ETF",             "approx_price": 1.0,  "category": "industry"},
    {"code": "512710", "name": "军工龙头ETF",         "approx_price": 0.7,  "category": "industry"},
    {"code": "159995", "name": "芯片ETF",             "approx_price": 1.1,  "category": "industry"},
    {"code": "515790", "name": "光伏ETF(华泰)",       "approx_price": 1.2,  "category": "industry"},
    {"code": "513330", "name": "恒生互联ETF",         "approx_price": 0.6,  "category": "industry"},
    {"code": "512690", "name": "酒ETF",               "approx_price": 1.5,  "category": "industry"},
    {"code": "516160", "name": "新能源ETF",           "approx_price": 0.9,  "category": "industry"},

    # ── 防御/逆周期 (Defensive/Counter-cyclical) — 5 ETFs ──
    {"code": "512010", "name": "医药ETF",             "approx_price": 0.5,  "category": "defensive"},
    {"code": "512170", "name": "医疗ETF",             "approx_price": 0.4,  "category": "defensive"},
    {"code": "512800", "name": "银行ETF",             "approx_price": 1.2,  "category": "defensive"},
    {"code": "516410", "name": "消费ETF",             "approx_price": 0.9,  "category": "defensive"},
    {"code": "512880", "name": "证券ETF",             "approx_price": 0.9,  "category": "defensive"},

    # ── 跨境 (Cross-border, T+0) — 4 ETFs ──
    {"code": "513100", "name": "纳指ETF",             "approx_price": 1.5,  "category": "cross_border"},
    {"code": "513050", "name": "中概互联ETF",          "approx_price": 1.2,  "category": "cross_border"},
    {"code": "159920", "name": "恒生ETF",             "approx_price": 1.1,  "category": "cross_border"},
    {"code": "513180", "name": "恒生科技ETF",          "approx_price": 0.7,  "category": "cross_border"},

    # ── 商品 (Commodity) — 1 ETF ──
    {"code": "518880", "name": "黄金ETF",             "approx_price": 4.5,  "category": "commodity"},
]


# =============================================================================
# Static helpers — shared between backtest, live signal, and ETF selection
# =============================================================================

def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized RSI(14) computation. Returns array same length as input.

    First *period* values are NaN (not enough data for first avg gain/loss).
    """
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing: first avg is simple mean, then EMA
    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (middle_band, upper_band, lower_band) as numpy arrays."""
    if len(closes) < period:
        nan_arr = np.full(len(closes), np.nan)
        return nan_arr, nan_arr, nan_arr

    middle = np.full(len(closes), np.nan)
    upper = np.full(len(closes), np.nan)
    lower = np.full(len(closes), np.nan)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std

    return middle, upper, lower


# =============================================================================
# Strategy class
# =============================================================================

class ShortTermBandStrategy(BaseStrategy):
    """Enhanced short-term band trading — multi-factor entry, pyramid, tiered exits.

    Entry: 6-factor weighted scoring (0-100).  Buy when score ≥ 60.
    Position: 3-layer pyramid (50% / 30% / 20%) for cost averaging.
    Exit: hard stop → time stop → tiered profit → trailing stop → MA exits.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "短线波段"

    @property
    def description(self) -> str:
        return (
            "增强版短线波段：6因子综合评分入场（阴线+均线+RSI+连跌+成交量），"
            "金字塔3层建仓（50%→30%→20%），5级分级出场（硬止损→时间→分级止盈→移动止损→MA辅助）。\n\n"
            "**入场**: 0-100综合评分≥60买入首仓，回调加仓至3层。\n"
            "**出场**: -3%硬止损 | +3%卖半仓/+5%清仓 | 盈利>2%保本止损 | 5天时间止盈。\n"
            "**选基**: 从31只ETF中按回测(40%)+波动率(25%)+趋势质量(20%)+流动性(15%)自动挑选。\n\n"
            "目标：周盈利3%，2000元级别小资金友好。"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return {
            # Entry
            "entry_score_threshold": 60,
            "ma_period": 10,
            "ma_proximity_pct": 0.03,
            "rsi_period": 14,
            "require_down_day": True,

            # Exit
            "take_profit_pct": 0.05,
            "partial_profit_pct": 0.03,
            "stop_loss_pct": 0.03,
            "max_hold_days": 5,
            "trailing_activate_pct": 0.02,
            "use_ma_exits": True,

            # Pyramid
            "pyramid_layers": 3,
            "pyramid_fractions": "50,30,20",
            "pyramid_thresholds": "60,70,80",

            # Position sizing
            "position_pct": 1.0,
            "max_etf_price": 15.0,

            # Supplementary
            "enable_macro_filter": True,
            "rotation_score_diff": 15,
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "entry_score_threshold": {
                "label": "入场评分门槛",
                "type": "slider",
                "min": 40, "max": 80, "step": 5,
                "help": "综合评分达到此值触发买入首仓（默认60分）。越高越保守",
            },
            "ma_period": {
                "label": "均线周期",
                "type": "number",
                "min": 5, "max": 30, "step": 1,
                "help": "参考均线天数（默认10日），入场评分中距离此项均线越近分越高",
            },
            "ma_proximity_pct": {
                "label": "均线偏离容忍度",
                "type": "slider",
                "min": 0.01, "max": 0.06, "step": 0.005,
                "help": "距均线多远仍能得分（默认3%），梯度递减至0",
            },
            "rsi_period": {
                "label": "RSI周期",
                "type": "number",
                "min": 7, "max": 21, "step": 1,
                "help": "RSI计算周期（默认14），用于超卖检测",
            },
            "require_down_day": {
                "label": "要求昨日下跌",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=昨天收阴线才能得那20分 | 关闭=忽略此因子",
            },
            "take_profit_pct": {
                "label": "全仓止盈线",
                "type": "slider",
                "min": 0.03, "max": 0.12, "step": 0.01,
                "help": "盈利达到此比例卖出剩余仓位（默认5%）",
            },
            "partial_profit_pct": {
                "label": "半仓止盈线",
                "type": "slider",
                "min": 0.02, "max": 0.06, "step": 0.005,
                "help": "盈利达到此比例先卖50%锁定利润（默认3%）",
            },
            "stop_loss_pct": {
                "label": "硬止损线",
                "type": "slider",
                "min": 0.02, "max": 0.08, "step": 0.01,
                "help": "亏损达到此比例立即全部卖出（默认3%），最高优先级",
            },
            "max_hold_days": {
                "label": "最长持有天数",
                "type": "number",
                "min": 3, "max": 10, "step": 1,
                "help": "超过此天数无论盈亏强制平仓（默认5天）",
            },
            "trailing_activate_pct": {
                "label": "移动止损激活线",
                "type": "slider",
                "min": 0.01, "max": 0.04, "step": 0.005,
                "help": "盈利超过此比例后止损线移到保本价（默认2%）",
            },
            "use_ma_exits": {
                "label": "MA辅助出场",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=盈利中跌破MA10卖半仓、跌破MA20全卖 | 关闭=只用止盈止损",
            },
            "pyramid_layers": {
                "label": "金字塔层数",
                "type": "number",
                "min": 1, "max": 3, "step": 1,
                "help": "分批建仓层数（默认3层）。设为1=一次性满仓",
            },
            "pyramid_fractions": {
                "label": "各层仓位比例",
                "type": "text",
                "help": "逗号分隔，如'50,30,20'表示首仓50%、第二层30%、第三层20%，总和应≤100",
            },
            "pyramid_thresholds": {
                "label": "各层触发评分",
                "type": "text",
                "help": "逗号分隔，如'60,70,80'对应各层入场评分要求",
            },
            "position_pct": {
                "label": "仓位比例",
                "type": "slider",
                "min": 0.3, "max": 1.0, "step": 0.1,
                "help": "可用资金的使用比例（2000元建议100%=满仓）",
            },
            "max_etf_price": {
                "label": "ETF最高单价",
                "type": "number",
                "min": 5.0, "max": 30.0, "step": 1.0,
                "help": "超过此价格的ETF不会被选入（100股起买，控制单笔金额）",
            },
            "enable_macro_filter": {
                "label": "宏观风险过滤",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=极端恐慌时暂停买入 | 关闭=忽略宏观情绪",
            },
            "rotation_score_diff": {
                "label": "轮动触发分差",
                "type": "number",
                "min": 5, "max": 30, "step": 5,
                "help": "另一只ETF评分高出此值时建议换仓（需当前不亏损）",
            },
        }

    # ------------------------------------------------------------------
    # Entry scoring (static — usable from both backtest and live signal)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entry_score(
        df_sorted: pd.DataFrame,
        idx: int,
        params: dict,
    ) -> dict:
        """Compute the multi-factor entry score (0-100) for bar at position *idx*.

        Returns dict with:
            total_score, down_day, ma10_dist, ma20_support,
            consecutive_down, rsi, volume, details (list[str])
        """
        row = df_sorted.iloc[idx]
        close = float(row["close"])

        require_down = str(params.get("require_down_day", "True")).lower() in ("true", "1", "yes")
        ma_period = int(params["ma_period"])
        ma_prox = float(params["ma_proximity_pct"])
        rsi_period = int(params.get("rsi_period", 14))

        details: list[str] = []

        # ── Factor 1: Yesterday down day (0 or 20) ──
        down_day_score = 0.0
        if idx >= 1:
            prev_close = float(df_sorted.iloc[idx - 1]["close"])
            prev_open = float(df_sorted.iloc[idx - 1]["open"])
            is_down = prev_close < prev_open
            if is_down:
                down_day_score = 20.0
                details.append(f"✅ 昨收阴({prev_close:.3f}<开{prev_open:.3f}) +20")
            else:
                if require_down:
                    details.append(f"❌ 昨收阳({prev_close:.3f}≥开{prev_open:.3f}) +0")
                else:
                    details.append(f"⚪ 昨收阳(未要求) +0")
        else:
            details.append("⚪ 无前日数据 +0")

        # ── Factor 2: Distance to MA (0-25) ──
        ma10_score = 0.0
        ma_val = None
        if "_ma" in df_sorted.columns:
            ma_val = df_sorted.at[idx, "_ma"]
            if pd.notna(ma_val) and ma_val > 0:
                dist = abs(close - ma_val) / ma_val
                ma10_score = round(25.0 * max(0, 1.0 - dist / ma_prox), 1)
                details.append(f"{'✅' if ma10_score>=15 else '🟡' if ma10_score>=8 else '❌'} "
                             f"距MA{ma_period} {dist:+.1%} → +{ma10_score:.0f}")
            else:
                details.append("⚪ MA无数据 +0")
        else:
            details.append("⚪ MA未计算 +0")

        # ── Factor 3: MA20 support (-15 to +15) ──
        ma20_score = 0.0
        if "_ma20" in df_sorted.columns:
            ma20_val = df_sorted.at[idx, "_ma20"]
            if pd.notna(ma20_val) and ma20_val > 0:
                if close >= ma20_val:
                    # Above MA20: bonus
                    bonus_pct = (close - ma20_val) / ma20_val
                    ma20_score = round(15.0 * min(1.0, bonus_pct / 0.02), 1)
                    details.append(f"{'✅' if ma20_score>=8 else '🟡'} "
                                 f"在MA20上方{bonus_pct:+.1%} +{ma20_score:.0f}")
                else:
                    # Below MA20: penalty
                    penalty_pct = (ma20_val - close) / ma20_val
                    ma20_score = round(-15.0 * min(1.0, penalty_pct / 0.03), 1)
                    details.append(f"{'🔴' if ma20_score<=-8 else '🟡'} "
                                 f"跌破MA20 {penalty_pct:.1%} {ma20_score:.0f}")
            else:
                details.append("⚪ MA20无数据 +0")
        else:
            details.append("⚪ MA20未计算 +0")

        # ── Factor 4: Consecutive down days (0-15) ──
        consecutive_score = 0.0
        if idx >= 1:
            consec = 0
            for j in range(idx, max(idx - 10, -1), -1):
                if j >= 1:
                    pc = float(df_sorted.iloc[j - 1]["close"])
                    po = float(df_sorted.iloc[j - 1]["open"])
                    if pc < po:
                        consec += 1
                    else:
                        break
            if consec == 0:
                consecutive_score = 0
            elif consec == 1:
                consecutive_score = 8
            elif consec in (2, 3):
                consecutive_score = 15
            else:
                consecutive_score = max(0, 15 - (consec - 3) * 5)
            details.append(f"{'✅' if consecutive_score>=15 else '🟡' if consecutive_score>=8 else '❌'} "
                         f"连跌{consec}天 +{consecutive_score:.0f}")
        else:
            details.append("⚪ 无前日数据 +0")

        # ── Factor 5: RSI oversold (0-20, including bonus) ──
        rsi_score = 0.0
        rsi_val = None
        if idx >= rsi_period:
            closes_arr = df_sorted["close"].values[:idx + 1].astype(float)
            rsi_arr = _compute_rsi(closes_arr, rsi_period)
            rsi_val = rsi_arr[idx]
            if pd.notna(rsi_val):
                if rsi_val < 25:
                    rsi_score = 20.0  # 15 base + 5 extreme bonus
                elif rsi_val < 30:
                    rsi_score = 15.0
                elif rsi_val < 35:
                    rsi_score = 12.0
                elif rsi_val < 40:
                    rsi_score = 8.0
                elif rsi_val <= 60:
                    rsi_score = 4.0
                elif rsi_val > 70:
                    rsi_score = -5.0
                else:
                    rsi_score = 0.0
                icon = "✅" if rsi_score >= 15 else "🟡" if rsi_score >= 8 else "⚠️" if rsi_score < 0 else "❌"
                details.append(f"{icon} RSI({rsi_period})={rsi_val:.1f} → {rsi_score:+.0f}")
            else:
                details.append("⚪ RSI无数据 +0")
        else:
            details.append(f"⚪ 数据不足(需{rsi_period}根K线) +0")

        # ── Factor 6: Volume confirmation (0-10) ──
        volume_score = 0.0
        if idx >= 5 and "volume" in df_sorted.columns:
            vol_today = float(row.get("volume", 0) or 0)
            avg_vol_5 = float(np.mean([
                float(df_sorted.iloc[i].get("volume", 0) or 0)
                for i in range(max(0, idx - 5), idx)
            ]))
            if avg_vol_5 > 0:
                vol_ratio = vol_today / avg_vol_5
                if vol_ratio >= 1.0:
                    volume_score = 10.0
                elif vol_ratio >= 0.8:
                    volume_score = 5.0
                details.append(f"{'✅' if volume_score>=10 else '🟡' if volume_score>=5 else '❌'} "
                             f"量比{vol_ratio:.1f}x +{volume_score:.0f}")
            else:
                details.append("⚪ 均量=0 +0")
        else:
            details.append("⚪ 量数据不足 +0")

        total = round(down_day_score + ma10_score + ma20_score + consecutive_score + rsi_score + volume_score, 1)
        total = max(0.0, min(100.0, total))

        return {
            "total_score": total,
            "down_day": down_day_score,
            "ma10_dist": ma10_score,
            "ma20_support": ma20_score,
            "consecutive_down": consecutive_score,
            "rsi": rsi_score,
            "rsi_value": rsi_val,
            "volume": volume_score,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Backtest signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}

        # ── Parse params ──
        take_profit = float(params["take_profit_pct"])
        partial_profit = float(params["partial_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        ma_period = int(params["ma_period"])
        position_pct = float(params["position_pct"])
        use_ma_exits = str(params.get("use_ma_exits", "True")).lower() in ("true", "1", "yes")
        enable_macro = str(params.get("enable_macro_filter", "True")).lower() in ("true", "1", "yes")
        pyramid_layers = int(params.get("pyramid_layers", 3))

        # Pyramid fractions
        frac_str = str(params.get("pyramid_fractions", "50,30,20"))
        pyramid_fractions = [float(x) / 100 for x in frac_str.split(",")]
        # Ensure we have enough fractions
        while len(pyramid_fractions) < pyramid_layers:
            pyramid_fractions.append(0.0)
        pyramid_fractions = pyramid_fractions[:pyramid_layers]

        # Pyramid thresholds
        thr_str = str(params.get("pyramid_thresholds", "60,70,80"))
        pyramid_thresholds = [float(x) for x in thr_str.split(",")]
        while len(pyramid_thresholds) < pyramid_layers:
            pyramid_thresholds.append(100.0)
        pyramid_thresholds = pyramid_thresholds[:pyramid_layers]

        # Macro filter
        macro_pulse = kwargs.get("macro_pulse")
        macro_suppress = False
        macro_penalty = 0
        if enable_macro and macro_pulse is not None:
            if macro_pulse.risk_level == "extreme":
                macro_suppress = True
            elif macro_pulse.risk_level == "high":
                macro_penalty = 10

        # Backtest capital
        backtest_capital = float(kwargs.get("backtest_capital", 100_000))

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        if "close" not in df.columns:
            df["signal"] = "hold"; df["signal_price"] = 0.0
            df["signal_shares"] = 0; df["signal_reason"] = ""
            return df

        # ── Precompute indicators ──
        df["_ma"] = df["close"].rolling(window=ma_period, min_periods=ma_period).mean()
        df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()

        # ── Init signal columns ──
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        # ── State ──
        in_position = False
        entry_price = 0.0       # weighted average entry
        entry_idx = -1
        total_entry_shares = 0
        pyramid_layer = 0
        partial_taken = False
        trailing_active = False
        highest_since_entry = 0.0
        total_budget = position_pct * backtest_capital

        for i in range(len(df)):
            close = float(df.at[i, "close"])

            # ── Exit logic (in position) ──
            if in_position:
                pnl_pct = (close - entry_price) / entry_price if entry_price > 0 else 0
                hold_days = i - entry_idx

                # Track highest price for trailing stop
                if close > highest_since_entry:
                    highest_since_entry = close

                # Activate trailing stop?
                if pnl_pct >= float(params["trailing_activate_pct"]) and not trailing_active:
                    trailing_active = True

                effective_stop = -stop_loss
                if trailing_active:
                    effective_stop = max(effective_stop, 0.0)  # move to breakeven

                exit_reason = ""
                exit_shares = 0
                is_partial = False

                # Tier 1: Hard stop (highest priority)
                if pnl_pct <= -stop_loss:
                    exit_reason = (
                        f"🛑 硬止损：亏损 {pnl_pct:.1%}（≤-{stop_loss:.0%}），"
                        f"入场 {entry_price:.3f}，出场 {close:.3f}"
                    )
                    exit_shares = total_entry_shares

                # Tier 2: Time stop
                elif hold_days >= max_hold:
                    exit_reason = (
                        f"⏰ 时间止盈：持有 {hold_days} 天（≥{max_hold}天），"
                        f"盈亏 {pnl_pct:+.1%}，入场 {entry_price:.3f}，出场 {close:.3f}"
                    )
                    exit_shares = total_entry_shares

                # Tier 3: Tiered profit taking
                elif pnl_pct >= take_profit:
                    exit_reason = (
                        f"🎯 全仓止盈：盈利 {pnl_pct:.1%}（≥{take_profit:.0%}），"
                        f"入场 {entry_price:.3f}，出场 {close:.3f}"
                    )
                    exit_shares = total_entry_shares

                elif partial_profit < take_profit and pnl_pct >= partial_profit and not partial_taken:
                    # Sell 50% of position
                    half = (total_entry_shares // 200) * 100
                    if half >= 100:
                        exit_reason = (
                            f"📈 半仓止盈：盈利 {pnl_pct:.1%}（≥{partial_profit:.0%}），"
                            f"卖出 {half} 股锁定利润，剩余 {total_entry_shares - half} 股继续持有"
                        )
                        exit_shares = half
                        is_partial = True

                # Tier 4: Trailing stop
                elif trailing_active and pnl_pct <= effective_stop:
                    exit_reason = (
                        f"🔒 移动止损：保本退出，入场 {entry_price:.3f}，出场 {close:.3f}，"
                        f"最高盈利 {((highest_since_entry-entry_price)/entry_price):.1%}"
                    )
                    exit_shares = total_entry_shares

                # Tier 5: MA exits (in profit only)
                elif use_ma_exits and pnl_pct > 0:
                    ma_val = df.at[i, "_ma"]
                    ma20_val = df.at[i, "_ma20"]
                    if pd.notna(ma20_val) and close < ma20_val:
                        exit_reason = (
                            f"📉 MA20破位：现价 {close:.3f} < MA20 {ma20_val:.3f}，"
                            f"盈利 {pnl_pct:+.1%}，全部卖出"
                        )
                        exit_shares = total_entry_shares
                    elif pd.notna(ma_val) and close < ma_val and not partial_taken:
                        half = (total_entry_shares // 200) * 100
                        if half >= 100:
                            exit_reason = (
                                f"📉 MA{ma_period}破位：现价 {close:.3f} < MA{ma_period} {ma_val:.3f}，"
                                f"盈利 {pnl_pct:+.1%}，卖出半仓"
                            )
                            exit_shares = half
                            is_partial = True

                # Execute exit
                if exit_shares >= 100:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = exit_shares
                    df.at[i, "signal_reason"] = exit_reason

                    if is_partial:
                        total_entry_shares -= exit_shares
                        partial_taken = True
                        if total_entry_shares < 100:
                            in_position = False
                    else:
                        in_position = False
                        total_entry_shares = 0

            # ── Entry / pyramid logic (not in full position) ──
            else:
                if pd.isna(df.at[i, "_ma"]):
                    continue

                # Macro suppression
                if macro_suppress:
                    continue

                entry_result = self._compute_entry_score(df, i, params)
                score = entry_result["total_score"] - macro_penalty

                # Determine which pyramid layer to enter
                target_layer = 0
                for layer_idx in range(pyramid_layer, pyramid_layers):
                    if (layer_idx == 0 and score >= pyramid_thresholds[0]) or \
                       (layer_idx > 0 and score >= pyramid_thresholds[layer_idx]):
                        target_layer = layer_idx + 1
                        # Layer 2+ requires price < layer 1 entry or better score
                        if layer_idx > 0 and in_position:
                            if close >= entry_price * 0.99 and score < pyramid_thresholds[layer_idx] + 5:
                                continue  # skip — not enough pullback
                        break

                if target_layer == 0:
                    continue

                # Calculate budget for this layer
                if not in_position:
                    fraction = pyramid_fractions[0]
                else:
                    fraction = pyramid_fractions[target_layer - 1]

                # If single-layer mode, use 100%
                if pyramid_layers == 1:
                    fraction = 1.0

                layer_budget = total_budget * fraction
                layer_shares = max(100, int(layer_budget / close) // 100 * 100)

                if layer_shares < 100:
                    continue

                # Calculate full position cap
                max_possible_shares = max(100, int(total_budget / close) // 100 * 100)
                new_total = (total_entry_shares + layer_shares) if in_position else layer_shares

                if new_total > max_possible_shares:
                    layer_shares = max_possible_shares - total_entry_shares
                    if layer_shares < 100:
                        continue

                # Build reason
                if not in_position:
                    layer_label = "首仓"
                    pyramid_layer = 1
                else:
                    pyramid_layer = target_layer
                    layer_label = f"第{pyramid_layer}层加仓"

                reason_parts = [f"🎯 {layer_label}：评分{score:.0f}分"]
                reason_parts.append(f"入场价{close:.3f}")
                if in_position:
                    reason_parts.append(f"均价→{(entry_price*total_entry_shares+close*layer_shares)/(total_entry_shares+layer_shares):.3f}")
                reason_parts.extend(entry_result["details"][:3])  # top 3 reasons
                reason = " | ".join(reason_parts)

                # Average in
                if in_position:
                    total_cost = entry_price * total_entry_shares + close * layer_shares
                    total_entry_shares += layer_shares
                    entry_price = total_cost / total_entry_shares
                else:
                    entry_price = close
                    total_entry_shares = layer_shares
                    in_position = True
                    highest_since_entry = close

                entry_idx = i
                partial_taken = False
                trailing_active = False

                df.at[i, "signal"] = "buy"
                df.at[i, "signal_price"] = close
                df.at[i, "signal_shares"] = layer_shares
                df.at[i, "signal_reason"] = reason

        # ── Cleanup ──
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

        if "_ma" not in df.columns:
            ma_period = int(params["ma_period"])
            df = df.copy()
            df["_ma"] = df["close"].rolling(window=ma_period, min_periods=ma_period).mean()
            df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

        current_price = info.get("current_price") if info else None
        if current_price is None:
            current_price = float(df_sorted.iloc[-1]["close"])

        take_profit = float(params["take_profit_pct"])
        partial_profit = float(params["partial_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        enable_macro = str(params.get("enable_macro_filter", "True")).lower() in ("true", "1", "yes")

        # ── Macro filter ──
        macro_pulse = kwargs.get("macro_pulse")
        macro_suppress = False
        macro_penalty = 0
        if enable_macro and macro_pulse is not None:
            if macro_pulse.risk_level == "extreme":
                macro_suppress = True
            elif macro_pulse.risk_level == "high":
                macro_penalty = 10

        # ── Portfolio context ──
        pf = kwargs.get("portfolio_context") or {}
        has_position = pf.get("has_position", False)
        holding_cost = pf.get("holding_avg_cost")
        holding_shares = pf.get("holding_shares", 0)
        last_buy_date = pf.get("last_buy_date")
        partial_taken = pf.get("partial_profit_taken", False)
        current_layer = pf.get("pyramid_layer", 0)

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

            # Tier 1: Hard stop
            if pnl_pct <= -stop_loss:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"亏损 {pnl_pct:.1%}，触发硬止损 -{stop_loss:.0%}",
                    reason=f"🛑 硬止损！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（{pnl_pct:.1%}），建议全部卖出。",
                    urgency_level="high", current_zone="止损区",
                )

            # Tier 2: Time stop
            if hold_days >= max_hold:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"持有 {hold_days} 天，触发时间止盈",
                    reason=f"⏰ 时间到！持有 {hold_days} 天（≥{max_hold}天），盈亏 {pnl_pct:+.1%}，建议卖出换标的。",
                    urgency_level="medium", current_zone="时间止盈区",
                )

            # Tier 3: Full take-profit
            if pnl_pct >= take_profit:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"盈利 {pnl_pct:.1%}，触发全仓止盈 {take_profit:.0%}",
                    reason=f"🎯 止盈！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（+{pnl_pct:.1%}），建议全部卖出。",
                    urgency_level="high", current_zone="止盈区",
                )

            # Tier 3b: Partial take-profit
            if partial_profit < take_profit and pnl_pct >= partial_profit and not partial_taken:
                half = (holding_shares // 200) * 100
                if half >= 100:
                    return LiveSignal(
                        action="sell", current_price=round(current_price, 4),
                        suggested_shares=half,
                        suggested_amount=round(half * current_price, 2),
                        trigger_description=f"盈利 {pnl_pct:.1%}，触发半仓止盈 {partial_profit:.0%}",
                        reason=f"📈 半仓止盈：卖出 {half} 股锁定利润（¥{half*current_price:.0f}），剩余 {holding_shares-half} 股博取更高收益。",
                        urgency_level="high", current_zone="半仓止盈区",
                    )

            # Tier 5: MA exits (in profit)
            if pnl_pct > 0 and str(params.get("use_ma_exits", "True")).lower() in ("true", "1", "yes"):
                ma_val = float(df_sorted.iloc[-1]["_ma"]) if pd.notna(df_sorted.iloc[-1].get("_ma")) else None
                ma20_val = float(df_sorted.iloc[-1]["_ma20"]) if pd.notna(df_sorted.iloc[-1].get("_ma20")) else None
                if ma20_val and current_price < ma20_val:
                    return LiveSignal(
                        action="sell", current_price=round(current_price, 4),
                        suggested_shares=holding_shares,
                        trigger_description=f"跌破MA20({ma20_val:.3f})，建议全卖",
                        reason=f"📉 MA20破位：现价 {current_price:.3f} < MA20 {ma20_val:.3f}，盈利 {pnl_pct:+.1%}，建议全部卖出。",
                        urgency_level="medium", current_zone="MA20破位",
                    )

            # Still holding
            next_tp = round(holding_cost * (1 + take_profit), 4)
            next_sl = round(holding_cost * (1 - stop_loss), 4)
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                trigger_description=f"持仓中 盈亏{pnl_pct:+.1%} 第{hold_days}天",
                next_trigger_price=next_tp,
                reason=f"📌 持仓中：入场 ¥{holding_cost:.3f}，现价 ¥{current_price:.3f}（{pnl_pct:+.1%}），止盈 ¥{next_tp}，止损 ¥{next_sl}",
                urgency_level="low",
                portions_used=hold_days, portions_total=max_hold,
                current_zone=f"持仓中 第{hold_days}天 第{current_layer}层",
            )

        # ── Not in position (or pyramiding): check entry ──
        if macro_suppress:
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                reason="⚠️ 宏观情绪极端恐慌，暂停所有买入操作。等待情绪恢复后再入场。",
                urgency_level="low", current_zone="宏观暂停区",
            )

        # Need at least ma_period bars
        if len(df_sorted) < int(params["ma_period"]) + 1:
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                reason=f"数据不足（需要至少{int(params['ma_period'])+1}个交易日）",
                urgency_level="low",
            )

        # Compute entry score for today
        # We append a synthetic "today" row to make the scorer work
        last_row = df_sorted.iloc[-1:].copy()
        last_row["close"] = current_price
        # For the score, we use idx = len(df_sorted) - 1 (yesterday's data) + today's price
        entry_result = self._compute_entry_score(df_sorted, len(df_sorted) - 1, params)

        # Override close-based scores with current price
        ma_val = df_sorted.at[len(df_sorted) - 1, "_ma"]
        if pd.notna(ma_val) and ma_val > 0:
            dist = abs(current_price - ma_val) / ma_val
            ma_prox = float(params["ma_proximity_pct"])
            new_ma_score = round(25.0 * max(0, 1.0 - dist / ma_prox), 1)
            entry_result["total_score"] = round(
                entry_result["total_score"] - entry_result["ma10_dist"] + new_ma_score, 1
            )
            entry_result["ma10_dist"] = new_ma_score

        score = max(0, min(100, entry_result["total_score"] - macro_penalty))

        # Determine pyramid layer
        pyramid_layers = int(params.get("pyramid_layers", 3))
        frac_str = str(params.get("pyramid_fractions", "50,30,20"))
        pyramid_fractions = [float(x) / 100 for x in frac_str.split(",")]
        while len(pyramid_fractions) < pyramid_layers:
            pyramid_fractions.append(0.0)
        thr_str = str(params.get("pyramid_thresholds", "60,70,80"))
        pyramid_thresholds = [float(x) for x in thr_str.split(",")]
        while len(pyramid_thresholds) < pyramid_layers:
            pyramid_thresholds.append(100.0)

        target_layer = 0
        for layer_idx in range(current_layer, pyramid_layers):
            if score >= pyramid_thresholds[layer_idx]:
                target_layer = layer_idx + 1
                break

        if target_layer == 0 or score < pyramid_thresholds[0]:
            # Not enough score — tell user what's missing
            suggestions = []
            for d in entry_result["details"]:
                if "❌" in d:
                    suggestions.append(d.split(" ")[1] if " " in d else d)
            wait_reason = f"⏳ 评分 {score:.0f}/100（需≥{pyramid_thresholds[0]:.0f}）。"
            if suggestions:
                wait_reason += f" 短板：{'、'.join(suggestions[:2])}。"
            return LiveSignal(
                action="wait_for_drop", current_price=round(current_price, 4),
                trigger_description=f"入场评分 {score:.0f}/100",
                next_trigger_price=round(float(ma_val), 4) if pd.notna(ma_val) else None,
                reason=wait_reason + f" 等回调到MA{params['ma_period']}附近（¥{ma_val:.3f}）再入场。",
                urgency_level="low", current_zone="等待回调",
            )

        # Calculate shares
        available_cash = 2000.0
        if pf and pf.get("available_cash", 0) > 0:
            available_cash = float(pf["available_cash"])
        if has_position and holding_shares > 0 and holding_cost:
            # Pyramiding: use remaining budget
            if current_layer < pyramid_layers:
                fraction = pyramid_fractions[current_layer]
                budget = available_cash * float(params["position_pct"]) * fraction
            else:
                budget = 0
        else:
            budget = available_cash * float(params["position_pct"]) * pyramid_fractions[0]

        if pyramid_layers == 1:
            budget = available_cash * float(params["position_pct"])

        raw_shares = int(budget / current_price) if current_price > 0 else 0
        shares = max(100, (raw_shares // 100) * 100)
        if shares < 100:
            return LiveSignal(
                action="wait_for_drop", current_price=round(current_price, 4),
                reason=f"资金不足（需¥{current_price*100:.0f}买100股，可用¥{budget:.0f}）",
                urgency_level="low", current_zone="等待资金",
            )

        amount = round(shares * current_price, 2)
        layer_label = "首仓" if not has_position else f"第{target_layer}层加仓"

        return LiveSignal(
            action="buy", current_price=round(current_price, 4),
            suggested_shares=shares, suggested_amount=amount,
            trigger_description=f"入场评分 {score:.0f}/100 · {layer_label}",
            next_trigger_price=round(float(ma_val), 4) if pd.notna(ma_val) else None,
            reason=(
                f"🎯 买入信号（{layer_label}）！综合评分 {score:.0f}/100。\n"
                + "\n".join(entry_result["details"])
                + f"\n\n建议买入 {shares} 股（≈¥{amount:.0f}），"
                + f"止盈 +{take_profit:.0%}（¥{current_price*(1+take_profit):.3f}），"
                + f"止损 -{stop_loss:.0%}（¥{current_price*(1-stop_loss):.3f}）。"
            ),
            urgency_level="high",
            current_zone=f"买入区 · {layer_label}",
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
        ma_period = int(params["ma_period"])
        rsi_period = int(params.get("rsi_period", 14))
        take_profit = float(params["take_profit_pct"])
        partial_profit = float(params["partial_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
        closes = df_sorted["close"].values

        # Ensure MAs
        if "_ma" not in df_sorted.columns:
            df_sorted["_ma"] = df_sorted["close"].rolling(window=ma_period, min_periods=ma_period).mean()
        if "_ma20" not in df_sorted.columns:
            df_sorted["_ma20"] = df_sorted["close"].rolling(window=20, min_periods=20).mean()

        current_price = info.get("current_price") if info else None
        if current_price is None and len(df_sorted) > 0:
            current_price = float(df_sorted.iloc[-1]["close"])

        cards: list[DashboardCard] = []

        # ── Card 1: Entry score breakdown ──
        if len(df_sorted) >= ma_period + 1:
            try:
                entry_result = self._compute_entry_score(df_sorted, len(df_sorted) - 1, params)
            except Exception:
                entry_result = None
        else:
            entry_result = None

        if entry_result:
            score = entry_result["total_score"]
            threshold = int(params.get("entry_score_threshold", 60))

            # Build factor breakdown with individual scores
            factors = [
                {"label": "昨日收阴", "score": int(entry_result["down_day"]), "max": 20},
                {"label": f"距MA{ma_period}", "score": int(entry_result["ma10_dist"]), "max": 25},
                {"label": "MA20支撑", "score": int(entry_result["ma20_support"]), "max": 15},
                {"label": "连跌天数", "score": int(entry_result["consecutive_down"]), "max": 15},
                {"label": f"RSI({rsi_period})", "score": int(entry_result["rsi"]), "max": 20},
                {"label": "成交量", "score": int(entry_result["volume"]), "max": 10},
            ]

            cards.append(DashboardCard(
                card_id="entry_score",
                title=f"入场评分 · {score:.0f}/100（需≥{threshold}）",
                card_type="progress",
                content={
                    "value_pct": score,
                    "max_value": 100,
                    "threshold": threshold,
                    "factors": factors,
                    "details": entry_result["details"],
                    "ready": score >= threshold,
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
                    {"label": "硬止损", "value": f"-{stop_loss:.0%}", "priority": "最高"},
                    {"label": "时间止盈", "value": f"持有{max_hold}天", "priority": "高"},
                    {"label": "全仓止盈", "value": f"+{take_profit:.0%}", "priority": "高"},
                    {"label": "半仓止盈", "value": f"+{partial_profit:.0%}", "priority": "中"},
                    {"label": "移动止损", "value": "盈利>2%保本", "priority": "中"},
                    {"label": "MA辅助出场", "value": "破MA10半/MA20全", "priority": "低"},
                ],
            },
            priority=1,
        ))

        # ── Card 3: Market context ──
        ma_val = df_sorted.at[len(df_sorted) - 1, "_ma"] if len(df_sorted) > 0 else None
        ma20_val = df_sorted.at[len(df_sorted) - 1, "_ma20"] if len(df_sorted) > 0 else None

        # RSI
        rsi_val = None
        if len(closes) >= rsi_period + 1:
            rsi_arr = _compute_rsi(closes.astype(float), rsi_period)
            rsi_val = round(float(rsi_arr[-1]), 1) if pd.notna(rsi_arr[-1]) else None

        # Volatility
        last_5 = closes[-5:] if len(closes) >= 5 else closes
        vol_5d = float(np.std(last_5) / np.mean(last_5)) if len(last_5) > 1 and np.mean(last_5) > 0 else 0

        # Consecutive down days
        consec = 0
        for j in range(len(df_sorted) - 1, max(len(df_sorted) - 11, -1), -1):
            if j >= 1:
                pc = float(df_sorted.iloc[j - 1]["close"])
                po = float(df_sorted.iloc[j - 1]["open"])
                if pc < po:
                    consec += 1
                else:
                    break

        ma10_dist = (current_price - ma_val) / ma_val if current_price and ma_val else None

        cards.append(DashboardCard(
            card_id="market_context",
            title="市场环境",
            card_type="info",
            content={
                "current_price": round(current_price, 4) if current_price else None,
                "ma_value": round(float(ma_val), 4) if pd.notna(ma_val) else None,
                "ma20_value": round(float(ma20_val), 4) if pd.notna(ma20_val) else None,
                "ma10_dist_pct": round(float(ma10_dist) * 100, 2) if ma10_dist is not None else None,
                "rsi": rsi_val,
                "consecutive_down": consec,
                "volatility_5d": f"{vol_5d:.2%}",
            },
            priority=2,
        ))

        # ── Card 4: Pyramid status (if in position) ──
        pf = kwargs.get("portfolio_context") or {}
        if pf.get("has_position"):
            current_layer = pf.get("pyramid_layer", 1)
            total_layers = int(params.get("pyramid_layers", 3))
            cards.append(DashboardCard(
                card_id="pyramid_status",
                title="金字塔建仓",
                card_type="progress",
                content={
                    "value_pct": current_layer / total_layers * 100 if total_layers > 0 else 100,
                    "label": f"第{current_layer}/{total_layers}层",
                    "max_value": 100,
                },
                priority=1,
            ))

        # ── Card 5: Macro filter status ──
        macro_pulse = kwargs.get("macro_pulse")
        if macro_pulse is not None and macro_pulse.total_signals > 0:
            if macro_pulse.risk_level in ("extreme", "high"):
                cards.append(DashboardCard(
                    card_id="macro_warning",
                    title="宏观风险警告",
                    card_type="warning",
                    content={
                        "message": f"⚠️ 宏观情绪{macro_pulse.risk_level}，{'暂停买入' if macro_pulse.risk_level=='extreme' else '提高入场门槛+10分'}",
                    },
                    priority=1,
                ))

        return cards

    # =====================================================================
    # ETF selection (static methods)
    # =====================================================================

    @staticmethod
    def get_candidate_pool() -> list[dict]:
        return list(CANDIDATE_ETFS)

    # -- Scoring helpers ----------------------------------------------------

    @staticmethod
    def _compute_volatility_score(info: dict) -> float:
        amplitude = info.get("amplitude", 0) or 0
        turnover = info.get("turnover_rate", 0) or 0
        volume = info.get("volume", 0) or 0
        return round(amplitude * 40 + turnover * 30 + min(volume / 100_000_000, 1) * 30, 2)

    @staticmethod
    def _compute_liquidity_score(info: dict) -> float:
        """Score based on bid-ask spread tightness (0-15).

        Tighter spreads = better for small accounts where wide spreads eat profits.
        Falls back to turnover-only if bid/ask data unavailable.
        """
        bid1 = info.get("bid1_price")
        ask1 = info.get("ask1_price")
        current = info.get("current_price")

        spread_score = 0.0
        if bid1 and ask1 and current and current > 0:
            spread_pct = (ask1 - bid1) / current
            # Spread < 0.1% = perfect (10 pts), < 0.3% = good (7), < 0.5% = ok (4)
            if spread_pct < 0.001:
                spread_score = 10.0
            elif spread_pct < 0.003:
                spread_score = 7.0
            elif spread_pct < 0.005:
                spread_score = 4.0
            else:
                spread_score = 1.0
        else:
            spread_score = 3.0  # neutral if no data

        turnover = info.get("turnover_rate", 0) or 0
        turnover_score = min(turnover * 20, 5.0)  # 0-5 points

        return round(spread_score + turnover_score, 2)

    @staticmethod
    def _compute_trend_quality(
        hist_df: pd.DataFrame,
        params: dict,
    ) -> float:
        """Score the recent pullback quality: are we at a good entry point?

        Runs entry scoring on the last 5 bars, returns the best score.
        A higher score means better entry timing right now.
        """
        if hist_df is None or hist_df.empty or len(hist_df) < 15:
            return 0.0

        df_sorted = hist_df.sort_values("date", ascending=True).reset_index(drop=True)

        # Ensure MAs
        ma_period = int(params.get("ma_period", 10))
        df_sorted["_ma"] = df_sorted["close"].rolling(window=ma_period, min_periods=ma_period).mean()
        df_sorted["_ma20"] = df_sorted["close"].rolling(window=20, min_periods=20).mean()

        best_score = 0.0
        start_idx = max(0, len(df_sorted) - 5)
        for idx in range(start_idx, len(df_sorted)):
            try:
                result = ShortTermBandStrategy._compute_entry_score(df_sorted, idx, params)
                if result["total_score"] > best_score:
                    best_score = result["total_score"]
            except Exception:
                continue

        return round(best_score, 1)

    @staticmethod
    def _simulate_trades_from_signals(sig_df: pd.DataFrame) -> list[dict]:
        df = sig_df.sort_values("date", ascending=True).reset_index(drop=True)
        trades: list[dict] = []
        open_entry: dict | None = None

        for i in range(len(df)):
            row = df.iloc[i]
            action = row.get("signal", "hold")
            price = row.get("signal_price", row["close"])
            date = row["date"]

            if action == "buy" and open_entry is None:
                open_entry = {"entry_date": date, "entry_price": float(price)}
            elif action == "sell" and open_entry is not None:
                exit_price = float(price)
                pnl_pct = (exit_price - open_entry["entry_price"]) / open_entry["entry_price"]
                holding_days = max((date - open_entry["entry_date"]).days, 1)
                trades.append({
                    "pnl_pct": round(pnl_pct, 6),
                    "winning": pnl_pct > 0,
                    "holding_days": holding_days,
                    "entry_date": open_entry["entry_date"],
                    "exit_date": date,
                    "entry_price": open_entry["entry_price"],
                    "exit_price": exit_price,
                })
                open_entry = None

        if open_entry is not None:
            final_row = df.iloc[-1]
            final_price = float(final_row["close"])
            pnl_pct = (final_price - open_entry["entry_price"]) / open_entry["entry_price"]
            holding_days = max((final_row["date"] - open_entry["entry_date"]).days, 1)
            trades.append({
                "pnl_pct": round(pnl_pct, 6),
                "winning": pnl_pct > 0,
                "holding_days": holding_days,
                "entry_date": open_entry["entry_date"],
                "exit_date": final_row["date"],
                "entry_price": open_entry["entry_price"],
                "exit_price": final_price,
            })

        return trades

    @staticmethod
    def _compute_backtest_score(trades: list[dict]) -> float:
        if not trades:
            return 0.0

        total_return_pct = sum(t["pnl_pct"] for t in trades)
        win_count = sum(1 for t in trades if t["winning"])
        win_rate = win_count / len(trades)
        trade_count = len(trades)

        score = (
            (total_return_pct * 100) * 0.5
            + (win_rate * 100) * 0.3
            + min(trade_count, 5) / 5.0 * 20
        )

        if win_rate == 0.0 and total_return_pct < 0:
            score *= 0.1

        return round(score, 2)

    @staticmethod
    def _run_mini_backtest(
        code: str,
        hist_cache: dict[str, pd.DataFrame | None],
    ) -> tuple[float, list[dict]]:
        cached = hist_cache.get(code)
        if cached is None and code in hist_cache:
            return 0.0, []

        if code not in hist_cache:
            from src.data.fetcher import fetch_etf_hist
            try:
                hist_cache[code] = fetch_etf_hist(code)
            except Exception:
                hist_cache[code] = None
                return 0.0, []

        hist_df = hist_cache[code]
        if hist_df is None or hist_df.empty:
            return 0.0, []

        strategy = ShortTermBandStrategy()
        sig_df = strategy.generate_signals(hist_df)
        trades = ShortTermBandStrategy._simulate_trades_from_signals(sig_df)
        score = ShortTermBandStrategy._compute_backtest_score(trades)
        return score, trades

    # -- Main selector ----------------------------------------------------

    @staticmethod
    def select_best_etf(
        candidates: list[dict] | None = None,
        max_price: float = 20.0,
    ) -> dict | None:
        """Select the best ETF for short-term band trading.

        Scores each candidate on 4 dimensions:
        1. Mini-backtest (40%) — historical strategy performance
        2. Volatility (25%) — amplitude × turnover × volume
        3. Trend quality (20%) — current pullback / entry timing
        4. Liquidity (15%) — bid-ask spread tightness

        Returns:
            Best candidate dict, or None if no suitable ETF found.
        """
        from src.data.fetcher import fetch_etf_info, fetch_etf_hist

        pool = candidates if candidates is not None else list(CANDIDATE_ETFS)

        # Phase 1 — parallel historical data fetch
        hist_cache: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(fetch_etf_hist, etf["code"]): etf["code"]
                for etf in pool
            }
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    hist_cache[code] = future.result()
                except Exception:
                    hist_cache[code] = None

        # Phase 2 — score each candidate
        scored: list[dict] = []
        params = ShortTermBandStrategy().get_default_params()

        for etf in pool:
            code = etf["code"]

            try:
                info = fetch_etf_info(code)
            except Exception:
                continue

            current_price = info.get("current_price")
            if current_price is None or current_price <= 0:
                continue
            if current_price > max_price:
                continue

            # 1. Volatility score
            vol_score = ShortTermBandStrategy._compute_volatility_score(info)

            # 2. Liquidity score
            liq_score = ShortTermBandStrategy._compute_liquidity_score(info)

            # 3. Mini-backtest
            try:
                bt_score, trades = ShortTermBandStrategy._run_mini_backtest(code, hist_cache)
            except Exception:
                bt_score = 0.0
                trades = []

            # 4. Trend quality
            hist_df = hist_cache.get(code)
            trend_score = ShortTermBandStrategy._compute_trend_quality(hist_df, params) if hist_df is not None else 0.0

            # Composite: 40/25/20/15
            if bt_score > 0:
                final_score = round(bt_score * 0.4 + vol_score * 0.25 + trend_score * 0.2 + liq_score * 0.15, 2)
            else:
                # Backtest failed — weight more on other factors
                final_score = round(vol_score * 0.4 + trend_score * 0.35 + liq_score * 0.25, 2)

            total_return_pct = sum(t["pnl_pct"] for t in trades) if trades else 0.0
            win_count = sum(1 for t in trades if t["winning"]) if trades else 0

            scored.append({
                **etf,
                "current_price": current_price,
                "amplitude": info.get("amplitude", 0) or 0,
                "volume": info.get("volume", 0) or 0,
                "turnover_rate": info.get("turnover_rate", 0) or 0,
                "score": final_score,
                "volatility_score": vol_score,
                "backtest_score": bt_score,
                "trend_score": trend_score,
                "liquidity_score": liq_score,
                "trade_count": len(trades),
                "total_return_pct": round(total_return_pct, 4),
                "win_rate": round(win_count / len(trades), 3) if trades else 0.0,
                "name_from_api": info.get("name", etf["name"]),
            })

        if not scored:
            return None

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[0]
