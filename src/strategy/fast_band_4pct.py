"""4%快速波段 — 宽基ETF超跌反弹，PE+技术面双维度选基。

核心理念：在低估宽基中找超跌反弹机会。
- 候选池：25只核心宽基ETF（与4%定投法同池）
- 选基：PE低估(25%) + 入场时机(35%) + 波动性(25%) + 流动性(15%) → Top 5
- 入场：近3日跌幅(4) + K线反转(2) + RSI超卖(2) + 均线支撑(2) = 0-10分，≥5买入
- 出场：+2.5%止盈 / -2%止损 / 3天时间止盈
- 扫描：批量API + 并行计算，秒级出Top 5
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.signals import DashboardCard, LiveSignal


# =============================================================================
# Candidate pool — 25 broad-market ETFs (same as 4% DCA pool)
# =============================================================================

CANDIDATE_ETFS: list[dict] = [
    # ── 大盘蓝筹 ──
    {"code": "510300", "name": "沪深300ETF",       "approx_price": 3.9,  "category": "broad"},
    {"code": "510050", "name": "上证50ETF",         "approx_price": 2.7,  "category": "broad"},
    {"code": "510180", "name": "上证180ETF",        "approx_price": 3.5,  "category": "broad"},
    {"code": "159901", "name": "深证100ETF",        "approx_price": 3.8,  "category": "broad"},
    # ── 中盘 ──
    {"code": "510500", "name": "中证500ETF",         "approx_price": 5.8,  "category": "broad"},
    {"code": "515800", "name": "中证800ETF",         "approx_price": 2.5,  "category": "broad"},
    # ── 小盘 ──
    {"code": "512100", "name": "中证1000ETF",        "approx_price": 2.3,  "category": "broad"},
    {"code": "563300", "name": "中证2000ETF",        "approx_price": 0.9,  "category": "broad"},
    {"code": "159628", "name": "国证2000ETF",        "approx_price": 1.0,  "category": "broad"},
    # ── 新宽基 ──
    {"code": "159593", "name": "中证A50ETF",         "approx_price": 1.0,  "category": "broad"},
    {"code": "159338", "name": "中证A500ETF",        "approx_price": 0.9,  "category": "broad"},
    {"code": "560050", "name": "MSCI A50ETF",        "approx_price": 0.8,  "category": "broad"},
    # ── 创业板 ──
    {"code": "159915", "name": "创业板ETF",          "approx_price": 2.2,  "category": "broad"},
    {"code": "159949", "name": "创业板50ETF",        "approx_price": 0.9,  "category": "broad"},
    # ── 科创板 ──
    {"code": "588000", "name": "科创50ETF",          "approx_price": 0.9,  "category": "broad"},
    {"code": "588190", "name": "科创100ETF",         "approx_price": 0.8,  "category": "broad"},
    # ── 双创 ──
    {"code": "159781", "name": "双创50ETF",          "approx_price": 1.1,  "category": "broad"},
    # ── 策略宽基（红利/低波/质量）──
    {"code": "510880", "name": "中证红利ETF",        "approx_price": 3.0,  "category": "broad50"},
    {"code": "515080", "name": "中证红利ETF(招商)",  "approx_price": 1.5,  "category": "broad50"},
    {"code": "512890", "name": "红利低波ETF",        "approx_price": 1.5,  "category": "broad50"},
    {"code": "515180", "name": "红利低波100ETF",     "approx_price": 1.3,  "category": "broad50"},
    {"code": "563020", "name": "红利低波ETF(易方达)","approx_price": 1.0,  "category": "broad50"},
    {"code": "515450", "name": "红利质量ETF",        "approx_price": 1.2,  "category": "broad50"},
    {"code": "159905", "name": "深红利ETF",          "approx_price": 2.0,  "category": "broad50"},
    {"code": "562060", "name": "中证A50增强ETF",     "approx_price": 1.0,  "category": "broad50"},
]


# =============================================================================
# K-line reversal detection
# =============================================================================

def _detect_reversal(row: pd.Series, prev_row: pd.Series) -> tuple[int, str]:
    """Detect bullish reversal candlestick patterns. Returns (score 0-2, label)."""
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    po, pc = float(prev_row["open"]), float(prev_row["close"])
    ph, pl = float(prev_row["high"]), float(prev_row["low"])

    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l
    total_range = h - l if h > l else 0.001

    # 锤子线
    if body > 0 and lower_shadow >= 2 * body and upper_shadow <= body * 0.5:
        return 2, "锤子线（长下影反转）"
    # 看涨吞没
    if c > o and pc < po and o <= pc and c >= po:
        return 2, "看涨吞没（阳包阴）"
    # 启明星
    prev_body = abs(pc - po)
    if prev_body > 0 and body < prev_body * 0.5 and pc < po and c > o:
        if o <= pc and c > (pc + po) / 2:
            return 2, "启明星（晨星反转）"
    # 十字星
    if body < total_range * 0.1 and lower_shadow > body * 2:
        return 1, "十字星（底部企稳）"
    # 长下影线
    if lower_shadow > body * 1.5 and c > o:
        return 1, "长下影线（买盘支撑）"

    return 0, "无反转形态"


# =============================================================================
# RSI helper
# =============================================================================

def _compute_rsi_val(closes: np.ndarray, period: int, idx: int) -> float | None:
    """Compute simple (non-Wilder) RSI for a single index."""
    if idx < period:
        return None
    window = closes[idx - period:idx + 1]
    deltas = np.diff(window)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


# =============================================================================
# Strategy class
# =============================================================================

class FastBand4PctStrategy(BaseStrategy):
    """4%快速波段 — PE+技术面双维度选基，捕捉宽基超跌反弹。

    选基（25只→Top 5）：
    - PE低估 25%：PE分位越低越好（<30%满分）
    - 入场时机 35%：跌幅+K线+RSI+支撑综合评分
    - 波动性 25%：振幅越大反弹空间越大
    - 流动性 15%：换手率越高越容易进出

    入场（0-10分，≥5买入）：
    - 近3日跌幅 0-4分 | K线反转 0-2分 | RSI超卖 0-2分 | 均线支撑 0-2分

    出场：
    - +2.5%止盈 | -2%止损 | 3天时间止盈
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "4%快速波段"

    @property
    def description(self) -> str:
        return (
            "宽基ETF超跌反弹捕手：PE估值+技术面双维度打分，从25只宽基中精选Top 5。\n\n"
            "**选基**：PE低估(25%) + 入场时机(35%) + 波动性(25%) + 流动性(15%)。\n"
            "**入场**：近3日跌幅(4分) + K线反转(2分) + RSI超卖(2分) + 均线支撑(2分)，≥5分买入。\n"
            "**出场**：+2.5%止盈 | -2%止损 | 3天时间止盈。\n\n"
            "买在低估+超跌时，赚取反弹快钱。"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return {
            "entry_threshold": 5,
            "take_profit_pct": 0.025,
            "stop_loss_pct": 0.02,
            "max_hold_days": 3,
            "decline_lookback": 3,
            "rsi_period": 14,
            "position_pct": 0.5,
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "entry_threshold": {
                "label": "入场门槛", "type": "slider",
                "min": 3, "max": 8, "step": 1,
                "help": "综合评分≥此值买入（默认5，越高越保守）",
            },
            "take_profit_pct": {
                "label": "止盈线", "type": "slider",
                "min": 0.015, "max": 0.05, "step": 0.005,
                "help": "盈利达此比例卖出（默认2.5%）",
            },
            "stop_loss_pct": {
                "label": "止损线", "type": "slider",
                "min": 0.01, "max": 0.04, "step": 0.005,
                "help": "亏损达此比例止损（默认2%）",
            },
            "max_hold_days": {
                "label": "最长持有天数", "type": "number",
                "min": 2, "max": 7, "step": 1,
                "help": "持有超此天数强制平仓（默认3天）",
            },
            "decline_lookback": {
                "label": "跌幅回看天数", "type": "number",
                "min": 2, "max": 5, "step": 1,
                "help": "计算近期跌幅的天数（默认3天）",
            },
            "rsi_period": {
                "label": "RSI周期", "type": "number",
                "min": 7, "max": 21, "step": 1,
                "help": "RSI计算周期（默认14）",
            },
            "position_pct": {
                "label": "仓位比例", "type": "slider",
                "min": 0.3, "max": 1.0, "step": 0.1,
                "help": "可用资金使用比例（默认50%，留弹药）",
            },
        }

    # ------------------------------------------------------------------
    # Entry scoring (0-10)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entry_score(
        df_sorted: pd.DataFrame, idx: int, params: dict,
    ) -> dict:
        """Score today's entry quality (0-10).

        Returns dict with total_score and factor breakdown + details list.
        """
        row = df_sorted.iloc[idx]
        close = float(row["close"])
        decline_days = int(params.get("decline_lookback", 3))
        rsi_period = int(params.get("rsi_period", 14))
        details: list[str] = []

        # ── Factor 1: Recent decline (0-4) ──
        decline_score = 0.0
        if idx >= decline_days:
            prev_close = float(df_sorted.iloc[idx - decline_days]["close"])
            if prev_close > 0:
                decline_pct = (prev_close - close) / prev_close  # positive = down, negative = up
                if decline_pct >= 0.05:
                    decline_score = 4.0
                elif decline_pct >= 0.04:
                    decline_score = 3.5
                elif decline_pct >= 0.03:
                    decline_score = 3.0
                elif decline_pct >= 0.02:
                    decline_score = 2.0
                elif decline_pct >= 0.01:
                    decline_score = 1.0
                # Format: show direction clearly
                if decline_pct > 0:
                    details.append(f"近{decline_days}日跌{decline_pct:.1%} → +{decline_score:.1f}")
                elif decline_pct < -0.005:
                    details.append(f"近{decline_days}日涨{abs(decline_pct):.1%}（逆势上涨，不加分） → +0")
                else:
                    details.append(f"近{decline_days}日持平 → +0")
        if not details:
            details.append("近N日跌幅 数据不足 → +0")

        # ── Factor 2: K-line reversal (0-2) ──
        reversal_score = 0
        reversal_label = "无反转形态"
        if idx >= 1:
            reversal_score, reversal_label = _detect_reversal(row, df_sorted.iloc[idx - 1])
        details.append(f"K线: {reversal_label} → +{reversal_score}")

        # ── Factor 3: RSI oversold (0-2) ──
        rsi_score = 0.0
        rsi_val = None
        if idx >= rsi_period:
            closes_arr = df_sorted["close"].values[:idx + 1].astype(float)
            rsi_val = _compute_rsi_val(closes_arr, rsi_period, idx)
            if rsi_val is not None:
                if rsi_val < 25:
                    rsi_score = 2.0
                elif rsi_val < 30:
                    rsi_score = 1.5
                elif rsi_val < 35:
                    rsi_score = 1.0
                elif rsi_val < 40:
                    rsi_score = 0.5
                details.append(f"RSI({rsi_period})={rsi_val:.1f} → +{rsi_score:.1f}")
        if rsi_val is None:
            details.append("RSI数据不足 → +0")

        # ── Factor 4: MA20 support proximity (0-2) ──
        support_score = 0.0
        if "_ma20" in df_sorted.columns:
            ma20 = df_sorted.at[idx, "_ma20"]
            if pd.notna(ma20) and ma20 > 0:
                dist = (close - ma20) / ma20
                if abs(dist) <= 0.02:
                    support_score = 2.0
                    details.append(f"紧贴MA20（偏离{dist:+.1%}）→ +2")
                elif abs(dist) <= 0.03:
                    support_score = 1.0
                    details.append(f"靠近MA20（偏离{dist:+.1%}）→ +1")
                else:
                    details.append(f"距MA20 {dist:+.1%} → +0")

        total = round(decline_score + reversal_score + rsi_score + support_score, 1)
        total = max(0.0, min(10.0, total))

        # Count consecutive down days
        consec = 0
        for j in range(idx, max(idx - 10, -1), -1):
            if j >= 1:
                if float(df_sorted.iloc[j - 1]["close"]) < float(df_sorted.iloc[j - 1]["open"]):
                    consec += 1
                else:
                    break

        return {
            "total_score": total,
            "decline": decline_score,
            "reversal": reversal_score,
            "reversal_label": reversal_label,
            "rsi": rsi_score,
            "rsi_value": rsi_val,
            "support": support_score,
            "consecutive_down": consec,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Signal generation (backtest)
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}
        threshold = int(params["entry_threshold"])
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        position_pct = float(params["position_pct"])
        backtest_capital = float(kwargs.get("backtest_capital", 100_000))

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()
        if "close" not in df.columns:
            df["signal"] = "hold"; df["signal_price"] = 0.0
            df["signal_shares"] = 0; df["signal_reason"] = ""
            return df

        df["_ma20"] = df["close"].rolling(window=20, min_periods=20).mean()

        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        in_position = False
        entry_price = 0.0
        entry_idx = -1
        entry_shares = 0
        budget = position_pct * backtest_capital

        for i in range(len(df)):
            close = float(df.at[i, "close"])

            if in_position:
                pnl_pct = (close - entry_price) / entry_price if entry_price > 0 else 0
                hold_days = i - entry_idx
                exit_reason = ""

                if pnl_pct <= -stop_loss:
                    exit_reason = f"🛑 止损：{pnl_pct:.1%}（≤-{stop_loss:.0%}）"
                elif pnl_pct >= take_profit:
                    exit_reason = f"🎯 止盈：{pnl_pct:.1%}（≥{take_profit:.0%}）"
                elif hold_days >= max_hold:
                    exit_reason = f"⏰ 时间到：持有{hold_days}天，盈亏{pnl_pct:+.1%}"

                if exit_reason:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = entry_shares
                    df.at[i, "signal_reason"] = exit_reason
                    in_position = False

            else:
                if i < max(int(params["decline_lookback"]), int(params["rsi_period"])) + 1:
                    continue

                score_result = self._compute_entry_score(df, i, params)
                if score_result["total_score"] < threshold:
                    continue

                shares = max(100, int(budget / close) // 100 * 100)
                if shares < 100:
                    continue

                df.at[i, "signal"] = "buy"
                df.at[i, "signal_price"] = close
                df.at[i, "signal_shares"] = shares
                reasons = " | ".join(score_result["details"][:3])
                df.at[i, "signal_reason"] = (
                    f"🎯 入场{score_result['total_score']:.0f}/10 | {reasons}"
                )
                in_position = True
                entry_price = close
                entry_idx = i
                entry_shares = shares

        return df.drop(columns=["_ma20"], errors="ignore")

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    def get_live_signal(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> LiveSignal:
        if df is None or df.empty:
            return LiveSignal(action="hold", reason="无历史数据", urgency_level="low")

        params = {**self.get_default_params(), **kwargs}
        threshold = int(params["entry_threshold"])
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        position_pct = float(params["position_pct"])
        decline_days = int(params["decline_lookback"])
        rsi_period = int(params["rsi_period"])

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
        if "_ma20" not in df_sorted.columns:
            df_sorted["_ma20"] = df_sorted["close"].rolling(window=20, min_periods=20).mean()

        current_price = info.get("current_price") if info else None
        if current_price is None:
            current_price = float(df_sorted.iloc[-1]["close"])

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
            except Exception:
                pass

        # ── In position: check exits ──
        if has_position and holding_cost and holding_cost > 0:
            pnl_pct = (current_price - holding_cost) / holding_cost

            if pnl_pct <= -stop_loss:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"亏损 {pnl_pct:.1%}，触发止损",
                    reason=f"🛑 止损！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}，全部卖出。",
                    urgency_level="high", current_zone="止损区",
                )
            if pnl_pct >= take_profit:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"盈利 {pnl_pct:.1%}，触发止盈",
                    reason=f"🎯 止盈！入场 ¥{holding_cost:.3f} → 现价 ¥{current_price:.3f}（+{pnl_pct:.1%}），落袋为安。",
                    urgency_level="high", current_zone="止盈区",
                )
            if hold_days >= max_hold:
                return LiveSignal(
                    action="sell", current_price=round(current_price, 4),
                    suggested_shares=holding_shares,
                    trigger_description=f"持有 {hold_days} 天，触发时间止盈",
                    reason=f"⏰ 时间到！持有{hold_days}天，盈亏{pnl_pct:+.1%}，建议卖出。",
                    urgency_level="medium", current_zone="时间止盈区",
                )

            tp = round(holding_cost * (1 + take_profit), 4)
            return LiveSignal(
                action="hold", current_price=round(current_price, 4),
                trigger_description=f"持仓中 盈亏{pnl_pct:+.1%} 第{hold_days}天",
                next_trigger_price=tp,
                reason=f"📌 持仓：入场 ¥{holding_cost:.3f}，现价 ¥{current_price:.3f}（{pnl_pct:+.1%}），止盈 ¥{tp}",
                urgency_level="low", portions_used=hold_days, portions_total=max_hold,
                current_zone=f"持仓第{hold_days}天",
            )

        # ── Not in position: entry check ──
        min_bars = max(decline_days, rsi_period) + 2
        if len(df_sorted) < min_bars:
            return LiveSignal(action="hold", reason=f"数据不足（需≥{min_bars}根K线）", urgency_level="low")

        score_result = self._compute_entry_score(df_sorted, len(df_sorted) - 1, params)
        score = score_result["total_score"]

        if score < threshold:
            need = threshold - score
            return LiveSignal(
                action="wait_for_drop",
                current_price=round(current_price, 4),
                trigger_description=f"评分 {score:.0f}/10，还差{need:.0f}分",
                reason=f"⏳ {score:.0f}/10（需≥{threshold}）。" + " | ".join(score_result["details"]),
                urgency_level="low", current_zone="等待回调",
            )

        available_cash = 2000.0
        if pf and pf.get("available_cash", 0) > 0:
            available_cash = float(pf["available_cash"])
        budget = available_cash * position_pct
        raw_shares = int(budget / current_price) if current_price > 0 else 0
        shares = max(100, (raw_shares // 100) * 100)

        return LiveSignal(
            action="buy",
            current_price=round(current_price, 4),
            suggested_shares=shares,
            suggested_amount=round(shares * current_price, 2),
            trigger_description=f"入场评分 {score:.0f}/10",
            reason=(
                f"🎯 买入信号！综合 {score:.0f}/10。\n"
                + "\n".join(score_result["details"])
                + f"\n\n建议买 {shares} 股（≈¥{shares*current_price:.0f}），"
                + f"止盈 +{take_profit:.0%}，止损 -{stop_loss:.0%}。"
            ),
            urgency_level="high", current_zone="买入区",
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
        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
        if "_ma20" not in df_sorted.columns:
            df_sorted["_ma20"] = df_sorted["close"].rolling(window=20, min_periods=20).mean()

        current_price = info.get("current_price") if info else None
        if current_price is None and len(df_sorted) > 0:
            current_price = float(df_sorted.iloc[-1]["close"])

        cards: list[DashboardCard] = []

        # Card 1: Entry score
        try:
            score_result = self._compute_entry_score(df_sorted, len(df_sorted) - 1, params)
        except Exception:
            score_result = None

        if score_result:
            threshold = int(params["entry_threshold"])
            factors = [
                {"label": "近3日跌幅", "score": int(score_result["decline"]), "max": 4},
                {"label": "K线反转", "score": int(score_result["reversal"]), "max": 2},
                {"label": "RSI超卖", "score": int(score_result["rsi"]), "max": 2},
                {"label": "均线支撑", "score": int(score_result["support"]), "max": 2},
            ]
            cards.append(DashboardCard(
                card_id="entry_score",
                title=f"入场评分 · {score_result['total_score']:.0f}/10（需≥{threshold}）",
                card_type="progress",
                content={
                    "value_pct": score_result["total_score"] * 10,
                    "max_value": 100, "threshold": threshold * 10,
                    "factors": factors, "details": score_result["details"],
                    "ready": score_result["total_score"] >= threshold,
                },
                priority=1,
            ))

        # Card 2: Exit rules
        cards.append(DashboardCard(
            card_id="exit_rules",
            title="出场规则",
            card_type="info",
            content={
                "rules": [
                    {"label": "止盈", "value": f"+{float(params['take_profit_pct']):.1%}"},
                    {"label": "止损", "value": f"-{float(params['stop_loss_pct']):.1%}"},
                    {"label": "时间止盈", "value": f"持有{int(params['max_hold_days'])}天"},
                ],
            },
            priority=1,
        ))

        # Card 3: Market context
        rsi_val = None
        rsi_period = int(params.get("rsi_period", 14))
        if len(df_sorted) >= rsi_period + 1:
            rsi_val = _compute_rsi_val(
                df_sorted["close"].values.astype(float), rsi_period, len(df_sorted) - 1,
            )

        consec = 0
        for j in range(len(df_sorted) - 1, max(len(df_sorted) - 11, -1), -1):
            if j >= 1:
                if float(df_sorted.iloc[j - 1]["close"]) < float(df_sorted.iloc[j - 1]["open"]):
                    consec += 1
                else:
                    break

        cards.append(DashboardCard(
            card_id="market_context",
            title="市场环境",
            card_type="info",
            content={
                "current_price": round(current_price, 4) if current_price else None,
                "rsi": round(rsi_val, 1) if rsi_val is not None else None,
                "consecutive_down": consec,
            },
            priority=2,
        ))

        return cards

    # =====================================================================
    # ETF selection — 25宽基 → Top 5, PE+技术面双维度
    # =====================================================================

    @staticmethod
    def get_candidate_pool() -> list[dict]:
        return list(CANDIDATE_ETFS)

    @staticmethod
    def select_top_etfs(
        n: int = 5,
        candidates: list[dict] | None = None,
        max_price: float = 20.0,
    ) -> list[dict]:
        """Score & rank 25 broad-market ETFs, return top N.

        Composite score (0-100):
        - PE低估 25%：PE分位 <30% → 满分25，30-50% → 20，50-70% → 12，>70% → 5
        - 入场时机 35%：当前 entry score / 10 * 35
        - 波动性 25%：振幅越大分越高（反弹空间大）
        - 流动性 15%：换手率越高越好

        Uses batch API + parallel history fetch.  ~2 seconds for 25 ETFs.
        """
        from src.data.fetcher import fetch_etf_hist, fetch_multi_etf_info
        from src.data.pe_history import get_etf_pe_percentile

        pool = candidates if candidates is not None else list(CANDIDATE_ETFS)
        codes = [e["code"] for e in pool]

        # ── Batch real-time quotes (1 API call) ──
        try:
            all_info = fetch_multi_etf_info(codes)
        except Exception:
            all_info = {}

        # ── Parallel history fetch ──
        hist_cache: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_map = {executor.submit(fetch_etf_hist, c): c for c in codes}
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    hist_cache[code] = future.result()
                except Exception:
                    hist_cache[code] = None

        params = FastBand4PctStrategy().get_default_params()
        scored: list[dict] = []

        for etf in pool:
            code = etf["code"]
            info = all_info.get(code)
            if info is None:
                continue

            cp = info.get("current_price")
            if cp is None or cp <= 0 or cp > max_price:
                continue

            # ═══ 1. PE估值分 (0-25) ═══
            pe_score = 0.0
            pe_pct = None
            pe_val = info.get("pe_ttm") or info.get("pe_static")
            try:
                pp = get_etf_pe_percentile(code, current_pe=pe_val)
                if pp is not None and pp.pe_percentile is not None:
                    pe_pct = pp.pe_percentile
            except Exception:
                pass

            if pe_pct is not None:
                if pe_pct < 10:
                    pe_score = 25.0
                elif pe_pct < 30:
                    pe_score = 22.0
                elif pe_pct < 50:
                    pe_score = 17.0
                elif pe_pct < 70:
                    pe_score = 10.0
                else:
                    pe_score = 5.0
            else:
                pe_score = 10.0  # no PE data, neutral

            # ═══ 2. 入场时机分 (0-35) ═══
            entry_score = 0.0
            entry_details: list[str] = []
            hist = hist_cache.get(code)
            if hist is not None and not hist.empty and len(hist) >= 16:
                h = hist.sort_values("date", ascending=True).reset_index(drop=True)
                h["_ma20"] = h["close"].rolling(window=20, min_periods=20).mean()
                try:
                    result = FastBand4PctStrategy._compute_entry_score(h, len(h) - 1, params)
                    entry_score = result["total_score"] / 10.0 * 35.0  # scale to 0-35
                    entry_details = result["details"]
                except Exception:
                    pass

            # ═══ 3. 波动性分 (0-25) ═══
            amp = info.get("amplitude", 0) or 0
            vol_score = min(amp * 5, 25.0)  # 5% amp → 25, linear

            # ═══ 4. 流动性分 (0-15) ═══
            turnover = info.get("turnover_rate", 0) or 0
            liq_score = min(turnover * 3, 15.0)  # 5% turnover → 15

            total = round(pe_score + entry_score + vol_score + liq_score, 1)

            # PE status badge
            if pe_pct is not None:
                if pe_pct < 30:
                    pe_badge = f"🟢 PE分位{pe_pct:.0f}% 低估"
                elif pe_pct < 70:
                    pe_badge = f"🟡 PE分位{pe_pct:.0f}% 合理"
                else:
                    pe_badge = f"🔴 PE分位{pe_pct:.0f}% 高估"
            else:
                pe_badge = "⚪ PE无数据"

            scored.append({
                **etf,
                "current_price": cp,
                "amplitude": amp,
                "turnover_rate": turnover,
                "score": total,
                "pe_score": round(pe_score, 1),
                "entry_score_raw": round(entry_score / 35.0 * 10, 1),  # back to 0-10
                "volatility_score": round(vol_score, 1),
                "liquidity_score": round(liq_score, 1),
                "pe_percentile": pe_pct,
                "pe_badge": pe_badge,
                "name_from_api": info.get("name", etf["name"]),
                "score_details": entry_details[:3],
            })

        if not scored:
            return []

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n]
