"""Trend Following strategy — MA golden-cross / death-cross signals.

When a faster MA crosses ABOVE a slower MA → buy signal (golden cross).
When it crosses BELOW → sell signal (death cross).

Uses the MA columns already present in the DataFrame from Baidu K-line
or locally computed by AKShare fallback.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.signals import LiveSignal, DashboardCard


class TrendFollowingStrategy(BaseStrategy):
    """MA crossover trend-following strategy.

    Vectorized implementation: all cross detections are column operations.
    """

    @property
    def name(self) -> str:
        return "趋势跟随"

    @property
    def description(self) -> str:
        return (
            "均线金叉买入、死叉卖出。短期均线上穿长期均线时全仓买入，"
            "下穿时清仓卖出。适合趋势明显的单边市，震荡市中容易反复止损。"
            "年化预期 8%-15%。"
        )

    def get_default_params(self) -> dict:
        return {
            "fast_ma": "ma5",
            "slow_ma": "ma20",
            "position_pct": 0.8,
            "confirmation_days": 0,
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "fast_ma": {
                "label": "快线",
                "type": "select",
                "options": ["ma5", "ma10", "ma20"],
                "help": "短期均线，上穿慢线时买入",
            },
            "slow_ma": {
                "label": "慢线",
                "type": "select",
                "options": ["ma10", "ma20"],
                "help": "长期均线，快线下穿时卖出",
            },
            "position_pct": {
                "label": "仓位比例",
                "type": "slider",
                "min": 0.1, "max": 1.0, "step": 0.1,
                "help": "每次买入使用可用资金的比例",
            },
            "confirmation_days": {
                "label": "确认天数",
                "type": "number",
                "min": 0, "max": 5, "step": 1,
                "help": "金叉/死叉后等待N天确认再交易（0=当天执行）",
            },
        }

    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}
        fast_col = params["fast_ma"]
        slow_col = params["slow_ma"]
        position_pct = float(params["position_pct"])
        confirm = int(params["confirmation_days"])

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # Validate MA columns exist
        if fast_col not in df.columns:
            raise ValueError(
                f"快线列 '{fast_col}' 不存在。可用列: {list(df.columns)}"
            )
        if slow_col not in df.columns:
            raise ValueError(
                f"慢线列 '{slow_col}' 不存在。可用列: {list(df.columns)}"
            )

        fast = df[fast_col]
        slow = df[slow_col]

        # Vectorized cross detection
        cross_above = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_below = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        # Apply confirmation delay
        if confirm > 0:
            cross_above = cross_above.shift(confirm).fillna(False)
            cross_below = cross_below.shift(confirm).fillna(False)

        # Build signal columns
        df["signal"] = "hold"
        df.loc[cross_above, "signal"] = "buy"
        df.loc[cross_below, "signal"] = "sell"

        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        # Buy signals — deploy position_pct of available cash
        # (Exact shares depend on portfolio state at runtime; engine recalculates)
        for idx in df[df["signal"] == "buy"].index:
            fast_v = fast.loc[idx] if pd.notna(fast.loc[idx]) else 0
            slow_v = slow.loc[idx] if pd.notna(slow.loc[idx]) else 0
            df.at[idx, "signal_reason"] = (
                f"{fast_col.upper()}({fast_v:.3f}) 上穿 "
                f"{slow_col.upper()}({slow_v:.3f})，金叉买入"
            )
            df.at[idx, "signal_shares"] = int(
                df.at[idx, "close"] * 100 / df.at[idx, "close"]
            )  # placeholder — engine computes actual

        for idx in df[df["signal"] == "sell"].index:
            fast_v = fast.loc[idx] if pd.notna(fast.loc[idx]) else 0
            slow_v = slow.loc[idx] if pd.notna(slow.loc[idx]) else 0
            df.at[idx, "signal_reason"] = (
                f"{fast_col.upper()}({fast_v:.3f}) 下穿 "
                f"{slow_col.upper()}({slow_v:.3f})，死叉卖出"
            )
            # Sell all shares — engine clamps to actual holdings
            df.at[idx, "signal_shares"] = 999_999_999

        # Position sizing: buy signals use close as reference price
        # Engine will use signal_price * signal_shares to estimate, then
        # apply position_pct to available cash for actual sizing
        for idx in df[df["signal"] == "buy"].index:
            # Signal placeholder shares — engine uses position_pct of cash
            df.at[idx, "signal_shares"] = int(
                position_pct * 100_000 / df.at[idx, "close"]
            )

        return df

    # ------------------------------------------------------------------
    # Live signal for real-time dashboard
    # ------------------------------------------------------------------

    def get_live_signal(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> LiveSignal:
        """Generate an actionable trading recommendation from the latest bar.

        Compares fast_ma vs slow_ma on the last two bars to detect golden-cross
        (buy) / death-cross (sell) events, or the ongoing trend direction.
        """
        params = {**self.get_default_params(), **kwargs}
        fast_col = params["fast_ma"]
        slow_col = params["slow_ma"]

        # -- Edge case: no data -------------------------------------------------
        if df is None or len(df) == 0:
            price = info.get("current_price") if info else None
            return LiveSignal(
                action="hold",
                current_price=price,
                reason="无历史数据，无法生成均线信号",
                urgency_level="low",
                current_zone="无数据",
            )

        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        last = df.iloc[-1]

        # -- Edge case: missing MA columns -------------------------------------
        if fast_col not in df.columns or slow_col not in df.columns:
            price = info.get("current_price") if info else float(last.get("close", 0))
            return LiveSignal(
                action="hold",
                current_price=price,
                reason=f"缺少均线列 {fast_col} 或 {slow_col}，无法计算交叉信号",
                urgency_level="low",
                current_zone="数据缺失",
            )

        fast_now = last[fast_col]
        slow_now = last[slow_col]

        # Previous bar values for crossover detection
        if len(df) >= 2:
            prev = df.iloc[-2]
            fast_prev = prev[fast_col]
            slow_prev = prev[slow_col]
        else:
            fast_prev = None
            slow_prev = None

        # -- Determine action / zone / urgency ---------------------------------
        if pd.notna(fast_now) and pd.notna(slow_now):
            if fast_now > slow_now:
                if (
                    fast_prev is not None
                    and slow_prev is not None
                    and pd.notna(fast_prev)
                    and pd.notna(slow_prev)
                    and fast_prev <= slow_prev
                ):
                    action = "buy"
                    zone = "金叉买入"
                    urgency = "high"
                else:
                    action = "hold"
                    zone = "多头排列"
                    urgency = "low"
            elif fast_now < slow_now:
                if (
                    fast_prev is not None
                    and slow_prev is not None
                    and pd.notna(fast_prev)
                    and pd.notna(slow_prev)
                    and fast_prev >= slow_prev
                ):
                    action = "sell"
                    zone = "死叉卖出"
                    urgency = "high"
                else:
                    action = "hold"
                    zone = "空头排列"
                    urgency = "low"
            else:
                action = "hold"
                zone = "均线粘合"
                urgency = "low"
        else:
            action = "hold"
            zone = "数据缺失"
            urgency = "low"

        # -- Build descriptions -------------------------------------------------
        trigger_description = (
            f"{fast_col.upper()}={fast_now:.3f} vs "
            f"{slow_col.upper()}={slow_now:.3f}"
        )

        if zone == "金叉买入":
            reason = (
                f"{fast_col.upper()}上穿{slow_col.upper()}，"
                f"短期趋势转强，建议买入"
            )
        elif zone == "死叉卖出":
            reason = (
                f"{fast_col.upper()}下穿{slow_col.upper()}，"
                f"短期趋势转弱，建议卖出"
            )
        elif zone == "多头排列":
            reason = (
                f"{fast_col.upper()}持续高于{slow_col.upper()}，"
                f"多头趋势延续，宜持仓不动"
            )
        elif zone == "空头排列":
            reason = (
                f"{fast_col.upper()}持续低于{slow_col.upper()}，"
                f"空头趋势延续，宜观望等待"
            )
        else:
            reason = f"{fast_col.upper()}与{slow_col.upper()}接近，方向不明"

        current_price = (
            info.get("current_price")
            if info and info.get("current_price")
            else float(last.get("close", 0))
        )

        return LiveSignal(
            action=action,
            current_price=current_price,
            trigger_description=trigger_description,
            reason=reason,
            urgency_level=urgency,
            current_zone=zone,
        )

    # ------------------------------------------------------------------
    # Dashboard cards for UI rendering
    # ------------------------------------------------------------------

    def get_dashboard_cards(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> list[DashboardCard]:
        """Return three info cards summarising the current MA state.

        Cards:
        1. 均线状态 — metric: fast/slow values + direction arrow
        2. 趋势区间 — info: zone name + strength
        3. 均线间距 — metric: gap percentage
        """
        params = {**self.get_default_params(), **kwargs}
        fast_col = params["fast_ma"]
        slow_col = params["slow_ma"]

        # -- Edge case: no data -------------------------------------------------
        if df is None or len(df) == 0:
            return [
                DashboardCard(
                    card_id=f"{self.name}_ma_status",
                    title="均线状态",
                    card_type="metric",
                    content={"value": "无数据", "trend": "—"},
                    priority=1,
                ),
                DashboardCard(
                    card_id=f"{self.name}_trend_zone",
                    title="趋势区间",
                    card_type="info",
                    content={"zone": "无数据", "strength": "—"},
                    priority=2,
                ),
                DashboardCard(
                    card_id=f"{self.name}_ma_gap",
                    title="均线间距",
                    card_type="metric",
                    content={"value": "—", "gap_pct": None},
                    priority=2,
                ),
            ]

        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        last = df.iloc[-1]

        fast_val = last.get(fast_col) if fast_col in df.columns else None
        slow_val = last.get(slow_col) if slow_col in df.columns else None

        # -- Edge case: missing MA columns -------------------------------------
        if not pd.notna(fast_val) or not pd.notna(slow_val):
            return [
                DashboardCard(
                    card_id=f"{self.name}_ma_status",
                    title="均线状态",
                    card_type="metric",
                    content={"value": "数据缺失", "trend": "—"},
                    priority=1,
                ),
                DashboardCard(
                    card_id=f"{self.name}_trend_zone",
                    title="趋势区间",
                    card_type="info",
                    content={"zone": "数据不足", "strength": "—"},
                    priority=2,
                ),
                DashboardCard(
                    card_id=f"{self.name}_ma_gap",
                    title="均线间距",
                    card_type="metric",
                    content={"value": "—", "gap_pct": None},
                    priority=2,
                ),
            ]

        # -- Determine trend state -----------------------------------------------
        if fast_val > slow_val:
            direction = "↑"
            trend = "多头"
            zone_name = "多头排列"
            strength = "偏多"
        elif fast_val < slow_val:
            direction = "↓"
            trend = "空头"
            zone_name = "空头排列"
            strength = "偏空"
        else:
            direction = "→"
            trend = "粘合"
            zone_name = "均线粘合"
            strength = "中性"

        gap_pct = (
            (fast_val - slow_val) / slow_val * 100 if slow_val != 0 else 0.0
        )

        card_ma_status = DashboardCard(
            card_id=f"{self.name}_ma_status",
            title="均线状态",
            card_type="metric",
            content={
                "fast_label": fast_col.upper(),
                "fast_value": round(float(fast_val), 3),
                "slow_label": slow_col.upper(),
                "slow_value": round(float(slow_val), 3),
                "direction": direction,
                "trend": trend,
            },
            priority=1,
        )

        card_trend_zone = DashboardCard(
            card_id=f"{self.name}_trend_zone",
            title="趋势区间",
            card_type="info",
            content={
                "zone": zone_name,
                "strength": strength,
                "fast_col": fast_col.upper(),
                "slow_col": slow_col.upper(),
                "fast_val": round(float(fast_val), 3),
                "slow_val": round(float(slow_val), 3),
            },
            priority=2,
        )

        card_ma_gap = DashboardCard(
            card_id=f"{self.name}_ma_gap",
            title="均线间距",
            card_type="metric",
            content={
                "value": f"{gap_pct:+.2f}%",
                "gap_pct": round(float(gap_pct), 2),
                "description": (
                    f"{fast_col.upper()} 相对 {slow_col.upper()} 的偏离幅度"
                ),
            },
            priority=2,
        )

        return [card_ma_status, card_trend_zone, card_ma_gap]
