"""Trend Following strategy — MA golden-cross / death-cross signals.

When a faster MA crosses ABOVE a slower MA → buy signal (golden cross).
When it crosses BELOW → sell signal (death cross).

Uses the MA columns already present in the DataFrame from Baidu K-line
or locally computed by AKShare fallback.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy


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
