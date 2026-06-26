"""Hybrid strategy — DCA base position + grid trading overlay.

Splits capital into two pools:
  - DCA pool (default 60%): runs Value Averaging (PE-threshold periodic buys)
  - Grid pool (default 40%): runs Grid Trading (buy-low sell-high)

The two sub-strategies operate independently on their allocated capital.
Signals from both are merged: on the same bar, buy signals from both
pools are combined; sell signals are independent.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.value_averaging import ValueAveragingStrategy
from src.strategy.grid_trading import GridTradingStrategy


class HybridStrategy(BaseStrategy):
    """DCA + Grid hybrid — two-pool allocation."""

    @property
    def name(self) -> str:
        return "网格+定投"

    @property
    def description(self) -> str:
        return (
            "定投打底仓 + 网格做波段。大部分资金用于PE估值定投建立长期仓位，"
            "小部分资金在价格区间内做网格交易增厚收益。"
            "兼顾长期持有和短期波动，年化预期 8%-12%。"
        )

    def get_default_params(self) -> dict:
        return {
            "dca_allocation_pct": 0.6,
            # DCA params
            "dca_base_amount": 1000,
            "pe_low": 15.0,
            "pe_mid": 20.0,
            "pe_high": 30.0,
            "pe_max": 40.0,
            "dca_frequency": "monthly",
            # Grid params
            "grid_count": 10,
            "upper_padding_pct": 0.05,
            "lower_padding_pct": 0.05,
            "position_per_grid_pct": 0.08,
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "dca_allocation_pct": {
                "label": "定投占比",
                "type": "slider",
                "min": 0.3, "max": 0.8, "step": 0.05,
                "help": "总资产中用于定投的比例，剩余用于网格",
            },
            "dca_base_amount": {
                "label": "定投基准金额",
                "type": "number",
                "min": 100, "max": 50000, "step": 100,
                "help": "每期定投基准金额",
            },
            "pe_low": {
                "label": "PE低估值线",
                "type": "number",
                "min": 5.0, "max": 25.0, "step": 1.0,
                "help": "PE低于此 → 2倍定投",
            },
            "pe_mid": {
                "label": "PE中估值线",
                "type": "number",
                "min": 10.0, "max": 35.0, "step": 1.0,
                "help": "PE介于低-中 → 1.5倍",
            },
            "pe_high": {
                "label": "PE高估值线",
                "type": "number",
                "min": 15.0, "max": 45.0, "step": 1.0,
                "help": "PE介于中-高 → 1.0倍",
            },
            "pe_max": {
                "label": "PE停止线",
                "type": "number",
                "min": 20.0, "max": 60.0, "step": 1.0,
                "help": "PE高于此 → 暂停定投",
            },
            "dca_frequency": {
                "label": "定投频率",
                "type": "select",
                "options": ["weekly", "monthly"],
                "help": "定投执行频率",
            },
            "grid_count": {
                "label": "网格档数",
                "type": "number",
                "min": 3, "max": 30, "step": 1,
                "help": "网格交易档数",
            },
            "upper_padding_pct": {
                "label": "网格上延",
                "type": "slider",
                "min": 0.0, "max": 0.20, "step": 0.01,
                "help": "在历史最高价之上额外延伸的比例",
            },
            "lower_padding_pct": {
                "label": "网格下延",
                "type": "slider",
                "min": 0.0, "max": 0.20, "step": 0.01,
                "help": "在历史最低价之下额外延伸的比例",
            },
            "position_per_grid_pct": {
                "label": "每档仓位",
                "type": "slider",
                "min": 0.02, "max": 0.15, "step": 0.01,
                "help": "每档网格占总资产比例",
            },
        }

    def generate_signals(
        self, df: pd.DataFrame, pe_value: float | None = None, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # --- Run DCA sub-strategy ---
        dca = ValueAveragingStrategy()
        dca_params = {
            "base_amount": params["dca_base_amount"],
            "pe_low": params["pe_low"],
            "pe_mid": params["pe_mid"],
            "pe_high": params["pe_high"],
            "pe_max": params["pe_max"],
            "frequency": params["dca_frequency"],
        }
        dca_df = dca.generate_signals(df, pe_value=pe_value, **dca_params)

        # --- Run Grid sub-strategy ---
        grid = GridTradingStrategy()
        grid_params = {
            "grid_count": params["grid_count"],
            "upper_padding_pct": params["upper_padding_pct"],
            "lower_padding_pct": params["lower_padding_pct"],
            "position_per_grid_pct": params["position_per_grid_pct"],
        }
        grid_df = grid.generate_signals(df, **grid_params)

        # --- Merge signals ---
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        dca_alloc = float(params["dca_allocation_pct"])

        for idx in df.index:
            dca_sig = dca_df.at[idx, "signal"]
            grid_sig = grid_df.at[idx, "signal"]
            price = df.at[idx, "close"]

            # Both buy → combine
            if dca_sig == "buy" and grid_sig == "buy":
                df.at[idx, "signal"] = "buy"
                df.at[idx, "signal_price"] = price
                dca_shares = int(dca_df.at[idx, "signal_shares"])
                grid_shares = int(grid_df.at[idx, "signal_shares"])
                df.at[idx, "signal_shares"] = dca_shares + grid_shares
                df.at[idx, "signal_reason"] = (
                    f"定投+网格同时买入: "
                    f"{dca_df.at[idx, 'signal_reason']} | "
                    f"{grid_df.at[idx, 'signal_reason']}"
                )

            # DCA buy only
            elif dca_sig == "buy":
                df.at[idx, "signal"] = "buy"
                df.at[idx, "signal_price"] = price
                df.at[idx, "signal_shares"] = dca_df.at[idx, "signal_shares"]
                df.at[idx, "signal_reason"] = (
                    f"[定投] {dca_df.at[idx, 'signal_reason']}"
                )

            # Grid buy only
            elif grid_sig == "buy":
                df.at[idx, "signal"] = "buy"
                df.at[idx, "signal_price"] = price
                df.at[idx, "signal_shares"] = grid_df.at[idx, "signal_shares"]
                df.at[idx, "signal_reason"] = (
                    f"[网格] {grid_df.at[idx, 'signal_reason']}"
                )

            # Sell — grid dominates (DCA rarely sells)
            elif grid_sig == "sell":
                df.at[idx, "signal"] = "sell"
                df.at[idx, "signal_price"] = price
                df.at[idx, "signal_shares"] = grid_df.at[idx, "signal_shares"]
                df.at[idx, "signal_reason"] = (
                    f"[网格] {grid_df.at[idx, 'signal_reason']}"
                )

            elif dca_sig == "sell":
                df.at[idx, "signal"] = "sell"
                df.at[idx, "signal_price"] = price
                df.at[idx, "signal_shares"] = dca_df.at[idx, "signal_shares"]
                df.at[idx, "signal_reason"] = (
                    f"[定投] {dca_df.at[idx, 'signal_reason']}"
                )

        # Store allocation metadata
        df.attrs["dca_allocation_pct"] = dca_alloc
        df.attrs["grid_upper"] = grid_df.attrs.get("grid_upper")
        df.attrs["grid_lower"] = grid_df.attrs.get("grid_lower")

        return df
