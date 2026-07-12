"""Grid Trading strategy — buy low, sell high within a predefined price band.

Divides the historical price range into N equal levels. When the price
crosses below a level where we have no inventory → buy. When it crosses
above a level where we have inventory → sell.

State (grid inventory per level) is managed by the BacktestEngine, not
by this strategy. The strategy only marks which grid level the price
is at on each bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.signals import DashboardCard, LiveSignal


class GridTradingStrategy(BaseStrategy):
    """Price-band grid trading strategy.

    Generates buy/sell signals based on which grid level the current
    price occupies.  The engine tracks per-level inventory across bars.
    """

    @property
    def name(self) -> str:
        return "网格交易"

    @property
    def description(self) -> str:
        return (
            "将历史价格范围等分为N档，价穿上档卖出、价穿下档买入，"
            "在震荡市中反复收割波动。上涨市中容易卖飞，下跌市中容易"
            "满仓套牢。年化预期 7%-12%。"
        )

    def get_default_params(self) -> dict:
        return {
            "grid_count": 10,
            "upper_padding_pct": 0.05,
            "lower_padding_pct": 0.05,
            "position_per_grid_pct": 0.08,
            "base_price_source": "close",
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "grid_count": {
                "label": "网格档数",
                "type": "number",
                "min": 3, "max": 50, "step": 1,
                "help": "价格区间等分为多少档（3-50）",
            },
            "upper_padding_pct": {
                "label": "上延比例",
                "type": "slider",
                "min": 0.0, "max": 0.20, "step": 0.01,
                "help": "在历史最高价之上额外延伸的比例",
            },
            "lower_padding_pct": {
                "label": "下延比例",
                "type": "slider",
                "min": 0.0, "max": 0.20, "step": 0.01,
                "help": "在历史最低价之下额外延伸的比例",
            },
            "position_per_grid_pct": {
                "label": "每档仓位",
                "type": "slider",
                "min": 0.02, "max": 0.20, "step": 0.01,
                "help": "每档网格使用的资金比例（占总资产）",
            },
            "base_price_source": {
                "label": "基准价",
                "type": "select",
                "options": ["close", "mid"],
                "help": "收盘价 或 (最高+最低)/2",
            },
        }

    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}
        grid_count = int(params["grid_count"])
        upper_pad = float(params["upper_padding_pct"])
        lower_pad = float(params["lower_padding_pct"])
        pos_per_grid = float(params["position_per_grid_pct"])
        price_src = str(params["base_price_source"])

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # --- Compute grid levels from historical range ---
        if price_src == "mid":
            high = df["high"].max()
            low = df["low"].min()
        else:
            high = df["close"].max()
            low = df["close"].min()

        if pd.isna(high) or pd.isna(low) or high <= low:
            df["signal"] = "hold"
            df["signal_price"] = df["close"]
            df["signal_shares"] = 0
            df["signal_reason"] = ""
            return df

        grid_upper = high * (1.0 + upper_pad)
        grid_lower = low * (1.0 - lower_pad)
        grid_step = (grid_upper - grid_lower) / grid_count

        # Build level boundaries — level 0 is the bottom, level N-1 is the top
        level_prices = [
            grid_lower + i * grid_step for i in range(grid_count + 1)
        ]
        # Midpoint of each level (the "trigger line")
        level_mids = [
            (level_prices[i] + level_prices[i + 1]) / 2
            for i in range(grid_count)
        ]

        # Store grid metadata for the engine
        df.attrs["grid_level_prices"] = level_prices
        df.attrs["grid_level_mids"] = level_mids
        df.attrs["grid_upper"] = grid_upper
        df.attrs["grid_lower"] = grid_lower
        df.attrs["grid_step"] = grid_step

        # --- Assign each bar to a grid level ---
        # Price goes into level i if between level_prices[i] and level_prices[i+1]
        current_price = df["close"].values
        level_array = np.digitize(current_price, level_prices) - 1
        level_array = np.clip(level_array, 0, grid_count - 1)

        df["_grid_level"] = level_array
        df["_grid_level_prev"] = df["_grid_level"].shift(1).fillna(-1).astype(int)

        # --- Generate signals based on level crossings ---
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        for idx in df.index:
            lvl = int(df.at[idx, "_grid_level"])
            prev_lvl = int(df.at[idx, "_grid_level_prev"])
            price = float(df.at[idx, "close"])

            if prev_lvl < 0:
                continue

            # Price dropped to a lower level → buy signal
            if lvl < prev_lvl:
                df.at[idx, "signal"] = "buy"
                df.at[idx, "signal_price"] = price
                # Shares for one grid level worth of position
                budget = pos_per_grid * 100_000  # placeholder — engine adjusts
                df.at[idx, "signal_shares"] = max(1, int(budget / price))
                df.at[idx, "signal_reason"] = (
                    f"价格 {price:.3f} 降至第{lvl+1}档 "
                    f"(区间 {level_prices[lvl]:.3f}-{level_prices[lvl+1]:.3f})，网格买入"
                )

            # Price rose to a higher level → sell signal
            elif lvl > prev_lvl:
                df.at[idx, "signal"] = "sell"
                df.at[idx, "signal_price"] = price
                budget = pos_per_grid * 100_000
                df.at[idx, "signal_shares"] = max(1, int(budget / price))
                df.at[idx, "signal_reason"] = (
                    f"价格 {price:.3f} 升至第{lvl+1}档 "
                    f"(区间 {level_prices[lvl]:.3f}-{level_prices[lvl+1]:.3f})，网格卖出"
                )

        # Clean up internal columns
        return df.drop(columns=["_grid_level", "_grid_level_prev"])

    # ------------------------------------------------------------------
    # Grid computation helper
    # ------------------------------------------------------------------

    def _compute_grid(
        self, df: pd.DataFrame, **kwargs
    ) -> dict | None:
        """Compute grid levels from historical price range.

        Returns a dict with grid metadata, or None if the data is
        insufficient (empty df, invalid range, etc.).
        """
        params = {**self.get_default_params(), **kwargs}
        grid_count = int(params["grid_count"])
        upper_pad = float(params["upper_padding_pct"])
        lower_pad = float(params["lower_padding_pct"])
        price_src = str(params["base_price_source"])

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

        if price_src == "mid":
            high = df_sorted["high"].max()
            low = df_sorted["low"].min()
        else:
            high = df_sorted["close"].max()
            low = df_sorted["close"].min()

        if pd.isna(high) or pd.isna(low) or high <= low:
            return None

        grid_upper = high * (1.0 + upper_pad)
        grid_lower = low * (1.0 - lower_pad)
        grid_step = (grid_upper - grid_lower) / grid_count

        level_prices = [
            grid_lower + i * grid_step for i in range(grid_count + 1)
        ]
        level_mids = [
            (level_prices[i] + level_prices[i + 1]) / 2
            for i in range(grid_count)
        ]

        return {
            "params": params,
            "grid_count": grid_count,
            "grid_upper": grid_upper,
            "grid_lower": grid_lower,
            "grid_step": grid_step,
            "level_prices": level_prices,
            "level_mids": level_mids,
        }

    # ------------------------------------------------------------------
    # Live signal & dashboard cards
    # ------------------------------------------------------------------

    def get_live_signal(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> LiveSignal:
        """Generate an actionable recommendation based on current market data.

        Computes the grid from historical range, locates the current price
        on the grid, and returns buy/sell/hold with the nearest trigger.
        """
        if df is None or df.empty:
            return LiveSignal(
                action="hold",
                reason="无历史数据，无法计算网格区间",
                urgency_level="low",
            )

        grid = self._compute_grid(df, **kwargs)
        if grid is None:
            return LiveSignal(
                action="hold",
                reason="无法计算网格区间（历史价格范围无效）",
                urgency_level="low",
            )

        grid_count = grid["grid_count"]
        level_prices = grid["level_prices"]
        pos_per_grid = float(grid["params"]["position_per_grid_pct"])

        # --- Current price (real-time > last close) ---
        current_price = info.get("current_price") if info else None
        if current_price is None:
            df_by_date = df.sort_values(
                "date", ascending=True
            ).reset_index(drop=True)
            current_price = float(df_by_date.iloc[-1]["close"])

        # --- Which grid level? ---
        level = int(np.clip(
            np.digitize(current_price, level_prices) - 1,
            0, grid_count - 1,
        ))

        # --- Signal logic (bottom/top 30%) ---
        boundary = max(1, int(grid_count * 0.3))
        if level < boundary:
            action = "buy"
            zone = f"第{level+1}档 (低位)"
        elif level >= grid_count - boundary:
            action = "sell"
            zone = f"第{level+1}档 (高位)"
        else:
            action = "hold"
            zone = f"第{level+1}档 (中位)"

        # --- Nearest grid boundary as next trigger ---
        lower_bound = level_prices[level]
        upper_bound = level_prices[level + 1]
        if (current_price - lower_bound) < (upper_bound - current_price):
            next_trigger = lower_bound
        else:
            next_trigger = upper_bound

        # --- Suggested shares (one grid level's worth) ---
        budget = pos_per_grid * 100_000
        suggested_shares = max(1, int(budget / current_price)) if current_price > 0 else 0

        return LiveSignal(
            action=action,
            current_price=round(current_price, 4),
            suggested_shares=suggested_shares,
            suggested_amount=round(suggested_shares * current_price, 2),
            trigger_description=(
                f"当前价格 {current_price:.3f}，位于{zone}"
            ),
            next_trigger_price=round(next_trigger, 4),
            reason=(
                f"网格交易：价格在{zone}，"
                f"距下一触发线 {abs(current_price - next_trigger):.3f}"
            ),
            urgency_level="high" if action != "hold" else "low",
            portions_used=level + 1,
            portions_total=grid_count,
            current_zone=zone,
        )

    def get_dashboard_cards(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> list[DashboardCard]:
        """Return strategy-specific info cards for the dashboard grid.

        Three cards: grid levels (with current highlighted), trigger prices,
        and grid range summary.
        """
        if df is None or df.empty:
            return []

        grid = self._compute_grid(df, **kwargs)
        if grid is None:
            return []

        grid_count = grid["grid_count"]
        grid_upper = grid["grid_upper"]
        grid_lower = grid["grid_lower"]
        grid_step = grid["grid_step"]
        level_prices = grid["level_prices"]

        # --- Current price ---
        current_price = info.get("current_price") if info else None
        if current_price is None:
            df_by_date = df.sort_values(
                "date", ascending=True
            ).reset_index(drop=True)
            current_price = float(df_by_date.iloc[-1]["close"])

        # --- Current grid level ---
        level = int(np.clip(
            np.digitize(current_price, level_prices) - 1,
            0, grid_count - 1,
        ))

        # --- Card A: 网格档位 ---
        level_rows = []
        for i in range(grid_count):
            level_rows.append({
                "label": f"第{i+1}档",
                "lower": round(level_prices[i], 4),
                "upper": round(level_prices[i + 1], 4),
                "is_current": i == level,
            })

        cards: list[DashboardCard] = [
            DashboardCard(
                card_id="grid_levels",
                title="网格档位",
                card_type="info",
                content={
                    "levels": level_rows,
                    "current_level": level + 1,
                    "total_levels": grid_count,
                },
                priority=1,
            ),
        ]

        # --- Card B: 触发价位 ---
        next_buy = round(level_prices[level], 4) if level > 0 else None
        next_sell = (
            round(level_prices[level + 1], 4)
            if level < grid_count - 1
            else None
        )

        cards.append(DashboardCard(
            card_id="grid_triggers",
            title="触发价位",
            card_type="trigger",
            content={
                "next_buy": next_buy,
                "next_sell": next_sell,
                "current_price": round(current_price, 4),
            },
            priority=1,
        ))

        # --- Card C: 网格区间 ---
        cards.append(DashboardCard(
            card_id="grid_range",
            title="网格区间",
            card_type="info",
            content={
                "grid_upper": round(grid_upper, 4),
                "grid_lower": round(grid_lower, 4),
                "grid_step": round(grid_step, 4),
                "total_levels": grid_count,
            },
            priority=2,
        ))

        return cards
