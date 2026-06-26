"""Value Averaging (DCA) strategy — PE-threshold based periodic investing.

On scheduled dates (e.g., first trading day of each month), buys the ETF
with an amount that varies by the current PE snapshot:

    PE < pe_low   → 2.0× base amount  (aggressive)
    pe_low-mid    → 1.5×              (moderate)
    pe_mid-high   → 1.0×              (normal)
    pe_high-max   → 0.5×              (cautious)
    PE >= pe_max  → hold (skip)       (expensive)

Uses the current PE snapshot (from Tencent).  Historical PE data is NOT
yet available — this is clearly labeled in the UI.

Future v0.2c: replace with percentile-based logic using index PE history.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy


class ValueAveragingStrategy(BaseStrategy):
    """PE-threshold periodic DCA strategy (simplified, snapshot PE)."""

    @property
    def name(self) -> str:
        return "估值定投"

    @property
    def description(self) -> str:
        return (
            "根据PE(TTM)高低调整定投金额：低估值多买、高估值少买或不买。"
            "每月固定日期执行，纪律性强，适合长期持有。"
            "⚠️ 当前使用PE快照（非历史分位），仅供参考。"
            "年化预期 6%-10%。"
        )

    def get_default_params(self) -> dict:
        return {
            "base_amount": 1000,
            "pe_low": 15.0,
            "pe_mid": 20.0,
            "pe_high": 30.0,
            "pe_max": 40.0,
            "frequency": "monthly",
            "pe_field": "pe_ttm",
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "base_amount": {
                "label": "基准金额(元)",
                "type": "number",
                "min": 100, "max": 100000, "step": 100,
                "help": "PE适中时每期定投金额",
            },
            "pe_low": {
                "label": "PE低估值线",
                "type": "number",
                "min": 5.0, "max": 25.0, "step": 1.0,
                "help": "PE低于此值 → 2倍定投",
            },
            "pe_mid": {
                "label": "PE中估值线",
                "type": "number",
                "min": 10.0, "max": 35.0, "step": 1.0,
                "help": "PE介于低-中 → 1.5倍定投",
            },
            "pe_high": {
                "label": "PE高估值线",
                "type": "number",
                "min": 15.0, "max": 45.0, "step": 1.0,
                "help": "PE介于中-高 → 1.0倍定投",
            },
            "pe_max": {
                "label": "PE停止线",
                "type": "number",
                "min": 20.0, "max": 60.0, "step": 1.0,
                "help": "PE高于此值 → 停止定投",
            },
            "frequency": {
                "label": "定投频率",
                "type": "select",
                "options": ["weekly", "monthly"],
                "help": "每周/每月第一个交易日执行",
            },
            "pe_field": {
                "label": "PE字段",
                "type": "select",
                "options": ["pe_ttm", "pe_static"],
                "help": "使用滚动市盈率或静态市盈率",
            },
        }

    def generate_signals(
        self, df: pd.DataFrame, pe_value: float | None = None, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}
        base_amount = float(params["base_amount"])
        pe_low = float(params["pe_low"])
        pe_mid = float(params["pe_mid"])
        pe_high = float(params["pe_high"])
        pe_max = float(params["pe_max"])
        frequency = str(params["frequency"])

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # --- Determine multiplier from PE ---
        if pe_value is None:
            multiplier = 1.0
            pe_note = "PE无数据 → 基准定投"
        elif pe_value <= 0:
            multiplier = 1.0
            pe_note = "PE为负(亏损) → 基准定投"
        elif pe_value < pe_low:
            multiplier = 2.0
            pe_note = f"PE({pe_value:.1f}) < {pe_low} → 2倍定投"
        elif pe_value < pe_mid:
            multiplier = 1.5
            pe_note = f"PE({pe_value:.1f}) < {pe_mid} → 1.5倍定投"
        elif pe_value < pe_high:
            multiplier = 1.0
            pe_note = f"PE({pe_value:.1f}) < {pe_high} → 基准定投"
        elif pe_value < pe_max:
            multiplier = 0.5
            pe_note = f"PE({pe_value:.1f}) < {pe_max} → 0.5倍定投"
        else:
            multiplier = 0.0
            pe_note = f"PE({pe_value:.1f}) ≥ {pe_max} → 暂停定投"

        invest_amount = base_amount * multiplier

        # --- Select scheduled dates ---
        if frequency == "weekly":
            # Group by ISO week — first trading day of each week
            df["_week"] = df["date"].dt.isocalendar().week
            df["_year"] = df["date"].dt.isocalendar().year
            df["_is_schedule"] = ~df.duplicated(subset=["_year", "_week"])
        else:  # monthly
            df["_month"] = df["date"].dt.month
            df["_year"] = df["date"].dt.year
            df["_is_schedule"] = ~df.duplicated(subset=["_year", "_month"])

        # --- Build signals ---
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        for idx in df[df["_is_schedule"]].index:
            price = float(df.at[idx, "close"])
            if pd.isna(price) or price <= 0:
                continue

            shares = int(invest_amount / price)
            if shares <= 0:
                continue

            if multiplier > 0:
                df.at[idx, "signal"] = "buy"
                df.at[idx, "signal_price"] = price
                df.at[idx, "signal_shares"] = shares
                df.at[idx, "signal_reason"] = (
                    f"{frequency}定投 | {pe_note} | "
                    f"金额 {invest_amount:.0f}元 → {shares}股"
                )

        return df.drop(columns=["_is_schedule"], errors="ignore")
