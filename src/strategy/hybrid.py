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
from src.strategy.signals import LiveSignal, DashboardCard
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

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    def get_live_signal(
        self, df: pd.DataFrame, info: dict, pe_value: float | None = None, **kwargs
    ) -> LiveSignal:
        """Generate actionable recommendation by merging DCA + Grid sub-signals.

        Delegates to ``ValueAveragingStrategy.get_live_signal()`` and
        ``GridTradingStrategy.get_live_signal()``, then combines results
        according to the multi-pool merge rules defined for this strategy.
        """
        params = {**self.get_default_params(), **kwargs}

        # --- Handle empty / missing data ---
        if df is None or df.empty:
            return LiveSignal(
                action="hold",
                reason="无历史数据，无法生成混合策略信号",
                current_zone="未知",
            )

        # --- Sub-strategy param split ---
        dca_params = {
            "base_amount": params["dca_base_amount"],
            "pe_low": params["pe_low"],
            "pe_mid": params["pe_mid"],
            "pe_high": params["pe_high"],
            "pe_max": params["pe_max"],
            "frequency": params["dca_frequency"],
        }
        grid_params = {
            "grid_count": params["grid_count"],
            "upper_padding_pct": params["upper_padding_pct"],
            "lower_padding_pct": params["lower_padding_pct"],
            "position_per_grid_pct": params["position_per_grid_pct"],
        }

        # --- Current price ---
        latest_price = _safe_float(
            info.get("current_price") if info else None,
            float(df.iloc[-1]["close"]) if not df.empty else None,
        )

        # --- Fetch sub-strategy signals (fail-safe) ---
        dca_signal = _try_live_signal(
            ValueAveragingStrategy, df, info, pe_value=pe_value, extra=dca_params
        )
        grid_signal = _try_live_signal(
            GridTradingStrategy, df, info, extra=grid_params
        )

        dca_zone = dca_signal.current_zone if dca_signal else "未知"
        grid_zone = grid_signal.current_zone if grid_signal else "未知"
        dca_action = dca_signal.action if dca_signal else "hold"
        grid_action = grid_signal.action if grid_signal else "hold"

        # --- Merge rules ---
        # Both buy → strongest buy
        if dca_action == "buy" and grid_action == "buy":
            return LiveSignal(
                action="buy",
                current_price=latest_price,
                trigger_description="定投 + 网格同时买入",
                reason=_merge_reason("定投+网格同时买入", dca_signal, grid_signal),
                urgency_level=_max_urgency(dca_signal, grid_signal),
                suggested_shares=_safe_int(dca_signal.suggested_shares) + _safe_int(grid_signal.suggested_shares),
                suggested_amount=_safe_float(dca_signal.suggested_amount) + _safe_float(grid_signal.suggested_amount),
                current_zone=f"定投: {dca_zone} | 网格: {grid_zone}",
            )

        # DCA buy only
        if dca_action == "buy":
            return LiveSignal(
                action="buy",
                current_price=latest_price,
                trigger_description=_safe_str(dca_signal.trigger_description, "定投买入信号"),
                reason=_prefixed_reason("定投", dca_signal),
                urgency_level=_safe_str(dca_signal.urgency_level, "medium"),
                suggested_shares=_safe_int(dca_signal.suggested_shares),
                suggested_amount=_safe_float(dca_signal.suggested_amount),
                current_zone=f"定投: {dca_zone} | 网格: {grid_zone}",
            )

        # Grid buy only
        if grid_action == "buy":
            return LiveSignal(
                action="buy",
                current_price=latest_price,
                trigger_description=_safe_str(grid_signal.trigger_description, "网格买入信号"),
                reason=_prefixed_reason("网格", grid_signal),
                urgency_level=_safe_str(grid_signal.urgency_level, "medium"),
                suggested_shares=_safe_int(grid_signal.suggested_shares),
                suggested_amount=_safe_float(grid_signal.suggested_amount),
                current_zone=f"定投: {dca_zone} | 网格: {grid_zone}",
            )

        # Grid sell (DCA rarely sells, so grid sell dominates)
        if grid_action == "sell":
            return LiveSignal(
                action="sell",
                current_price=latest_price,
                trigger_description=_safe_str(grid_signal.trigger_description, "网格卖出信号"),
                reason=_prefixed_reason("网格", grid_signal),
                urgency_level=_safe_str(grid_signal.urgency_level, "medium"),
                suggested_shares=_safe_int(grid_signal.suggested_shares),
                suggested_amount=_safe_float(grid_signal.suggested_amount),
                current_zone=f"定投: {dca_zone} | 网格: {grid_zone}",
            )

        # DCA sell (fallback)
        if dca_action == "sell":
            return LiveSignal(
                action="sell",
                current_price=latest_price,
                trigger_description=_safe_str(dca_signal.trigger_description, "定投卖出信号"),
                reason=_prefixed_reason("定投", dca_signal),
                urgency_level=_safe_str(dca_signal.urgency_level, "low"),
                current_zone=f"定投: {dca_zone} | 网格: {grid_zone}",
            )

        # Default: hold
        return LiveSignal(
            action="hold",
            current_price=latest_price,
            reason="定投和网格均无交易信号",
            current_zone=f"定投: {dca_zone} | 网格: {grid_zone}",
        )

    # ------------------------------------------------------------------
    # Dashboard cards
    # ------------------------------------------------------------------

    def get_dashboard_cards(
        self, df: pd.DataFrame, info: dict, pe_value: float | None = None, **kwargs
    ) -> list[DashboardCard]:
        """Return 4 hybrid-specific dashboard cards.

        Cards:
        1. 资金分配 — progress bar showing DCA vs Grid allocation
        2. 定投状态 — DCA sub-strategy status (delegated)
        3. 网格状态 — Grid sub-strategy status (delegated)
        4. 组合状态 — merged summary of both sub-strategies
        """
        params = {**self.get_default_params(), **kwargs}
        dca_alloc = float(params["dca_allocation_pct"])
        grid_alloc = 1.0 - dca_alloc

        # Sub-strategy param split
        dca_params = {
            "base_amount": params["dca_base_amount"],
            "pe_low": params["pe_low"],
            "pe_mid": params["pe_mid"],
            "pe_high": params["pe_high"],
            "pe_max": params["pe_max"],
            "frequency": params["dca_frequency"],
        }
        grid_params = {
            "grid_count": params["grid_count"],
            "upper_padding_pct": params["upper_padding_pct"],
            "lower_padding_pct": params["lower_padding_pct"],
            "position_per_grid_pct": params["position_per_grid_pct"],
        }

        cards: list[DashboardCard] = []

        # ---- Card 1: 资金分配 ----
        cards.append(DashboardCard(
            card_id="hybrid_allocation",
            title="资金分配",
            card_type="progress",
            content={
                "dca_allocation_pct": round(dca_alloc, 2),
                "grid_allocation_pct": round(grid_alloc, 2),
                "label": f"定投 {dca_alloc:.0%} | 网格 {grid_alloc:.0%}",
            },
            priority=1,
        ))

        # ---- Card 2: DCA status (delegate to sub-strategy) ----
        dca_card = _try_dashboard_card(
            ValueAveragingStrategy, df, info, pe_value=pe_value, extra=dca_params,
            fallback_title="定投状态",
        )
        cards.append(dca_card)

        # ---- Card 3: Grid status (delegate to sub-strategy) ----
        grid_card = _try_dashboard_card(
            GridTradingStrategy, df, info, extra=grid_params,
            fallback_title="网格状态",
        )
        cards.append(grid_card)

        # ---- Card 4: 组合状态 (merged summary) ----
        dca_signal = _try_live_signal(
            ValueAveragingStrategy, df, info, pe_value=pe_value, extra=dca_params
        )
        grid_signal = _try_live_signal(
            GridTradingStrategy, df, info, extra=grid_params
        )

        dca_summary = (
            f"{dca_signal.current_zone} → {dca_signal.action}"
            if dca_signal else "未就绪"
        )
        grid_summary = (
            f"{grid_signal.current_zone} → {grid_signal.action}"
            if grid_signal else "未就绪"
        )

        cards.append(DashboardCard(
            card_id="hybrid_summary",
            title="组合状态",
            card_type="info",
            content={
                "dca_status": dca_summary,
                "grid_status": grid_summary,
                "dca_allocation_pct": round(dca_alloc, 2),
                "grid_allocation_pct": round(grid_alloc, 2),
            },
            priority=1,
        ))

        return cards


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _try_live_signal(
    strategy_cls, df: pd.DataFrame, info: dict,
    pe_value: float | None = None, extra: dict | None = None,
) -> LiveSignal | None:
    """Safely call ``strategy_cls().get_live_signal()``, returning None on any error."""
    try:
        inst = strategy_cls()
        kwargs = extra.copy() if extra else {}
        if pe_value is not None:
            kwargs["pe_value"] = pe_value
        return inst.get_live_signal(df, info, **kwargs)
    except Exception:
        return None


def _try_dashboard_card(
    strategy_cls, df: pd.DataFrame, info: dict,
    pe_value: float | None = None, extra: dict | None = None,
    fallback_title: str = "状态",
) -> DashboardCard:
    """Safely delegate to ``strategy_cls().get_dashboard_cards()``, returning
    the first card on success or a fallback ``info`` card on failure."""
    try:
        inst = strategy_cls()
        kwargs = extra.copy() if extra else {}
        if pe_value is not None:
            kwargs["pe_value"] = pe_value
        cards = inst.get_dashboard_cards(df, info, **kwargs)
        if cards:
            return cards[0]
    except Exception:
        pass
    return DashboardCard(
        card_id=f"hybrid_{strategy_cls.__name__.lower()}_fallback",
        title=fallback_title,
        card_type="info",
        content={"status": "子策略未就绪"},
        priority=2,
    )


def _merge_reason(prefix: str, dca, grid) -> str:
    """Combine reasoning strings from both sub-signals."""
    dca_r = _safe_str(dca.reason) if dca else ""
    grid_r = _safe_str(grid.reason) if grid else ""
    if dca_r and grid_r:
        return f"{prefix}: [{dca_r}] | [{grid_r}]"
    return prefix


def _prefixed_reason(tag: str, signal) -> str:
    """Return reason prefixed with a sub-strategy tag."""
    if signal is None:
        return f"[{tag}] 信号生成失败"
    return f"[{tag}] {_safe_str(signal.reason, '')}"


def _max_urgency(dca, grid) -> str:
    """Return the higher urgency level of two signals."""
    order = {"high": 3, "medium": 2, "low": 1}
    dca_u = _safe_str(dca.urgency_level if dca else "", "low")
    grid_u = _safe_str(grid.urgency_level if grid else "", "low")
    dca_score = order.get(dca_u, 1)
    grid_score = order.get(grid_u, 1)
    if dca_score >= grid_score:
        return dca_u
    return grid_u


def _safe_str(val, default: str = "") -> str:
    """Coerce *val* to str; return *default* when falsy or NaN."""
    if val is None:
        return default
    s = str(val)
    if s == "" or s.lower() == "nan":
        return default
    return s


def _safe_int(val, default: int = 0) -> int:
    """Coerce *val* to int; return *default* on error."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce *val* to float; return *default* on error."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
