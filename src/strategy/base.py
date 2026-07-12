"""Abstract base class for all trading strategies.

Strategies are STATELESS and VECTORIZED: ``generate_signals()`` receives
the full historical DataFrame and appends signal columns.  Portfolio state
tracking is handled by the BacktestEngine, not by strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import pandas as pd

from src.strategy.signals import LiveSignal, DashboardCard

if TYPE_CHECKING:
    from src.data.pe_history import PEPercentile
    from src.data.macro_pulse import MacroPulse


class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Subclasses MUST override:
      - name (property)
      - description (property)
      - get_default_params()
      - get_param_descriptions()
      - generate_signals(df, **kwargs)
    """

    # ------------------------------------------------------------------
    # Metadata (override in subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short Chinese display name, e.g. "趋势跟随"."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph strategy description in Chinese."""
        ...

    # ------------------------------------------------------------------
    # Parameter schema (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return default parameter values as a dict."""
        ...

    @abstractmethod
    def get_param_descriptions(self) -> dict[str, dict]:
        """Return parameter metadata for UI rendering.

        Each key maps to: {
            "label": str,       # Chinese label
            "type": str,        # "select", "slider", "number"
            "options": list,    # for "select" type
            "min": float,       # for "slider"/"number"
            "max": float,
            "step": float,
            "help": str,        # tooltip
        }
        """
        ...

    # ------------------------------------------------------------------
    # Core signal generation (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Generate trading signals from OHLCV + MA data.

        Args:
            df: DataFrame with columns [date, open, high, low, close,
                volume, ma5, ma10, ma20, change_pct, amplitude].
            **kwargs: Strategy-specific parameters.

        Returns:
            DataFrame with original columns PLUS:
              signal       — "buy" / "sell" / "hold"
              signal_price — execution price (close of signal bar)
              signal_shares — number of shares to trade
              signal_reason — human-readable rationale
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate_params(self, params: dict) -> list[str]:
        """Validate parameters, returning list of error messages."""
        defaults = self.get_default_params()
        errors = []
        for key, default in defaults.items():
            if key not in params:
                errors.append(f"缺少参数: {key}")
                continue
            val = params[key]
            if isinstance(default, (int, float)) and not isinstance(val, (int, float)):
                errors.append(f"参数 {key} 应为数字，收到 {type(val).__name__}")
            elif isinstance(default, str) and not isinstance(val, str):
                errors.append(f"参数 {key} 应为字符串，收到 {type(val).__name__}")
        return errors

    # ------------------------------------------------------------------
    # Live signal & dashboard (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_live_signal(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> LiveSignal:
        """Generate an actionable recommendation based on current market data.

        Uses the latest bar from *df* plus the real-time quote in *info*
        to determine what the strategy would do *right now*.

        Args:
            df: Historical OHLCV DataFrame (date-descending, from fetch_etf_hist).
            info: Real-time quote dict (from fetch_etf_info).
            **kwargs: Strategy-specific parameters.

        Returns:
            LiveSignal with action, trigger prices, zone, and reasoning.
        """
        ...

    @abstractmethod
    def get_dashboard_cards(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> list[DashboardCard]:
        """Return strategy-specific info cards for the dashboard grid.

        Args:
            df: Historical OHLCV DataFrame.
            info: Real-time quote dict.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of DashboardCard objects (order = display order).
        """
        ...

    def get_signal_markers(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Extract buy/sell markers for chart overlay.

        Default: runs generate_signals() and filters to non-hold rows.
        Override for custom marker logic.

        Returns:
            DataFrame with columns [date, close, signal, signal_reason].
        """
        sig_df = self.generate_signals(df, **kwargs)
        markers = sig_df[sig_df["signal"].isin(["buy", "sell"])][
            ["date", "close", "signal", "signal_reason"]
        ].copy()
        return markers

    def __repr__(self) -> str:
        return f"<{self.name} strategy>"
