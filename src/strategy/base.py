"""Abstract base class for all trading strategies.

Strategies are STATELESS and VECTORIZED: ``generate_signals()`` receives
the full historical DataFrame and appends signal columns.  Portfolio state
tracking is handled by the BacktestEngine, not by strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


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

    def __repr__(self) -> str:
        return f"<{self.name} strategy>"
