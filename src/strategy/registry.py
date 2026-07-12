"""Strategy Registry — single source of truth for all available strategies.

Usage::

    from src.strategy.registry import get_registry

    registry = get_registry()
    strategy = registry.get_by_name("4%定投法")
    names = registry.get_names()  # → ["趋势跟随", "网格交易", ...]
"""

from __future__ import annotations

from src.strategy.base import BaseStrategy


class StrategyRegistry:
    """Central registry of all strategy instances.

    Singleton-like — use ``get_registry()`` to obtain the shared instance.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, BaseStrategy] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance (keyed by ``strategy.name``)."""
        self._strategies[strategy.name] = strategy

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_names(self) -> list[str]:
        """Return all registered strategy names in registration order."""
        return list(self._strategies.keys())

    def get_all(self) -> list[BaseStrategy]:
        """Return all registered strategy instances."""
        return list(self._strategies.values())

    def get_by_name(self, name: str) -> BaseStrategy:
        """Look up a strategy by its Chinese display name.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._strategies:
            raise KeyError(
                f"策略 '{name}' 未注册。可用策略: {self.get_names()}"
            )
        return self._strategies[name]

    def get_default(self) -> BaseStrategy:
        """Return a sensible default strategy (估值定投)."""
        default = "估值定投"
        if default in self._strategies:
            return self._strategies[default]
        return list(self._strategies.values())[0]

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: StrategyRegistry | None = None


def get_registry() -> StrategyRegistry:
    """Return the module-level StrategyRegistry singleton.

    On first call, auto-registers all built-in strategies.
    """
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
        _auto_register(_registry)
    return _registry


def _auto_register(reg: StrategyRegistry) -> None:
    """Register all built-in strategies (lazy imports to avoid circular deps)."""
    from src.strategy.trend_following import TrendFollowingStrategy
    from src.strategy.grid_trading import GridTradingStrategy
    from src.strategy.value_averaging import ValueAveragingStrategy
    from src.strategy.hybrid import HybridStrategy
    from src.strategy.four_percent_dca import FourPercentDCAStrategy
    from src.strategy.short_term_band import ShortTermBandStrategy
    from src.strategy.fast_band_4pct import FastBand4PctStrategy

    reg.register(TrendFollowingStrategy())
    reg.register(GridTradingStrategy())
    reg.register(ValueAveragingStrategy())
    reg.register(HybridStrategy())
    reg.register(FourPercentDCAStrategy())
    reg.register(ShortTermBandStrategy())
    reg.register(FastBand4PctStrategy())
