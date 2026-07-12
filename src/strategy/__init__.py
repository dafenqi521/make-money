"""Trading strategy implementations for ETF backtesting.

Strategies:
  - TrendFollowing:   MA golden-cross / death-cross signals
  - GridTrading:      price-band grid buy-low sell-high
  - ValueAveraging:   PE-threshold DCA (simplified, no historical PE)
  - Hybrid:           DCA base + grid overlay
  - FourPercentDCA:   雷牛牛4%定投法 — 每跌4%定投一份，低估买高估卖

Exports:
  - get_registry() → StrategyRegistry singleton
  - LiveSignal, DashboardCard — UI data contracts
"""

from src.strategy.registry import get_registry
from src.strategy.signals import LiveSignal, DashboardCard
