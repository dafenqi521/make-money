"""Trading strategy implementations for ETF backtesting.

Strategies:
  - TrendFollowing: MA golden-cross / death-cross signals
  - GridTrading:   price-band grid buy-low sell-high
  - ValueAveraging: PE-threshold DCA (simplified, no historical PE)
  - Hybrid:        DCA base + grid overlay
"""
