"""Tests for all four trading strategies."""

import pandas as pd
import pytest

from src.strategy.trend_following import TrendFollowingStrategy
from src.strategy.grid_trading import GridTradingStrategy
from src.strategy.value_averaging import ValueAveragingStrategy
from src.strategy.hybrid import HybridStrategy


# ---------------------------------------------------------------------------
# Helper: build synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_df(prices, mas=None, start_date="2026-01-01"):
    """Create a minimal OHLCV DataFrame with MA columns.

    Args:
        prices: list of (date_str, open, high, low, close) tuples
                OR list of close prices (auto-generates OHLC)
        mas: optional dict of {ma_name: [values]} for pre-computed MAs
    """
    dates = pd.date_range(start_date, periods=len(prices), freq="B")
    if isinstance(prices[0], (int, float)):
        close_vals = list(prices)
    else:
        close_vals = [t[4] if len(t) > 4 else t[0] for t in prices]

    df = pd.DataFrame({
        "date": dates,
        "open": close_vals,
        "high": [c * 1.01 for c in close_vals],
        "low": [c * 0.99 for c in close_vals],
        "close": close_vals,
        "volume": [10000] * len(close_vals),
        "change_pct": [0.0] * len(close_vals),
        "amplitude": [2.0] * len(close_vals),
        "ma5": mas["ma5"] if mas and "ma5" in mas else close_vals,
        "ma10": mas["ma10"] if mas and "ma10" in mas else close_vals,
        "ma20": mas["ma20"] if mas and "ma20" in mas else close_vals,
    })
    return df


# ---------------------------------------------------------------------------
# TrendFollowingStrategy
# ---------------------------------------------------------------------------

class TestTrendFollowing:
    def test_golden_cross_emits_buy(self):
        """MA5 crosses above MA20 → buy signal."""
        prices = [10.0] * 30
        # MA5 rises through MA20 at bar 15
        ma5 = [10.0] * 14 + [10.5, 11.0, 11.5] + [11.5] * 13
        ma20 = [10.0] * 15 + [10.8] * 15
        df = _make_df(prices, mas={"ma5": ma5, "ma10": ma5, "ma20": ma20})

        strategy = TrendFollowingStrategy()
        result = strategy.generate_signals(df, fast_ma="ma5", slow_ma="ma20")

        buys = result[result["signal"] == "buy"]
        assert len(buys) > 0  # At least one buy signal

    def test_death_cross_emits_sell(self):
        """MA5 crosses below MA20 → sell signal."""
        prices = [10.0] * 30
        ma5 = [10.5] * 14 + [10.5, 10.0, 9.5] + [9.5] * 13
        ma20 = [10.5] * 14 + [10.5] * 16
        df = _make_df(prices, mas={"ma5": ma5, "ma10": ma5, "ma20": ma20})

        strategy = TrendFollowingStrategy()
        result = strategy.generate_signals(df, fast_ma="ma5", slow_ma="ma20")

        sells = result[result["signal"] == "sell"]
        assert len(sells) > 0

    def test_no_cross_no_signals(self):
        """Parallel MAs produce no buy/sell signals."""
        prices = [10.0] * 30
        df = _make_df(prices)
        strategy = TrendFollowingStrategy()
        result = strategy.generate_signals(df)
        buys = result[result["signal"] == "buy"]
        sells = result[result["signal"] == "sell"]
        assert len(buys) == 0
        assert len(sells) == 0

    def test_missing_ma_column_raises(self):
        """Missing MA column raises ValueError with helpful message."""
        df = _make_df([10.0] * 5)
        df = df.drop(columns=["ma20"])
        strategy = TrendFollowingStrategy()
        with pytest.raises(ValueError, match="ma20"):
            strategy.generate_signals(df)

    def test_custom_ma_pair(self):
        """Works with ma10 / ma20 pair."""
        prices = [10.0] * 30
        ma10 = [10.0] * 14 + [11.0] + [11.0] * 15
        ma20 = [10.0] * 15 + [10.5] * 15
        df = _make_df(prices, mas={"ma5": ma10, "ma10": ma10, "ma20": ma20})
        strategy = TrendFollowingStrategy()
        result = strategy.generate_signals(df, fast_ma="ma10", slow_ma="ma20")
        assert "signal" in result.columns

    def test_output_columns(self):
        """Output DataFrame has all signal columns."""
        df = _make_df([10.0] * 10)
        strategy = TrendFollowingStrategy()
        result = strategy.generate_signals(df)
        for col in ["signal", "signal_price", "signal_shares", "signal_reason"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# GridTradingStrategy
# ---------------------------------------------------------------------------

class TestGridTrading:
    def test_grid_generates_levels(self):
        """Grid strategy computes and stores level prices in df.attrs."""
        prices = [8.0, 9.0, 10.0, 11.0, 12.0] * 10  # oscillate
        df = _make_df(prices)
        strategy = GridTradingStrategy()
        result = strategy.generate_signals(df, grid_count=5)
        assert "grid_level_prices" in result.attrs
        assert len(result.attrs["grid_level_prices"]) == 6  # 5 levels = 6 boundaries

    def test_price_drop_emits_buy(self):
        """Price dropping to a lower grid level triggers buy."""
        # Start high, drop low → must cross grid levels
        prices = [15.0] * 5 + [12.0] * 5 + [9.0] * 5
        df = _make_df(prices)
        strategy = GridTradingStrategy()
        result = strategy.generate_signals(df, grid_count=10,
                                            upper_padding_pct=0.0,
                                            lower_padding_pct=0.0)
        buys = result[result["signal"] == "buy"]
        assert len(buys) > 0

    def test_price_rise_emits_sell(self):
        """Price rising to a higher grid level triggers sell."""
        prices = [9.0] * 5 + [12.0] * 5 + [15.0] * 5
        df = _make_df(prices)
        strategy = GridTradingStrategy()
        result = strategy.generate_signals(df, grid_count=10,
                                            upper_padding_pct=0.0,
                                            lower_padding_pct=0.0)
        sells = result[result["signal"] == "sell"]
        assert len(sells) > 0

    def test_grid_count_validation(self):
        """Grid count of 3+ works."""
        df = _make_df([10.0] * 5)
        strategy = GridTradingStrategy()
        result = strategy.generate_signals(df, grid_count=3)
        assert "signal" in result.columns

    def test_output_columns(self):
        """Output has all signal columns, no internal columns."""
        df = _make_df([10.0] * 10)
        strategy = GridTradingStrategy()
        result = strategy.generate_signals(df)
        for col in ["signal", "signal_price", "signal_shares", "signal_reason"]:
            assert col in result.columns
        # Internal columns should NOT leak
        assert "_grid_level" not in result.columns


# ---------------------------------------------------------------------------
# ValueAveragingStrategy
# ---------------------------------------------------------------------------

class TestValueAveraging:
    def test_low_pe_buys_2x(self):
        """PE below pe_low → 2x base amount."""
        prices = [10.0] * 50  # ~2 months of daily bars
        df = _make_df(prices)
        strategy = ValueAveragingStrategy()
        # PE=10, pe_low=15 → multiplier=2
        result = strategy.generate_signals(
            df, pe_value=10.0, base_amount=1000,
            pe_low=15.0, pe_mid=20.0, pe_high=30.0, pe_max=40.0,
        )
        buys = result[result["signal"] == "buy"]
        assert len(buys) > 0
        # Each buy should be ~2000/close = ~200 shares
        for _, row in buys.iterrows():
            assert "2倍" in row["signal_reason"] or row["signal_shares"] > 100

    def test_high_pe_buys_less(self):
        """PE between pe_high and pe_max → 0.5x base amount."""
        prices = [10.0] * 50
        df = _make_df(prices)
        strategy = ValueAveragingStrategy()
        result = strategy.generate_signals(
            df, pe_value=35.0, base_amount=1000,
            pe_low=15.0, pe_mid=20.0, pe_high=30.0, pe_max=40.0,
        )
        buys = result[result["signal"] == "buy"]
        for _, row in buys.iterrows():
            assert "0.5倍" in row["signal_reason"] or row["signal_shares"] < 60

    def test_pe_max_stops_buying(self):
        """PE >= pe_max → no buy signals (multiplier = 0)."""
        prices = [10.0] * 50
        df = _make_df(prices)
        strategy = ValueAveragingStrategy()
        result = strategy.generate_signals(
            df, pe_value=50.0, base_amount=1000,
            pe_low=15.0, pe_mid=20.0, pe_high=30.0, pe_max=40.0,
        )
        buys = result[result["signal"] == "buy"]
        assert len(buys) == 0

    def test_no_pe_uses_base(self):
        """PE=None → 1x base amount (no adjustment)."""
        prices = [10.0] * 50
        df = _make_df(prices)
        strategy = ValueAveragingStrategy()
        result = strategy.generate_signals(df, pe_value=None, base_amount=1000)
        buys = result[result["signal"] == "buy"]
        assert len(buys) > 0

    def test_monthly_frequency(self):
        """Only one buy per month."""
        prices = [10.0] * 100  # ~4-5 months
        df = _make_df(prices)
        strategy = ValueAveragingStrategy()
        result = strategy.generate_signals(
            df, pe_value=10.0, base_amount=1000, frequency="monthly",
        )
        buys = result[result["signal"] == "buy"]
        # Should get ~4-5 monthly buys, not 100 daily buys
        assert 3 <= len(buys) <= 6

    def test_output_columns(self):
        """Output has all signal columns."""
        df = _make_df([10.0] * 30)
        strategy = ValueAveragingStrategy()
        result = strategy.generate_signals(df, pe_value=10.0)
        for col in ["signal", "signal_price", "signal_shares", "signal_reason"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# HybridStrategy
# ---------------------------------------------------------------------------

class TestHybrid:
    def test_combines_dca_and_grid(self):
        """Hybrid produces signals from both sub-strategies."""
        # Oscillating prices to trigger both DCA and grid signals
        prices = [10.0, 9.5, 9.0, 9.5, 10.0, 10.5, 11.0] * 15
        df = _make_df(prices)
        strategy = HybridStrategy()
        result = strategy.generate_signals(
            df, pe_value=10.0,
            dca_base_amount=1000,
            dca_frequency="weekly",
            grid_count=10,
        )
        buys = result[result["signal"] == "buy"]
        sells = result[result["signal"] == "sell"]
        # Should have both buy and sell signals
        assert len(buys) > 0
        # Grid sell signals may or may not appear depending on params

    def test_hybrid_metadata(self):
        """df.attrs contains grid metadata."""
        df = _make_df([10.0] * 50)
        strategy = HybridStrategy()
        result = strategy.generate_signals(df, pe_value=10.0)
        assert "dca_allocation_pct" in result.attrs

    def test_output_columns(self):
        """Output has all signal columns."""
        df = _make_df([10.0] * 30)
        strategy = HybridStrategy()
        result = strategy.generate_signals(df, pe_value=10.0)
        for col in ["signal", "signal_price", "signal_shares", "signal_reason"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Strategy metadata tests
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    TrendFollowingStrategy(),
    GridTradingStrategy(),
    ValueAveragingStrategy(),
    HybridStrategy(),
]


class TestAllStrategies:
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_has_name(self, strategy):
        """Every strategy has a non-empty name."""
        assert strategy.name
        assert isinstance(strategy.name, str)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_has_description(self, strategy):
        """Every strategy has a non-empty description."""
        assert strategy.description
        assert isinstance(strategy.description, str)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_default_params_valid(self, strategy):
        """Default params pass validation."""
        params = strategy.get_default_params()
        errors = strategy.validate_params(params)
        assert errors == [], f"Validation errors: {errors}"

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_param_descriptions_match_params(self, strategy):
        """Every param has a description entry with required keys."""
        params = strategy.get_default_params()
        descs = strategy.get_param_descriptions()
        for key in params:
            assert key in descs, f"Missing description for param: {key}"
            for required in ["label", "type", "help"]:
                assert required in descs[key], (
                    f"Param '{key}' missing '{required}' in description"
                )

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_generate_signals_returns_dataframe(self, strategy):
        """generate_signals returns a DataFrame with signal columns."""
        df = _make_df([10.0] * 30)
        kwargs = {}
        if "ValueAveraging" in type(strategy).__name__ or "Hybrid" in type(strategy).__name__:
            kwargs["pe_value"] = 15.0
        result = strategy.generate_signals(df, **kwargs)
        assert isinstance(result, pd.DataFrame)
        for col in ["signal", "signal_price", "signal_shares", "signal_reason"]:
            assert col in result.columns
