"""Tests for multi-factor daily signal panel."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.signal_panel import (
    FactorSignal,
    DailySignal,
    compute_pe_factor,
    compute_ma_factor,
    compute_grid_factor,
    compute_daily_signal,
    _lighten,
)
from src.data.pe_history import PEPercentile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """25 days of OHLCV data with MA columns, mild upward trend."""
    dates = pd.date_range("2026-06-01", periods=25, freq="B")
    np.random.seed(42)
    base = 4.0 + np.cumsum(np.random.randn(25) * 0.03)
    close = np.clip(base, 3.0, 6.0)
    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(100000, 500000, 25),
        "ma5": close + np.random.randn(25) * 0.02,
        "ma10": close + np.random.randn(25) * 0.03,
        "ma20": close + np.random.randn(25) * 0.05,
        "change_pct": np.random.randn(25) * 0.5,
        "amplitude": np.random.uniform(0.5, 2.0, 25),
    })
    return df


@pytest.fixture
def sample_info():
    """Real-time quote dict mimicking fetch_etf_info output."""
    return {
        "name": "沪深300ETF",
        "current_price": 4.50,
        "pe_ttm": 14.5,
        "pe_static": 15.2,
    }


def _make_pp(pe: float, pct: float, **overrides) -> PEPercentile:
    """Build a PEPercentile with defaults filled in."""
    kwargs = dict(
        current_pe=pe,
        pe_percentile=pct,
        pe_mean=16.0,
        pe_median=15.5,
        pe_plus_1std=22.0,
        pe_minus_1std=10.0,
        pe_min_5yr=10.0,
        pe_max_5yr=28.0,
        data_points=3500,
        date_range="2005-01 ~ 2026-07",
        index_name="沪深300",
        zone_label="",
        zone_color="",
    )
    kwargs.update(overrides)
    return PEPercentile(**kwargs)


# ---------------------------------------------------------------------------
# FactorSignal dataclass
# ---------------------------------------------------------------------------


class TestFactorSignal:
    """Basic dataclass sanity."""

    def test_default_construction(self):
        f = FactorSignal(
            name="PE估值", signal="bullish", score=0.80,
            label="低估", detail="PE处于历史低位", icon="🟢",
            color="#16a34a", weight=0.40,
        )
        assert f.name == "PE估值"
        assert f.signal == "bullish"
        assert f.score == 0.80
        assert f.weight == 0.40

    def test_signal_values(self):
        """All signal values are valid."""
        valid = {"bullish", "bearish", "neutral", "no_data"}
        f = FactorSignal(
            name="测试", signal="bullish", score=0.50,
            label="", detail="", icon="", color="#000", weight=0.33,
        )
        assert f.signal in valid


class TestDailySignal:
    """DailySignal dataclass."""

    def test_default_construction(self):
        ds = DailySignal()
        assert ds.composite_score == 0.50
        assert ds.composite_action == "hold"
        assert ds.factors == []
        assert ds.steps == []

    def test_with_factors_and_steps(self):
        f1 = FactorSignal(
            name="PE估值", signal="bullish", score=0.80,
            label="低估", detail="", icon="🟢",
            color="#16a34a", weight=0.40,
        )
        ds = DailySignal(
            factors=[f1],
            composite_score=0.72,
            composite_action="buy",
            action_label="建议买入",
            steps=["买入2000股", "设置止损"],
        )
        assert len(ds.factors) == 1
        assert ds.composite_action == "buy"
        assert len(ds.steps) == 2


# ---------------------------------------------------------------------------
# PE factor computation
# ---------------------------------------------------------------------------


class TestPEFactor:
    """PE valuation factor signal computation."""

    # ── Historical percentile mode ──

    def test_extreme_undervalued(self):
        """PE at 12th percentile → strong bullish."""
        pp = _make_pp(11.5, 12.0, zone_label="低估区", zone_color="#16a34a")
        f = compute_pe_factor(pp, None)
        assert f.signal == "bullish"
        assert f.score >= 0.70
        assert "低估" in f.label

    def test_overvalued(self):
        """PE at 85th percentile → bearish."""
        pp = _make_pp(28.0, 85.0, zone_label="高估区", zone_color="#dc2626")
        f = compute_pe_factor(pp, None)
        assert f.signal == "bearish"
        assert f.score < 0.40
        assert "高估" in f.label

    def test_percentile_below_10_is_max_bullish(self):
        """PE < 10th percentile → highest score."""
        pp = _make_pp(9.0, 5.0, zone_label="极度低估", zone_color="#166534")
        f = compute_pe_factor(pp, None)
        assert f.score >= 0.85
        assert "极度低估" in f.label

    def test_percentile_above_90_is_max_bearish(self):
        """PE > 90th percentile → lowest score."""
        pp = _make_pp(35.0, 95.0, zone_label="极度高估", zone_color="#dc2626")
        f = compute_pe_factor(pp, None)
        assert f.score <= 0.15
        assert "极度高估" in f.label

    def test_percentile_mid_range_is_neutral(self):
        """PE near 50th percentile → neutral."""
        pp = _make_pp(16.0, 48.0, zone_label="合理区", zone_color="#2563eb")
        f = compute_pe_factor(pp, None)
        assert f.signal == "neutral"
        assert 0.40 <= f.score <= 0.60

    # ── Static PE threshold mode (fallback) ──

    def test_static_pe_low(self):
        """PE < 12 using snapshot → bullish."""
        f = compute_pe_factor(None, 10.0)
        assert f.signal == "bullish"
        assert f.score >= 0.70
        assert "低估" in f.label

    def test_static_pe_moderate(self):
        """PE 12-18 → neutral-bullish."""
        f = compute_pe_factor(None, 15.0)
        assert f.signal == "neutral"
        assert 0.50 < f.score < 0.65

    def test_static_pe_high(self):
        """PE > 35 → bearish."""
        f = compute_pe_factor(None, 40.0)
        assert f.signal == "bearish"
        assert f.score < 0.30
        assert "极度高估" in f.label

    # ── No data ──

    def test_no_pe_data(self):
        """No PE data at all → no_data signal."""
        f = compute_pe_factor(None, None)
        assert f.signal == "no_data"
        assert f.score == 0.50

    def test_pe_percentile_field_none(self):
        """PEPercentile with pe_percentile=None falls through to static PE."""
        pp = _make_pp(15.0, None)
        # Static PE 15.0 → neutral-bullish (12-18 range)
        f = compute_pe_factor(pp, pp.current_pe)
        assert f.signal != "no_data"


# ---------------------------------------------------------------------------
# MA factor computation
# ---------------------------------------------------------------------------


class TestMAFactor:
    """Moving average trend factor computation."""

    def test_bullish_alignment(self, sample_df):
        """MA5 > MA10 > MA20 → bullish."""
        df = sample_df.copy()
        # Last two rows: consistent bullish alignment (no cross)
        for idx in [df.index[-2], df.index[-1]]:
            df.loc[idx, "ma5"] = 5.2
            df.loc[idx, "ma10"] = 5.0
            df.loc[idx, "ma20"] = 4.8
        f = compute_ma_factor(df)
        assert f.signal == "bullish"
        assert f.score >= 0.60
        assert "多头排列" in f.label

    def test_bearish_alignment(self, sample_df):
        """MA5 < MA10 < MA20 → bearish (no cross, just alignment)."""
        df = sample_df.copy()
        # Last two rows: consistent bearish alignment, ma5_prev > ma20_prev
        # to avoid triggering death cross at prev→last transition
        for idx in [df.index[-2], df.index[-1]]:
            df.loc[idx, "ma5"] = 4.5
            df.loc[idx, "ma10"] = 4.7
            df.loc[idx, "ma20"] = 4.9
        f = compute_ma_factor(df)
        assert f.signal == "bearish"
        assert f.score < 0.40
        assert "空头排列" in f.label

    def test_golden_cross(self, sample_df):
        """MA5 crosses above MA20 → golden cross signal."""
        df = sample_df.copy()
        # Previous bar: ma5 < ma20
        df.loc[df.index[-2], "ma5"] = 4.7
        df.loc[df.index[-2], "ma10"] = 4.8
        df.loc[df.index[-2], "ma20"] = 4.8
        # Current bar: ma5 > ma20 (cross above)
        df.loc[df.index[-1], "ma5"] = 4.9
        df.loc[df.index[-1], "ma10"] = 4.85
        df.loc[df.index[-1], "ma20"] = 4.8
        f = compute_ma_factor(df)
        assert f.signal == "bullish"
        assert f.score >= 0.80
        assert "金叉" in f.label

    def test_death_cross(self, sample_df):
        """MA5 crosses below MA20 → death cross signal."""
        df = sample_df.copy()
        # Previous bar: ma5 > ma20
        df.loc[df.index[-2], "ma5"] = 4.9
        df.loc[df.index[-2], "ma10"] = 4.85
        df.loc[df.index[-2], "ma20"] = 4.8
        # Current bar: ma5 < ma20 (cross below)
        df.loc[df.index[-1], "ma5"] = 4.7
        df.loc[df.index[-1], "ma10"] = 4.75
        df.loc[df.index[-1], "ma20"] = 4.8
        f = compute_ma_factor(df)
        assert f.signal == "bearish"
        assert f.score <= 0.20
        assert "死叉" in f.label

    def test_no_ma_columns(self):
        """DataFrame without MA columns → no_data."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-06-01", periods=5, freq="B"),
            "close": [4.5, 4.6, 4.55, 4.7, 4.65],
        })
        f = compute_ma_factor(df)
        assert f.signal == "no_data"

    def test_empty_df(self):
        """Empty DataFrame → no_data."""
        f = compute_ma_factor(pd.DataFrame())
        assert f.signal == "no_data"

    def test_none_df(self):
        """None → no_data."""
        f = compute_ma_factor(None)
        assert f.signal == "no_data"

    def test_nan_ma_values(self, sample_df):
        """NaN MA values → no_data."""
        df = sample_df.copy()
        df.loc[df.index[-1], "ma5"] = np.nan
        df.loc[df.index[-1], "ma20"] = np.nan
        f = compute_ma_factor(df)
        assert f.signal == "no_data"

    def test_single_row_returns_no_data(self):
        """Single row: len < 2 → no_data (can't detect crosses)."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2026-06-01")],
            "close": [4.5],
            "ma5": [4.6],
            "ma10": [4.5],
            "ma20": [4.4],
        })
        f = compute_ma_factor(df)
        # Single row means len(df) < 2, so no_data
        assert f.signal == "no_data"

    def test_ma5_above_ma20_no_full_alignment(self, sample_df):
        """MA5 > MA20 but not full MA5>MA10>MA20 → neutral-bullish."""
        df = sample_df.copy()
        df.loc[df.index[-1], "ma5"] = 5.0
        df.loc[df.index[-1], "ma10"] = 4.6  # ma5 > ma20 but ma10 < ma5
        df.loc[df.index[-1], "ma20"] = 4.7  # not full alignment
        f = compute_ma_factor(df)
        # Should be neutral because MA5 > MA20 but not full bullish alignment
        assert f.signal in ("neutral", "bullish")
        assert f.score >= 0.50


# ---------------------------------------------------------------------------
# Grid factor computation
# ---------------------------------------------------------------------------


class TestGridFactor:
    """Grid position factor computation."""

    def test_price_at_bottom(self, sample_df):
        """Current price near historical low → bullish."""
        lo = float(sample_df["close"].min())
        f = compute_grid_factor(sample_df, current_price=lo + 0.05)
        assert f.signal == "bullish"
        assert f.score >= 0.70
        assert "低位" in f.label

    def test_price_at_middle(self, sample_df):
        """Current price near middle of range → neutral."""
        lo = float(sample_df["close"].min())
        hi = float(sample_df["close"].max())
        mid = (lo + hi) / 2.0
        f = compute_grid_factor(sample_df, current_price=mid)
        assert f.signal == "neutral"
        assert 0.40 <= f.score <= 0.60

    def test_price_at_top(self, sample_df):
        """Current price near historical high → bearish."""
        hi = float(sample_df["close"].max())
        f = compute_grid_factor(sample_df, current_price=hi - 0.05)
        assert f.signal in ("bearish", "neutral")
        assert f.score < 0.40

    def test_no_price_falls_back_to_last_close(self, sample_df):
        """When current_price is None, uses last close."""
        f = compute_grid_factor(sample_df, current_price=None)
        assert f.signal != "no_data"

    def test_short_history_no_data(self):
        """Less than 20 rows → no_data."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-06-01", periods=10, freq="B"),
            "close": [4.5 + i * 0.01 for i in range(10)],
        })
        f = compute_grid_factor(df, current_price=4.55)
        assert f.signal == "no_data"

    def test_empty_df(self):
        """Empty DataFrame → no_data."""
        f = compute_grid_factor(pd.DataFrame(), current_price=5.0)
        assert f.signal == "no_data"

    def test_none_df(self):
        """None → no_data."""
        f = compute_grid_factor(None, current_price=5.0)
        assert f.signal == "no_data"

    def test_flat_prices_no_data(self):
        """All same price → invalid range."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-06-01", periods=25, freq="B"),
            "close": [5.0] * 25,
        })
        f = compute_grid_factor(df, current_price=5.0)
        assert f.signal == "no_data"


# ---------------------------------------------------------------------------
# DailySignal composite computation
# ---------------------------------------------------------------------------


class TestDailySignalComputation:
    """End-to-end daily signal computation."""

    def test_bullish_composite(self, sample_df, sample_info):
        """When all factors bullish → strong buy."""
        pp = _make_pp(11.5, 12.0, zone_label="低估区", zone_color="#16a34a")
        df = sample_df.copy()
        # Force bullish MA (no cross)
        for idx in [df.index[-2], df.index[-1]]:
            df.loc[idx, "ma5"] = 5.2
            df.loc[idx, "ma10"] = 5.0
            df.loc[idx, "ma20"] = 4.8
        # Force low current price for grid
        info = {**sample_info, "current_price": float(df["close"].min()) + 0.05}
        ds = compute_daily_signal(df, info, pe_percentile=pp)
        assert ds.composite_score >= 0.60
        assert ds.composite_action in ("buy", "accumulate")
        assert len(ds.factors) == 3
        assert len(ds.steps) >= 1

    def test_bearish_composite(self, sample_df, sample_info):
        """When all factors bearish → strong sell."""
        pp = _make_pp(28.0, 85.0, zone_label="高估区", zone_color="#dc2626")
        df = sample_df.copy()
        # Force bearish MA (no cross, just alignment)
        for idx in [df.index[-2], df.index[-1]]:
            df.loc[idx, "ma5"] = 4.5
            df.loc[idx, "ma10"] = 4.7
            df.loc[idx, "ma20"] = 4.9
        # Force high current price for grid
        info = {**sample_info, "current_price": float(df["close"].max()) - 0.05}
        ds = compute_daily_signal(df, info, pe_percentile=pp)
        assert ds.composite_score < 0.45
        assert ds.composite_action in ("reduce", "sell", "hold")
        assert len(ds.steps) >= 1

    def test_mixed_signals_neutral(self, sample_df, sample_info):
        """Bullish PE + bearish MA → hold (mixed)."""
        pp = _make_pp(11.5, 12.0, zone_label="低估区", zone_color="#16a34a")
        df = sample_df.copy()
        # Force bearish MA
        for idx in [df.index[-2], df.index[-1]]:
            df.loc[idx, "ma5"] = 4.5
            df.loc[idx, "ma10"] = 4.7
            df.loc[idx, "ma20"] = 4.9
        # Middle price for grid
        lo = float(df["close"].min())
        hi = float(df["close"].max())
        mid = (lo + hi) / 2.0
        info = {**sample_info, "current_price": mid}
        ds = compute_daily_signal(df, info, pe_percentile=pp)
        assert ds.composite_action in ("hold", "accumulate")
        assert len(ds.factors) == 3

    def test_no_pe_data_still_works(self, sample_df, sample_info):
        """Without PE data, MA + Grid can still produce a signal."""
        info_no_pe = {**sample_info, "pe_ttm": None, "pe_static": None}
        ds = compute_daily_signal(sample_df, info_no_pe)
        assert ds.composite_score is not None
        assert len(ds.factors) == 3
        pe_factor = ds.factors[0]
        assert pe_factor.signal == "no_data"

    def test_all_factors_have_weights(self, sample_df, sample_info):
        """Every factor has a valid weight."""
        ds = compute_daily_signal(sample_df, sample_info)
        weights = [f.weight for f in ds.factors]
        assert sum(weights) == 1.0
        for w in weights:
            assert 0 < w < 1.0

    def test_composite_score_in_range(self, sample_df, sample_info):
        """Composite score always in [0, 1]."""
        ds = compute_daily_signal(sample_df, sample_info)
        assert 0.0 <= ds.composite_score <= 1.0

    def test_action_steps_non_empty(self, sample_df, sample_info):
        """Every action produces at least one step."""
        ds = compute_daily_signal(sample_df, sample_info)
        assert len(ds.steps) >= 1

    def test_current_price_preserved(self, sample_df, sample_info):
        """DailySignal carries the current_price through."""
        ds = compute_daily_signal(sample_df, sample_info)
        assert ds.current_price == sample_info["current_price"]

    def test_pe_value_preserved(self, sample_df, sample_info):
        """DailySignal carries the pe_value through."""
        ds = compute_daily_signal(sample_df, sample_info)
        assert ds.pe_value == sample_info["pe_ttm"]


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------


class TestLighten:
    """_lighten() utility."""

    def test_returns_hex_string(self):
        result = _lighten("#16a34a", 0.9)
        assert result.startswith("#")
        assert len(result) == 7

    def test_factor_1_returns_same_color(self):
        result = _lighten("#ff0000", 1.0)
        assert result == "#ff0000"

    def test_factor_0_returns_white(self):
        result = _lighten("#000000", 0.0)
        assert result == "#ffffff"

    def test_invalid_hex_returns_fallback(self):
        result = _lighten("#xyz", 0.9)
        assert result == "#f8fafc"

    def test_no_hash_prefix(self):
        result = _lighten("16a34a", 0.9)
        assert result.startswith("#")

    def test_green_lightened_is_lighter(self):
        """A lightened green is closer to white (higher RGB values)."""
        original = "#16a34a"
        light = _lighten(original, 0.9)
        orig_r = int(original[1:3], 16)
        light_r = int(light[1:3], 16)
        assert light_r > orig_r
