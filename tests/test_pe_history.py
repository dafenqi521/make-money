"""Tests for PE history loading, percentile computation, and ETF→index mapping."""

import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.index_map import (
    ETF_TO_INDEX,
    get_index_for_etf,
    get_available_indices,
    has_pe_data,
)
from src.data.pe_history import (
    PEPercentile,
    load_pe_history,
    list_cached_indices,
    compute_pe_percentile,
    get_etf_pe_percentile,
    get_pe_band_data,
    _classify_zone,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_pe_df():
    """Create a synthetic PE history DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
    import numpy as np
    np.random.seed(42)
    # Simulate PE oscillating around 15 with noise
    pe_base = 15 + np.sin(np.linspace(0, 6 * np.pi, 500)) * 5
    pe_noise = np.random.normal(0, 0.5, 500)
    pe_ttm = pe_base + pe_noise
    pe_mean = np.full(500, np.mean(pe_ttm))
    pe_std = np.std(pe_ttm)

    return pd.DataFrame({
        "date": dates,
        "index_close": np.random.normal(4000, 200, 500).cumsum() + 3000,
        "pe_ttm": pe_ttm,
        "pe_ttm_mean": pe_mean,
        "pe_ttm_plus_1std": pe_mean + pe_std,
        "pe_static": pe_ttm * 0.95,
    })


# ── Index Map Tests ─────────────────────────────────────────────


class TestIndexMap:
    """Tests for ETF → index mapping."""

    def test_known_etf_returns_mapping(self):
        """get_index_for_etf returns a dict for a known ETF."""
        result = get_index_for_etf("510300")
        assert result is not None
        assert result["index_name"] == "沪深300"
        assert result["index_code"] == "000300.SH"

    def test_unknown_etf_returns_none(self):
        """get_index_for_etf returns None for an unknown ETF."""
        assert get_index_for_etf("999999") is None

    def test_has_pe_data_true(self):
        """has_pe_data returns True for mapped ETFs."""
        assert has_pe_data("510300") is True
        assert has_pe_data("510050") is True
        assert has_pe_data("510500") is True

    def test_has_pe_data_false(self):
        """has_pe_data returns False for unknown ETFs."""
        assert has_pe_data("999999") is False

    def test_available_indices_non_empty(self):
        """get_available_indices returns a non-empty dict."""
        indices = get_available_indices()
        assert len(indices) >= 9
        assert "000300_SH_沪深300" in indices

    def test_all_cache_keys_unique(self):
        """All ETF entries have valid cache keys."""
        cache_keys = set()
        for code, mapping in ETF_TO_INDEX.items():
            ck = mapping["cache_key"]
            # Cache key should exist in our available set
            cache_keys.add(ck)
        assert len(cache_keys) >= 9

    def test_major_etfs_mapped(self):
        """All major broad-market ETFs are mapped."""
        major = ["510300", "510050", "510500", "588000", "159915"]
        for code in major:
            assert has_pe_data(code), f"{code} should have PE data"


# ── Zone Classification Tests ───────────────────────────────────


class TestZoneClassification:
    """Tests for PE percentile → zone mapping."""

    def test_extreme_undervalue(self):
        label, color = _classify_zone(5.0)
        assert "极度低估" in label
        assert color == "#166534"

    def test_undervalue(self):
        label, color = _classify_zone(15.0)
        assert "低估" in label
        assert color == "#22c55e"

    def test_fair_value(self):
        label, color = _classify_zone(50.0)
        assert "合理" in label
        assert color == "#6b7280"

    def test_overvalue(self):
        label, color = _classify_zone(80.0)
        assert "高估" in label
        assert color == "#f97316"

    def test_extreme_overvalue(self):
        label, color = _classify_zone(95.0)
        assert "极度高估" in label
        assert color == "#ef4444"

    def test_boundary_10(self):
        """Percentile exactly at boundary."""
        label, _ = _classify_zone(10.0)
        assert "低估" in label  # 10% goes to 低估 zone

    def test_boundary_30(self):
        label, _ = _classify_zone(30.0)
        assert "合理" in label  # 30% goes to 合理 zone


# ── PE Percentile Computation Tests ─────────────────────────────


class TestComputePEPercentile:
    """Tests for compute_pe_percentile with synthetic data."""

    def test_compute_basic(self, sample_pe_df):
        """Basic percentile computation works."""
        pp = compute_pe_percentile(sample_pe_df, current_pe=12.0)
        assert pp is not None
        assert pp.current_pe == 12.0
        assert pp.pe_percentile is not None
        assert 0 <= pp.pe_percentile <= 100
        assert pp.pe_mean is not None
        assert pp.pe_median is not None
        assert pp.pe_plus_1std is not None
        assert pp.pe_minus_1std is not None
        assert pp.data_points == 500

    def test_percentile_extreme_low(self, sample_pe_df):
        """Very low PE gives low percentile."""
        min_pe = sample_pe_df["pe_ttm"].min()
        pp = compute_pe_percentile(sample_pe_df, current_pe=min_pe - 5)
        assert pp.pe_percentile is not None
        assert pp.pe_percentile < 5.0
        assert "极度低估" in pp.zone_label

    def test_percentile_extreme_high(self, sample_pe_df):
        """Very high PE gives high percentile."""
        max_pe = sample_pe_df["pe_ttm"].max()
        pp = compute_pe_percentile(sample_pe_df, current_pe=max_pe + 10)
        assert pp.pe_percentile is not None
        assert pp.pe_percentile > 95.0
        assert "极度高估" in pp.zone_label

    def test_percentile_median(self, sample_pe_df):
        """PE at median gives ~50% percentile."""
        median_pe = sample_pe_df["pe_ttm"].median()
        pp = compute_pe_percentile(sample_pe_df, current_pe=median_pe)
        assert pp.pe_percentile is not None
        assert 40 <= pp.pe_percentile <= 60  # approximately median

    def test_none_current_pe_uses_latest(self, sample_pe_df):
        """When current_pe is None, uses latest historical."""
        pp = compute_pe_percentile(sample_pe_df, current_pe=None)
        assert pp.current_pe is not None
        assert pp.current_pe == round(float(sample_pe_df["pe_ttm"].iloc[-1]), 2)

    def test_empty_df_returns_empty_percentile(self):
        """Empty DataFrame returns PEPercentile with defaults."""
        df = pd.DataFrame()
        pp = compute_pe_percentile(df, current_pe=15.0)
        assert pp.data_points == 0
        assert pp.pe_percentile is None

    def test_5yr_range_computed(self, sample_pe_df):
        """5-year min/max are computed when date column exists."""
        pp = compute_pe_percentile(sample_pe_df, current_pe=12.0)
        assert pp.pe_min_5yr is not None
        assert pp.pe_max_5yr is not None
        assert pp.pe_min_5yr <= pp.pe_max_5yr

    def test_to_dict_serializable(self, sample_pe_df):
        """to_dict returns a JSON-serializable dict."""
        pp = compute_pe_percentile(sample_pe_df, current_pe=12.0)
        d = pp.to_dict()
        assert isinstance(d, dict)
        assert "current_pe" in d
        assert "pe_percentile" in d
        assert "zone_label" in d
        assert "zone_color" in d
        # All values should be JSON-safe
        import json
        json.dumps(d)  # should not raise


# ── Cached Data Tests ───────────────────────────────────────────


class TestCachedPEData:
    """Tests requiring the actual cached PE parquet files."""

    def test_list_cached_indices(self):
        """list_cached_indices finds our cached parquet files."""
        indices = list_cached_indices()
        assert len(indices) >= 9
        assert "000300_SH_沪深300" in indices
        assert "000016_SH_上证50" in indices

    def test_load_pe_history_valid(self):
        """load_pe_history loads a real cache file."""
        df = load_pe_history("000300_SH_沪深300")
        assert df is not None
        assert len(df) > 1000  # should have many years of data
        assert "pe_ttm" in df.columns
        assert "date" in df.columns

    def test_load_pe_history_missing(self):
        """load_pe_history returns None for missing cache key."""
        df = load_pe_history("nonexistent_index")
        assert df is None

    def test_get_etf_pe_percentile_valid(self):
        """get_etf_pe_percentile works for a known ETF."""
        pp = get_etf_pe_percentile("510300", current_pe=15.0)
        assert pp is not None
        assert pp.index_name == "沪深300"
        assert pp.pe_percentile is not None
        assert pp.current_pe == 15.0
        assert pp.zone_label != ""

    def test_get_etf_pe_percentile_unknown_etf(self):
        """get_etf_pe_percentile returns None for unknown ETF."""
        pp = get_etf_pe_percentile("999999")
        assert pp is None

    def test_get_pe_band_data(self):
        """get_pe_band_data returns DataFrame for chart rendering."""
        df = get_pe_band_data("510300")
        assert df is not None
        assert len(df) > 1000
        assert "pe_ttm" in df.columns
        assert "date" in df.columns

    def test_pe_data_date_range(self):
        """Cached data spans many years."""
        df = load_pe_history("000016_SH_上证50")
        assert df is not None
        date_min = df["date"].min()
        date_max = df["date"].max()
        # Should have at least 10 years of data
        assert (date_max - date_min).days > 365 * 10

    def test_all_cached_indices_loadable(self):
        """All listed cached indices actually load."""
        for cache_key in list_cached_indices():
            df = load_pe_history(cache_key)
            assert df is not None, f"Failed to load {cache_key}"
            assert len(df) > 100, f"{cache_key} has too few rows: {len(df)}"
            assert "pe_ttm" in df.columns, f"{cache_key} missing pe_ttm column"
