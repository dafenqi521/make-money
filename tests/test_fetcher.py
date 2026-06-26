"""Tests for ETF data fetcher module."""

import pytest
import pandas as pd
from src.data.fetcher import (
    fetch_etf_hist,
    fetch_etf_info,
    get_available_etfs,
    _detect_prefix,
)


# ---------------------------------------------------------------------------
# Prefix auto-detection
# ---------------------------------------------------------------------------

def test_detect_prefix_shanghai():
    """6xx codes map to Shanghai prefix sh."""
    assert _detect_prefix("510300") == "sh510300"
    assert _detect_prefix("510050") == "sh510050"
    assert _detect_prefix("510500") == "sh510500"
    assert _detect_prefix("588000") == "sh588000"
    assert _detect_prefix("600000") == "sh600000"


def test_detect_prefix_shenzhen():
    """0xx/1xx/3xx codes map to Shenzhen prefix sz."""
    assert _detect_prefix("159915") == "sz159915"
    assert _detect_prefix("159919") == "sz159919"
    assert _detect_prefix("001234") == "sz001234"
    assert _detect_prefix("300750") == "sz300750"


def test_detect_prefix_already_prefixed():
    """Codes that already have a prefix are returned as-is."""
    assert _detect_prefix("sh510300") == "sh510300"
    assert _detect_prefix("sz159915") == "sz159915"


# ---------------------------------------------------------------------------
# fetch_etf_hist
# ---------------------------------------------------------------------------

def test_fetch_etf_hist_returns_dataframe():
    """fetch_etf_hist should return a DataFrame with expected columns."""
    df = fetch_etf_hist(symbol="510300")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {"date", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns))


def test_fetch_etf_hist_date_filter():
    """fetch_etf_hist should filter by start_date and end_date."""
    df = fetch_etf_hist(
        symbol="510300",
        start_date="20260601",
        end_date="20260626",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # All dates should be within the requested range
    assert (df["date"] >= pd.Timestamp("2026-06-01")).all()
    assert (df["date"] <= pd.Timestamp("2026-06-26")).all()


def test_fetch_etf_hist_empty_symbol():
    """fetch_etf_hist with empty symbol should raise ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_hist(symbol="")


def test_fetch_etf_hist_invalid_symbol_raises():
    """fetch_etf_hist with a non-existent symbol should raise ValueError."""
    with pytest.raises(ValueError):
        fetch_etf_hist(symbol="999999")


def test_fetch_etf_hist_with_prefixed_symbol():
    """fetch_etf_hist should work with already-prefixed symbols."""
    df = fetch_etf_hist(symbol="sh510300")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_fetch_etf_hist_shenzhen_etf():
    """fetch_etf_hist should work with Shenzhen ETFs."""
    df = fetch_etf_hist(symbol="159915")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {"date", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# fetch_etf_info
# ---------------------------------------------------------------------------

def test_fetch_etf_info_returns_dict():
    """fetch_etf_info should return a dict with expected keys."""
    import akshare as ak
    from unittest.mock import patch

    # Build a mock spot DataFrame that includes sh510300
    mock_df = pd.DataFrame({
        "代码": ["sh510300", "sz159915"],
        "名称": ["沪深300ETF", "创业板ETF"],
        "最新价": [3.850, 2.120],
        "涨跌幅": [0.52, -0.31],
        "成交量": [12345678, 9876543],
    })

    with patch.object(ak, "fund_etf_spot_em", return_value=mock_df):
        info = fetch_etf_info(symbol="510300")
        assert isinstance(info, dict)
        assert info["name"] == "沪深300ETF"
        assert info["current_price"] == 3.850
        assert info["change_pct"] == 0.52
        assert info["volume"] == 12345678


def test_fetch_etf_info_empty_symbol():
    """fetch_etf_info with empty symbol should raise ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_info(symbol="")


def test_fetch_etf_info_unknown_symbol_returns_graceful():
    """fetch_etf_info with unknown symbol should return dict with None values."""
    import akshare as ak
    from unittest.mock import patch

    # Mock spot DataFrame that does NOT contain 999999
    mock_df = pd.DataFrame({
        "代码": ["sh510300"],
        "名称": ["沪深300ETF"],
        "最新价": [3.850],
        "涨跌幅": [0.52],
        "成交量": [12345678],
    })

    with patch.object(ak, "fund_etf_spot_em", return_value=mock_df):
        info = fetch_etf_info(symbol="999999")
        assert isinstance(info, dict)
        assert "name" in info
        # Should still have name (fallback) but None for price data
        assert info["current_price"] is None


def test_fetch_etf_info_network_error_raises():
    """fetch_etf_info should raise ValueError when upstream fetch fails."""
    import akshare as ak
    from unittest.mock import patch

    with patch.object(ak, "fund_etf_spot_em", side_effect=ConnectionError("timeout")):
        with pytest.raises(ValueError, match="Failed to fetch ETF info"):
            fetch_etf_info(symbol="510300")


# ---------------------------------------------------------------------------
# get_available_etfs
# ---------------------------------------------------------------------------

def test_get_available_etfs_returns_dataframe():
    """get_available_etfs should return a non-empty DataFrame."""
    df = get_available_etfs()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_get_available_etfs_has_expected_columns():
    """get_available_etfs should have code and name columns."""
    df = get_available_etfs()
    assert "代码" in df.columns
    assert "名称" in df.columns
