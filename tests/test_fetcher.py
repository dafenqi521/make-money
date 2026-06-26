"""Tests for ETF data fetcher module."""

import pytest
import pandas as pd
import requests as requests_lib
from unittest.mock import patch, Mock

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
    """fetch_etf_hist returns a DataFrame with OHLCV + computed columns."""
    df = fetch_etf_hist(symbol="510300")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {
        "date", "open", "high", "low", "close", "volume",
        "change_pct", "amplitude", "ma5", "ma10", "ma20",
    }
    assert expected_cols.issubset(set(df.columns))


def test_fetch_etf_hist_date_filter():
    """fetch_etf_hist filters by start_date and end_date."""
    df = fetch_etf_hist(
        symbol="510300",
        start_date="20260601",
        end_date="20260626",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert (df["date"] >= pd.Timestamp("2026-06-01")).all()
    assert (df["date"] <= pd.Timestamp("2026-06-26")).all()


def test_fetch_etf_hist_empty_symbol():
    """fetch_etf_hist with empty symbol raises ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_hist(symbol="")


def test_fetch_etf_hist_invalid_symbol_raises():
    """fetch_etf_hist with a non-existent symbol raises ValueError."""
    with pytest.raises(ValueError):
        fetch_etf_hist(symbol="999999")


def test_fetch_etf_hist_with_prefixed_symbol():
    """fetch_etf_hist works with already-prefixed symbols."""
    df = fetch_etf_hist(symbol="sh510300")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_fetch_etf_hist_shenzhen_etf():
    """fetch_etf_hist works with Shenzhen ETFs."""
    df = fetch_etf_hist(symbol="159915")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "close" in df.columns
    assert "volume" in df.columns


def test_fetch_etf_hist_has_ma_columns():
    """Moving averages are computed when enough data exists."""
    df = fetch_etf_hist(symbol="510300")
    # At least some rows should have valid MA values
    assert df["ma20"].notna().any()
    assert df["change_pct"].notna().any()


# ---------------------------------------------------------------------------
# fetch_etf_info (Sina real-time API)
# ---------------------------------------------------------------------------

SINA_MOCK_RESPONSE = (
    'var hq_str_sh510300="沪深300ETF,5.008,5.048,4.907,5.015,'
    '4.880,4.906,6400,4.905,315200,4.904,240700,4.903,488596,'
    '4.902,226100,4.907,11400,4.908,47300,4.909,78500,4.910,75900,'
    '4.911,49300,2026-06-26,15:00:03,00";'
)


def test_fetch_etf_info_returns_full_dict():
    """fetch_etf_info returns a dict with all expected keys."""
    mock_resp = Mock()
    mock_resp.text = SINA_MOCK_RESPONSE
    mock_resp.encoding = "gbk"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        info = fetch_etf_info(symbol="510300")

    assert isinstance(info, dict)
    assert info["name"] == "沪深300ETF"
    assert info["current_price"] == 4.907
    assert info["prev_close"] == 5.048
    assert info["open"] == 5.008
    assert info["high"] == 5.015
    assert info["low"] == 4.880
    assert info["change"] == pytest.approx(-0.141, abs=0.01)
    assert info["change_pct"] == pytest.approx(-2.79, abs=0.1)
    assert info["date"] == "2026-06-26"
    assert info["time"] == "15:00:03"

    # Bid side
    assert info["bid1_price"] == 4.906
    assert info["bid1_volume"] == 6400
    assert info["bid5_price"] == 4.902
    assert info["bid5_volume"] == 226100

    # Ask side
    assert info["ask1_price"] == 4.907
    assert info["ask1_volume"] == 11400
    assert info["ask5_price"] == 4.911
    assert info["ask5_volume"] == 49300


def test_fetch_etf_info_empty_symbol():
    """fetch_etf_info with empty symbol raises ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_info(symbol="")


def test_fetch_etf_info_network_error_graceful():
    """fetch_etf_info returns graceful empty dict on network failure."""
    with patch.object(requests_lib, "get", side_effect=ConnectionError("timeout")):
        info = fetch_etf_info(symbol="510300")

    assert isinstance(info, dict)
    assert info["current_price"] is None
    assert info["bid1_price"] is None
    # Should still have the fallback name
    assert "510300" in info["name"]


def test_fetch_etf_info_invalid_response_graceful():
    """fetch_etf_info handles garbled Sina response gracefully."""
    mock_resp = Mock()
    mock_resp.text = "garbled nonsense without quotes"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        info = fetch_etf_info(symbol="510300")

    assert isinstance(info, dict)
    assert info["current_price"] is None


# ---------------------------------------------------------------------------
# get_available_etfs
# ---------------------------------------------------------------------------

def test_get_available_etfs_returns_dataframe():
    """get_available_etfs returns a non-empty DataFrame."""
    df = get_available_etfs()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_get_available_etfs_has_expected_columns():
    """get_available_etfs has code and name columns."""
    df = get_available_etfs()
    assert "代码" in df.columns
    assert "名称" in df.columns
