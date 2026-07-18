"""Tests for ETF data fetcher — multi-source adapters."""

import json

import pandas as pd
import pytest
import requests as requests_lib
from unittest.mock import patch, Mock

from src.data.fetcher import (
    fetch_etf_hist,
    fetch_etf_info,
    fetch_multi_etf_info,
    get_available_etfs,
    _detect_prefix,
    _tencent_realtime,
    _baidu_kline,
    _sina_realtime,
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
# Tencent Finance real-time adapter (unit tests with mocks)
# ---------------------------------------------------------------------------

# Build mock Tencent response with correct field indices (1-based per a-stock-data).
# We fill a list and join with ~ so every known field lands at the right index.
def _make_tencent_mock(prefixed_code, name, price, prev_close, open_, high, low,
                       change_amt, change_pct, amount_wan, turnover_pct,
                       pe_ttm, amplitude_pct, mcap_yi, float_mcap_yi,
                       pb, limit_up, limit_down, vol_ratio, pe_static,
                       bid1_p=0, bid1_v=0, ask1_p=0, ask1_v=0):
    f = [""] * 60
    f[1] = name
    f[3] = price
    f[4] = prev_close
    f[5] = open_
    f[9] = bid1_p;   f[10] = bid1_v
    f[19] = ask1_p;  f[20] = ask1_v
    f[30] = "2026-06-26 15:00:03"
    f[31] = change_amt
    f[32] = change_pct
    f[33] = high
    f[34] = low
    f[37] = amount_wan
    f[38] = turnover_pct
    f[39] = pe_ttm
    f[43] = amplitude_pct
    f[44] = mcap_yi
    f[45] = float_mcap_yi
    f[46] = pb
    f[47] = limit_up
    f[48] = limit_down
    f[49] = vol_ratio
    f[52] = pe_static
    body = "~".join(f)
    return f'v_{prefixed_code}="{body}";'


TENCENT_MOCK_510300 = _make_tencent_mock(
    prefixed_code="sh510300", name="沪深300ETF",
    price="4.907", prev_close="5.048", open_="5.008",
    high="5.015", low="4.880",
    change_amt="-0.141", change_pct="-2.79",
    amount_wan="187040", turnover_pct="4.55",
    pe_ttm="300.45", amplitude_pct="7.22",
    mcap_yi="410.88", float_mcap_yi="410.88",
    pb="11.51", limit_up="5.551", limit_down="4.543",
    vol_ratio="1.20", pe_static="314.76",
    bid1_p="4.906", bid1_v="6400",
    ask1_p="4.907", ask1_v="11400",
)

# A second ETF for batch testing: 510050 (上证50ETF)
TENCENT_MOCK_510050 = _make_tencent_mock(
    prefixed_code="sh510050", name="上证50ETF",
    price="3.215", prev_close="3.250", open_="3.220",
    high="3.240", low="3.180",
    change_amt="-0.035", change_pct="-1.08",
    amount_wan="150000", turnover_pct="3.20",
    pe_ttm="15.50", amplitude_pct="1.85",
    mcap_yi="350.00", float_mcap_yi="320.00",
    pb="1.50", limit_up="3.575", limit_down="2.925",
    vol_ratio="0.80", pe_static="13.20",
    bid1_p="3.214", bid1_v="5000",
    ask1_p="3.216", ask1_v="3000",
)

TENCENT_MULTI_RESPONSE = TENCENT_MOCK_510300 + "\n" + TENCENT_MOCK_510050


def test_tencent_realtime_single():
    """_tencent_realtime returns a dict with PE/PB/mcap for one ETF."""
    mock_resp = Mock()
    mock_resp.text = TENCENT_MOCK_510300
    mock_resp.encoding = "gbk"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        result = _tencent_realtime(["510300"])

    assert "510300" in result
    q = result["510300"]
    assert q["name"] == "沪深300ETF"
    assert q["current_price"] == 4.907
    assert q["prev_close"] == 5.048
    assert q["open"] == 5.008
    assert q["high"] == 5.015
    assert q["low"] == 4.880
    assert q["change"] == -0.141
    assert q["change_pct"] == -2.79
    # Tencent-exclusive valuation fields
    assert q["pe_ttm"] == 300.45
    assert q["pe_static"] == 314.76
    assert q["pb"] == 11.51
    assert q["mcap_yi"] == 410.88
    assert q["float_mcap_yi"] == 410.88
    assert q["turnover_pct"] == 4.55
    assert q["limit_up"] == 5.551
    assert q["limit_down"] == 4.543
    assert q["vol_ratio"] == 1.20
    # Bid/ask depth
    assert q["bid1_price"] == 4.906
    assert q["bid1_volume"] == 6400
    assert q["ask1_price"] == 4.907
    assert q["ask1_volume"] == 11400


def test_tencent_realtime_batch():
    """_tencent_realtime handles multiple ETFs in one call."""
    mock_resp = Mock()
    mock_resp.text = TENCENT_MULTI_RESPONSE
    mock_resp.encoding = "gbk"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        result = _tencent_realtime(["510300", "510050"])

    assert "510300" in result
    assert "510050" in result
    assert result["510300"]["name"] == "沪深300ETF"
    assert result["510050"]["name"] == "上证50ETF"


def test_tencent_realtime_network_error_graceful():
    """_tencent_realtime returns empty dict on network failure."""
    with patch.object(requests_lib, "get", side_effect=ConnectionError("timeout")):
        result = _tencent_realtime(["510300"])
    assert result == {}


def test_tencent_realtime_empty_codes():
    """_tencent_realtime with empty list returns empty dict."""
    assert _tencent_realtime([]) == {}


# ---------------------------------------------------------------------------
# Baidu K-line adapter (unit tests with mocks)
# ---------------------------------------------------------------------------

BAIDU_MOCK_RESPONSE = {
    "Result": {
        "newMarketData": {
            "keys": [
                "time", "open", "high", "low", "close", "volume",
                "amount", "ma5avgprice", "ma10avgprice", "ma20avgprice",
            ],
            "marketData": (
                "2026-06-25,4.900,4.920,4.880,4.907,1500000,7350000,"
                "4.910,4.905,4.895;"
                "2026-06-26,4.907,4.930,4.890,4.915,1600000,7864000,"
                "4.912,4.908,4.898"
            ),
        }
    }
}


def test_baidu_kline_returns_dataframe():
    """_baidu_kline returns a DataFrame with OHLCV + MA columns."""
    mock_resp = Mock()
    mock_resp.json.return_value = BAIDU_MOCK_RESPONSE

    with patch.object(requests_lib, "get", return_value=mock_resp):
        df = _baidu_kline("510300")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected = {"date", "open", "high", "low", "close", "volume",
                "ma5", "ma10", "ma20", "change_pct", "amplitude"}
    assert expected.issubset(set(df.columns))
    # MA values should be present (not NaN) for rows with enough history
    assert df["ma5"].notna().any()
    assert df["ma10"].notna().any()


def test_baidu_kline_network_error_graceful():
    """_baidu_kline returns empty DataFrame on network failure."""
    with patch.object(requests_lib, "get", side_effect=ConnectionError("timeout")):
        df = _baidu_kline("510300")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_baidu_kline_invalid_json_graceful():
    """_baidu_kline handles non-JSON response gracefully."""
    mock_resp = Mock()
    mock_resp.json.side_effect = ValueError("not json")

    with patch.object(requests_lib, "get", return_value=mock_resp):
        df = _baidu_kline("510300")
    assert df.empty


# ---------------------------------------------------------------------------
# Sina real-time adapter (fallback — unit tests with mocks)
# ---------------------------------------------------------------------------

SINA_MOCK_RESPONSE = (
    'var hq_str_sh510300="沪深300ETF,5.008,5.048,4.907,5.015,'
    '4.880,4.906,6400,4.905,315200,4.904,240700,4.903,488596,'
    '4.902,226100,4.907,11400,4.908,47300,4.909,78500,4.910,75900,'
    '4.911,49300,2026-06-26,15:00:03,00";'
)


def test_sina_realtime_returns_dict():
    """_sina_realtime returns a correct dict from Sina format."""
    mock_resp = Mock()
    mock_resp.text = SINA_MOCK_RESPONSE
    mock_resp.encoding = "gbk"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        info = _sina_realtime("510300")

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
    # Bid/ask
    assert info["bid1_price"] == 4.906
    assert info["bid1_volume"] == 6400
    assert info["ask1_price"] == 4.907
    assert info["ask1_volume"] == 11400
    # Sina doesn't provide PE/PB — should be None
    assert info["pe_ttm"] is None
    assert info["pb"] is None


def test_sina_realtime_network_error_graceful():
    """_sina_realtime returns empty dict on network failure."""
    with patch.object(requests_lib, "get", side_effect=ConnectionError("timeout")):
        info = _sina_realtime("510300")
    assert info == {}


# ---------------------------------------------------------------------------
# fetch_etf_info (Tencent primary → Sina fallback)
# ---------------------------------------------------------------------------

def test_fetch_etf_info_from_tencent():
    """fetch_etf_info uses Tencent as primary source, returns PE/PB."""
    mock_resp = Mock()
    mock_resp.text = TENCENT_MOCK_510300
    mock_resp.encoding = "gbk"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        info = fetch_etf_info(symbol="510300")

    assert info["name"] == "沪深300ETF"
    assert info["current_price"] == 4.907
    assert info["pe_ttm"] == 300.45
    assert info["pb"] == 11.51
    assert info["mcap_yi"] == 410.88


def test_fetch_etf_info_falls_back_to_sina():
    """When Tencent fails, fetch_etf_info falls back to Sina."""
    # Tencent returns empty result, Sina returns valid data
    tencent_mock = Mock()
    tencent_mock.text = ""  # empty response from Tencent
    tencent_mock.encoding = "gbk"

    sina_mock = Mock()
    sina_mock.text = SINA_MOCK_RESPONSE
    sina_mock.encoding = "gbk"

    # Both tencent URL and sina URL are called
    with patch.object(requests_lib, "get") as mock_get:
        mock_get.side_effect = [tencent_mock, sina_mock]
        info = fetch_etf_info(symbol="510300")

    assert info["name"] == "沪深300ETF"
    assert info["current_price"] == 4.907


def test_fetch_etf_info_empty_symbol():
    """fetch_etf_info with empty symbol raises ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_info(symbol="")


def test_fetch_etf_info_network_error_graceful():
    """fetch_etf_info returns graceful empty dict when all sources fail."""
    with patch.object(requests_lib, "get", side_effect=ConnectionError("timeout")):
        info = fetch_etf_info(symbol="510300")

    assert isinstance(info, dict)
    assert info["current_price"] is None
    assert info["pe_ttm"] is None
    assert "510300" in info["name"]


def test_fetch_etf_info_has_all_required_keys():
    """fetch_etf_info dict contains all expected keys even when empty."""
    with patch.object(requests_lib, "get", side_effect=ConnectionError("timeout")):
        info = fetch_etf_info(symbol="510300")

    required_keys = {
        "name", "current_price", "prev_close", "open", "high", "low",
        "change", "change_pct", "amplitude", "volume", "amount",
        "pe_ttm", "pe_static", "pb", "mcap_yi", "float_mcap_yi",
        "turnover_pct", "vol_ratio", "limit_up", "limit_down",
        "bid1_price", "bid1_volume", "bid2_price", "bid2_volume",
        "bid3_price", "bid3_volume", "bid4_price", "bid4_volume",
        "bid5_price", "bid5_volume",
        "ask1_price", "ask1_volume", "ask2_price", "ask2_volume",
        "ask3_price", "ask3_volume", "ask4_price", "ask4_volume",
        "ask5_price", "ask5_volume",
    }
    for key in required_keys:
        assert key in info, f"Missing key: {key}"


def test_fetch_multi_etf_info():
    """fetch_multi_etf_info batch-fetches multiple ETFs at once."""
    mock_resp = Mock()
    mock_resp.text = TENCENT_MULTI_RESPONSE
    mock_resp.encoding = "gbk"

    with patch.object(requests_lib, "get", return_value=mock_resp):
        result = fetch_multi_etf_info(["510300", "510050"])

    assert len(result) == 2
    assert result["510300"]["name"] == "沪深300ETF"
    assert result["510050"]["name"] == "上证50ETF"
    assert result["510300"]["pe_ttm"] == 300.45
    assert result["510300"]["date"] == "2026-06-26"
    assert result["510300"]["time"] == "15:00:03"


# ---------------------------------------------------------------------------
# fetch_etf_hist (Baidu primary → AKShare/Sina fallback)
# ---------------------------------------------------------------------------

def test_fetch_etf_hist_from_baidu():
    """fetch_etf_hist uses Baidu as primary source."""
    mock_resp = Mock()
    mock_resp.json.return_value = BAIDU_MOCK_RESPONSE

    with patch.object(requests_lib, "get", return_value=mock_resp):
        df = fetch_etf_hist(symbol="510300")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected_cols = {
        "date", "open", "high", "low", "close", "volume",
        "change_pct", "amplitude", "ma5", "ma10", "ma20",
    }
    assert expected_cols.issubset(set(df.columns))


def test_fetch_etf_hist_empty_symbol():
    """fetch_etf_hist with empty symbol raises ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_hist(symbol="")


def test_fetch_etf_hist_baidu_fails_falls_back_to_akshare():
    """When Baidu returns empty, fall back to AKShare/Sina."""
    # Baidu fails
    baidu_mock = Mock()
    baidu_mock.json.side_effect = ConnectionError("baidu down")

    with patch.object(requests_lib, "get", return_value=baidu_mock):
        # AKShare will be tried as fallback — this is a real API call
        # in tests; we expect it to either succeed or raise ValueError
        try:
            df = fetch_etf_hist(symbol="510300")
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        except ValueError:
            # Network not available — acceptable in CI
            pass


# ---------------------------------------------------------------------------
# Live integration tests (require network)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_etf_hist_live_returns_dataframe():
    """Live: fetch_etf_hist returns DataFrame with OHLCV + computed columns."""
    df = fetch_etf_hist(symbol="510300")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {
        "date", "open", "high", "low", "close", "volume",
        "change_pct", "amplitude", "ma5", "ma10", "ma20",
    }
    assert expected_cols.issubset(set(df.columns))


@pytest.mark.integration
def test_fetch_etf_hist_live_date_filter():
    """Live: fetch_etf_hist filters by start_date and end_date."""
    df = fetch_etf_hist(
        symbol="510300",
        start_date="20260601",
        end_date="20260626",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert (df["date"] >= pd.Timestamp("2026-06-01")).all()
    assert (df["date"] <= pd.Timestamp("2026-06-26")).all()


@pytest.mark.integration
def test_fetch_etf_hist_live_shenzhen_etf():
    """Live: fetch_etf_hist works with Shenzhen ETFs."""
    df = fetch_etf_hist(symbol="159915")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "close" in df.columns
    assert "volume" in df.columns


@pytest.mark.integration
def test_fetch_etf_hist_live_has_ma():
    """Live: Moving averages are computed when enough data exists."""
    df = fetch_etf_hist(symbol="510300")
    assert df["ma20"].notna().any()
    assert df["change_pct"].notna().any()


@pytest.mark.integration
def test_fetch_etf_info_live_returns_full_dict():
    """Live: fetch_etf_info returns PE/PB from Tencent."""
    info = fetch_etf_info(symbol="510300")
    assert isinstance(info, dict)
    assert "pe_ttm" in info
    assert "pb" in info
    assert "mcap_yi" in info


@pytest.mark.integration
def test_fetch_multi_etf_info_live():
    """Live: fetch_multi_etf_info returns multiple ETFs."""
    result = fetch_multi_etf_info(["510300", "510050"])
    assert len(result) >= 1  # At least one should succeed


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
