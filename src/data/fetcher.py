"""ETF data fetching via multi-source adapters.

Data source priority (from a-stock-data v3.2.4):
  1. Tencent Finance (qt.gtimg.cn) — real-time + PE/PB/mcap, no IP block
  2. Baidu Gushitong — daily K-line with built-in MA5/10/20, no IP block
  3. mootdx (通达信 TCP 7709) — K-line + 5-level bid/ask, no IP block
  4. Sina (hq.sinajs.cn) — real-time fallback, low risk
  5. AKShare/Eastmoney — ETF list only, rate-limited

Exchange prefix auto-detection:
  - Shanghai (6xx...)  -> "sh" prefix
  - Shenzhen (0xx..., 1xx..., 3xx...) -> "sz" prefix
  - Already-prefixed symbols (shXXXXXX, szXXXXXX) are used as-is.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Environment & compat
# ---------------------------------------------------------------------------

# Disable system proxy — Windows proxy settings often interfere with HTTP requests.
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

# Suppress the Python 3.8 upgrade nag from akshare
warnings.filterwarnings("ignore", message=".*Python.*3\\.9.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TENCENT_URL = "https://qt.gtimg.cn/q="
_BAIDU_KLINE_URL = "https://finance.pae.baidu.com/selfselect/getstockquotation"
_SINA_URL = "https://hq.sinajs.cn/list="

# Tencent field index — calibrated 2026-05 per a-stock-data v3.2.4
# fmt: off
_T_IDX = {
    "name": 1,          "price": 3,        "last_close": 4,    "open": 5,
    "high": 33,         "low": 34,         "change_amt": 31,   "change_pct": 32,
    "amount_wan": 37,   "turnover_pct": 38,"pe_ttm": 39,       "amplitude_pct": 43,
    "mcap_yi": 44,      "float_mcap_yi": 45, "pb": 46,         "limit_up": 47,
    "limit_down": 48,   "vol_ratio": 49,   "pe_static": 52,
    # bid 1-5 prices:  9,11,13,15,17    bid 1-5 volumes: 10,12,14,16,18
    # ask 1-5 prices: 19,21,23,25,27    ask 1-5 volumes: 20,22,24,26,28
}
# fmt: on

# Eastmoney rate-limiting globals (used by _em_get if needed)
_EM_SESSION: Optional[requests.Session] = None
_EM_LAST_CALL = 0.0
_EM_MIN_INTERVAL = 1.2  # seconds between calls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_prefix(symbol: str) -> str:
    """Detect and prepend the exchange prefix (sh/sz/bj) for a bare ETF code."""
    symbol = symbol.strip().lower()

    if symbol.startswith(("sh", "sz", "bj")):
        return symbol

    first_char = symbol[0]
    if first_char == "6":
        return f"sh{symbol}"
    if first_char in ("0", "1", "3"):
        return f"sz{symbol}"
    if first_char in ("8", "9"):
        return f"bj{symbol}"

    return f"sh{symbol}"


def _em_session() -> requests.Session:
    """Lazy-init a requests.Session for Eastmoney (connection reuse)."""
    global _EM_SESSION
    if _EM_SESSION is None:
        _EM_SESSION = requests.Session()
        _EM_SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        })
    return _EM_SESSION


def _em_get(url: str, params: dict | None = None, headers: dict | None = None,
            timeout: int = 15, **kwargs) -> requests.Response:
    """Eastmoney unified rate-limited GET. Use for all eastmoney.com endpoints."""
    import random
    global _EM_LAST_CALL
    wait = _EM_MIN_INTERVAL - (time.time() - _EM_LAST_CALL)
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return _em_session().get(url, params=params, headers=headers,
                                 timeout=timeout, **kwargs)
    finally:
        _EM_LAST_CALL = time.time()


# ---------------------------------------------------------------------------
# Adapter 1: Tencent Finance  (qt.gtimg.cn)
# ---------------------------------------------------------------------------

def _tencent_realtime(codes: list[str]) -> dict[str, dict]:
    """Batch-fetch real-time quotes from Tencent Finance.

    Args:
        codes: List of bare ETF codes, e.g. ["510300", "159915"].

    Returns:
        {code: {name, price, pe_ttm, pb, mcap_yi, ...}} — 25+ fields per code.
        Codes with no data are omitted from the result.

    No IP blocking — safe for frequent calls.
    """
    if not codes:
        return {}

    prefixed = [_detect_prefix(c) for c in codes]
    url = _TENCENT_URL + ",".join(prefixed)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.qq.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        data = r.text
    except Exception:
        return {}

    result: dict[str, dict] = {}
    for line in data.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        # Line format: v_sh510300="field0~field1~...";
        key_part = line.split("=")[0]  # v_sh510300
        bare = key_part.split("_")[-1]  # sh510300
        code = bare[2:] if len(bare) >= 3 else bare  # strip sh/sz/bj prefix
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue

        def _f(idx):
            try:
                v = vals[idx]
                return float(v) if v else None
            except (ValueError, IndexError):
                return None

        def _i(idx):
            try:
                v = vals[idx]
                return int(v) if v else None
            except (ValueError, IndexError):
                return None

        result[code] = {
            # Basic
            "name": vals[_T_IDX["name"]],
            "current_price": _f(_T_IDX["price"]),
            "prev_close": _f(_T_IDX["last_close"]),
            "open": _f(_T_IDX["open"]),
            "high": _f(_T_IDX["high"]),
            "low": _f(_T_IDX["low"]),
            # Change
            "change": _f(_T_IDX["change_amt"]),
            "change_pct": _f(_T_IDX["change_pct"]),
            # Volume / turnover
            "amount": (
                _f(_T_IDX["amount_wan"]) * 10000
                if _f(_T_IDX["amount_wan"])
                else None
            ),
            "turnover_pct": _f(_T_IDX["turnover_pct"]),
            "vol_ratio": _f(_T_IDX["vol_ratio"]),
            # Valuation (Tencent exclusive — not in Sina)
            "pe_ttm": _f(_T_IDX["pe_ttm"]),
            "pe_static": _f(_T_IDX["pe_static"]),
            "pb": _f(_T_IDX["pb"]),
            "mcap_yi": _f(_T_IDX["mcap_yi"]),
            "float_mcap_yi": _f(_T_IDX["float_mcap_yi"]),
            # Price limits
            "limit_up": _f(_T_IDX["limit_up"]),
            "limit_down": _f(_T_IDX["limit_down"]),
            "amplitude": _f(_T_IDX["amplitude_pct"]),
            # 5-level bid/ask  (indices 9-28, pairs)
            "bid1_price": _f(9),   "bid1_volume": _i(10),
            "bid2_price": _f(11),  "bid2_volume": _i(12),
            "bid3_price": _f(13),  "bid3_volume": _i(14),
            "bid4_price": _f(15),  "bid4_volume": _i(16),
            "bid5_price": _f(17),  "bid5_volume": _i(18),
            "ask1_price": _f(19),  "ask1_volume": _i(20),
            "ask2_price": _f(21),  "ask2_volume": _i(22),
            "ask3_price": _f(23),  "ask3_volume": _i(24),
            "ask4_price": _f(25),  "ask4_volume": _i(26),
            "ask5_price": _f(27),  "ask5_volume": _i(28),
        }
        # volume from amount_wan / price if no direct volume field
        if result[code]["amount"] is None:
            result[code]["amount"] = (
                _f(_T_IDX["amount_wan"]) * 10000
                if _f(_T_IDX["amount_wan"])
                else None
            )

    return result


# ---------------------------------------------------------------------------
# Adapter 2: Baidu Gushitong K-line  (finance.pae.baidu.com)
# ---------------------------------------------------------------------------

def _baidu_kline(symbol: str, start_time: str = "",
                 ktype: str = "1") -> pd.DataFrame:
    """Fetch daily K-line from Baidu Gushitong with built-in MA5/10/20.

    Args:
        symbol: Bare ETF code, e.g. "510300".
        start_time: Optional start timestamp string (empty = all).
        ktype: "1" = daily, "4" = weekly, "8" = monthly.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume,
        amount, ma5, ma10, ma20, change_pct.  Empty DataFrame on failure.
    """
    prefixed = _detect_prefix(symbol)
    params = {
        "all": "1", "isIndex": "false", "isBk": "false",
        "isBlock": "false", "isFutures": "false", "isStock": "true",
        "newFormat": "1", "group": "quotation_kline_ab",
        "finClientType": "pc", "code": prefixed,
        "start_time": start_time, "ktype": ktype,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }

    try:
        r = requests.get(_BAIDU_KLINE_URL, params=params,
                         headers=headers, timeout=10)
        d = r.json()
    except Exception:
        return pd.DataFrame()

    result = d.get("Result", {})
    # Baidu may return Result as a list (no data) or dict with newMarketData
    if not isinstance(result, dict):
        return pd.DataFrame()
    md = result.get("newMarketData", {})
    if not isinstance(md, dict):
        return pd.DataFrame()
    keys = md.get("keys", [])
    rows_str = md.get("marketData", "")

    if not keys or not rows_str:
        return pd.DataFrame()

    # Map Baidu field names to our standard names
    field_map = {
        "time": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
        "amount": "amount", "ma5avgprice": "ma5",
        "ma10avgprice": "ma10", "ma20avgprice": "ma20",
    }

    rows = []
    for line in rows_str.split(";"):
        if not line.strip():
            continue
        vals = line.split(",")
        if len(vals) < len(keys):
            continue
        row = {}
        for i, k in enumerate(keys):
            k_lower = k.lower()
            mapped = field_map.get(k_lower, k_lower)
            try:
                val = vals[i]
                row[mapped] = float(val) if val else None
            except (ValueError, IndexError):
                row[mapped] = None
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Ensure required columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = None

    # Compute change_pct
    if "close" in df.columns:
        df.loc[:, "close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df["change_pct"] = df["close"].pct_change() * 100
        df["change_pct"] = df["change_pct"].round(2)

    # Amplitude
    if all(c in df.columns for c in ["high", "low"]):
        df["prev_close"] = df["close"].shift(1)
        df["amplitude"] = (
            (df["high"] - df["low"]) / df["prev_close"] * 100
        )
        df["amplitude"] = df["amplitude"].round(2)
        df = df.drop(columns=["prev_close"], errors="ignore")

    # Ensure MA columns are numeric
    for ma_col in ["ma5", "ma10", "ma20"]:
        if ma_col in df.columns:
            df[ma_col] = pd.to_numeric(df[ma_col], errors="coerce").round(4)

    # Drop amount if present (not part of our standard schema; volume is)
    df = df.drop(columns=["amount"], errors="ignore")

    # Most recent first
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Adapter 3: Sina real-time fallback  (hq.sinajs.cn)
# ---------------------------------------------------------------------------

def _sina_realtime(symbol: str) -> dict:
    """Fetch real-time quote with 5-level bid/ask from Sina Finance.

    Kept as fallback — Tencent is the primary source.

    Args:
        symbol: Prefixed ETF code, e.g. "sh510300".

    Returns:
        dict with 30+ fields. Empty dict on failure.
    """
    prefixed = _detect_prefix(symbol)
    url = f"{_SINA_URL}{prefixed}"
    headers = {"Referer": "https://finance.sina.com.cn"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        text = r.text
    except Exception:
        return {}

    match = re.search(r'"([^"]*)"', text)
    if not match:
        return {}

    fields = match.group(1).split(",")
    if len(fields) < 28:
        return {}

    def _f(i):
        try:
            return float(fields[i])
        except (ValueError, IndexError):
            return None

    def _i(i):
        try:
            return int(fields[i])
        except (ValueError, IndexError):
            return None

    p3 = _f(3)
    p2 = _f(2)

    return {
        "name": fields[0],
        "date": fields[26],
        "time": fields[27] if len(fields) > 27 else "",
        "open": _f(1),           "prev_close": _f(2),
        "current_price": p3,      "high": _f(4),         "low": _f(5),
        "change": round(p3 - p2, 4) if p3 and p2 else None,
        "change_pct": (
            round((p3 - p2) / p2 * 100, 2)
            if p3 and p2 and p2 != 0 else None
        ),
        "amplitude": (
            round((_f(4) - _f(5)) / p2 * 100, 2)
            if _f(4) and _f(5) and p2 and p2 != 0 else None
        ),
        "volume": _i(8),
        "amount": _f(9),
        # Bid
        "bid1_price": _f(6),   "bid1_volume": _i(7),
        "bid2_price": _f(8),   "bid2_volume": _i(9),
        "bid3_price": _f(10),  "bid3_volume": _i(11),
        "bid4_price": _f(12),  "bid4_volume": _i(13),
        "bid5_price": _f(14),  "bid5_volume": _i(15),
        # Ask
        "ask1_price": _f(16),  "ask1_volume": _i(17),
        "ask2_price": _f(18),  "ask2_volume": _i(19),
        "ask3_price": _f(20),  "ask3_volume": _i(21),
        "ask4_price": _f(22),  "ask4_volume": _i(23),
        "ask5_price": _f(24),  "ask5_volume": _i(25),
        # Tencent-only fields are absent here; set to None for schema compat
        "pe_ttm": None, "pe_static": None, "pb": None,
        "mcap_yi": None, "float_mcap_yi": None,
        "turnover_pct": None, "vol_ratio": None,
        "limit_up": None, "limit_down": None,
    }


# ---------------------------------------------------------------------------
# Adapter 4: AKShare/Sina historical (fallback)
# ---------------------------------------------------------------------------

def _akshare_hist(symbol: str, start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
    """Fetch historical daily data via AKShare (Sina source).

    Fallback when Baidu K-line is unavailable.
    """
    import akshare as ak

    prefixed = _detect_prefix(symbol)

    try:
        df = ak.fund_etf_hist_sina(symbol=prefixed)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalise column names
    chinese_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()
        if col_lower == "date" or "日期" in col_str:
            chinese_map[col] = "date"
        elif col_lower == "open" or "开盘" in col_str:
            chinese_map[col] = "open"
        elif col_lower == "high" or "最高" in col_str:
            chinese_map[col] = "high"
        elif col_lower == "low" or "最低" in col_str:
            chinese_map[col] = "low"
        elif col_lower == "close" or "收盘" in col_str:
            chinese_map[col] = "close"
        elif col_lower == "volume" or "成交" in col_str:
            chinese_map[col] = "volume"

    if chinese_map:
        df = df.rename(columns=chinese_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            return pd.DataFrame()

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # Computed fields
    df["change_pct"] = df["close"].pct_change() * 100
    df["change_pct"] = df["change_pct"].round(2)

    df["prev_close"] = df["close"].shift(1)
    df["amplitude"] = ((df["high"] - df["low"]) / df["prev_close"]) * 100
    df["amplitude"] = df["amplitude"].round(2)

    # Moving averages (locally computed — AKShare doesn't provide them)
    df["ma5"] = df["close"].rolling(window=5).mean().round(4)
    df["ma10"] = df["close"].rolling(window=10).mean().round(4)
    df["ma20"] = df["close"].rolling(window=20).mean().round(4)
    df = df.drop(columns=["prev_close"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df.sort_values("date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_etf_info(symbol: str) -> dict:
    """Fetch real-time ETF quote with PE/PB/valuation from Tencent Finance.

    Falls back to Sina if Tencent is unreachable.

    Args:
        symbol: ETF ticker code, e.g. "510300".

    Returns:
        dict with 35+ keys: name, date, time, open, prev_close, current_price,
        high, low, change, change_pct, amplitude, volume, amount,
        pe_ttm, pe_static, pb, mcap_yi, float_mcap_yi, turnover_pct,
        vol_ratio, limit_up, limit_down,
        and bid1-5 / ask1-5 price+volume pairs.
        Values are None when unavailable.

    Raises:
        ValueError: If symbol is empty.
    """
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()

    # --- Try Tencent (primary) ---
    data = _tencent_realtime([symbol])
    if symbol in data:
        result = data[symbol]
        # Tencent doesn't provide date/time in the same format; add placeholder
        result.setdefault("date", None)
        result.setdefault("time", None)
        # volume: Tencent doesn't have a direct volume field; keep None
        if "volume" not in result:
            result["volume"] = None
        return result

    # --- Fallback: Sina ---
    sina_data = _sina_realtime(symbol)
    if sina_data:
        return sina_data

    # --- Graceful empty result ---
    return {
        "name": f"ETF {symbol}",
        "date": None, "time": None,
        "open": None, "prev_close": None,
        "current_price": None, "high": None, "low": None,
        "change": None, "change_pct": None, "amplitude": None,
        "volume": None, "amount": None,
        "pe_ttm": None, "pe_static": None, "pb": None,
        "mcap_yi": None, "float_mcap_yi": None,
        "turnover_pct": None, "vol_ratio": None,
        "limit_up": None, "limit_down": None,
        "bid1_price": None, "bid1_volume": None,
        "bid2_price": None, "bid2_volume": None,
        "bid3_price": None, "bid3_volume": None,
        "bid4_price": None, "bid4_volume": None,
        "bid5_price": None, "bid5_volume": None,
        "ask1_price": None, "ask1_volume": None,
        "ask2_price": None, "ask2_volume": None,
        "ask3_price": None, "ask3_volume": None,
        "ask4_price": None, "ask4_volume": None,
        "ask5_price": None, "ask5_volume": None,
    }


def fetch_etf_hist(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch historical daily OHLCV data with MA5/10/20 and computed indicators.

    Tries Baidu Gushitong first (built-in MA, no IP block),
    falls back to AKShare/Sina.

    Args:
        symbol: ETF ticker code, e.g. "510300".
        start_date: Optional start date in "YYYYMMDD" format.
        end_date: Optional end date in "YYYYMMDD" format.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume,
        change_pct, amplitude, ma5, ma10, ma20. Sorted date descending.

    Raises:
        ValueError: If symbol is empty or no data could be fetched.
    """
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()

    # --- Try Baidu (primary) ---
    df = _baidu_kline(symbol)
    if not df.empty:
        # Filter by date
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df["date"] >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df["date"] <= end]
        if df.empty:
            raise ValueError(
                f"No data for '{symbol}' in the specified date range."
            )
        return df.sort_values("date", ascending=False).reset_index(drop=True)

    # --- Fallback: AKShare/Sina ---
    df = _akshare_hist(symbol, start_date=start_date, end_date=end_date)
    if df is not None and not df.empty:
        return df

    raise ValueError(
        f"No data for '{symbol}'. Check the code is a valid ETF "
        f"and that the network can reach Baidu/Sina APIs."
    )


def fetch_multi_etf_info(symbols: list[str]) -> dict[str, dict]:
    """Batch-fetch real-time quotes for multiple ETFs via Tencent Finance.

    Much more efficient than calling fetch_etf_info() N times —
    one HTTP request for all symbols.

    Args:
        symbols: List of ETF codes, e.g. ["510300", "510050", "159915"].

    Returns:
        {code: info_dict} for each successfully fetched ETF.
    """
    if not symbols:
        return {}
    return _tencent_realtime([s.strip() for s in symbols])


def get_available_etfs() -> pd.DataFrame:
    """Get list of all available ETFs from the market.

    Returns:
        DataFrame with ETF codes and names.
    """
    try:
        import akshare as ak
        df = ak.fund_etf_category_sina()
        return df
    except Exception:
        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            if "代码" in df.columns and "名称" in df.columns:
                return df[["代码", "名称"]].drop_duplicates(subset="代码")
            return df
        except Exception:
            return pd.DataFrame(columns=["代码", "名称"])
