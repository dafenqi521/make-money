"""ETF data fetching via AKShare and Sina real-time APIs.

Provides functions to fetch ETF historical daily data, real-time quotes
with 5-level bid/ask depth, and the full list of available ETFs.

Exchange prefix auto-detection:
- Shanghai (6xx...)  -> "sh" prefix
- Shenzhen (0xx..., 1xx..., 3xx...) -> "sz" prefix
- Already-prefixed symbols (shXXXXXX, szXXXXXX) are used as-is.
"""

import os
import re
import warnings
import pandas as pd
import akshare as ak
import requests

# Disable system proxy — Windows proxy settings often interfere with
# AKShare's HTTP requests to Sina/Eastmoney. Must run before any HTTP call.
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

# Suppress the Python 3.8 upgrade nag from akshare
warnings.filterwarnings("ignore", message=".*Python.*3\\.9.*")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_prefix(symbol: str) -> str:
    """Detect and prepend the exchange prefix (sh/sz) for a bare ETF code."""
    symbol = symbol.strip().lower()

    if symbol.startswith("sh") or symbol.startswith("sz"):
        return symbol

    first_char = symbol[0]
    if first_char == "6":
        return f"sh{symbol}"
    if first_char in ("0", "1", "3"):
        return f"sz{symbol}"

    return f"sh{symbol}"


def _fetch_sina_realtime(symbol: str) -> dict:
    """Fetch real-time quote with 5-level bid/ask from Sina Finance.

    Args:
        symbol: Prefixed ETF code, e.g. "sh510300".

    Returns:
        dict with 30+ fields including bid/ask depth. Empty dict on failure.
    """
    prefixed = _detect_prefix(symbol)
    url = f"https://hq.sinajs.cn/list={prefixed}"
    headers = {"Referer": "https://finance.sina.com.cn"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        text = r.text
    except Exception:
        return {}

    # Parse: var hq_str_XX="field0,field1,field2,...";
    match = re.search(r'"([^"]*)"', text)
    if not match:
        return {}

    fields = match.group(1).split(",")
    if len(fields) < 28:
        return {}

    def _f(i):
        """Safely get float value from fields list."""
        try:
            return float(fields[i])
        except (ValueError, IndexError):
            return None

    def _i(i):
        """Safely get int value from fields list."""
        try:
            return int(fields[i])
        except (ValueError, IndexError):
            return None

    return {
        # Basic info
        "name": fields[0],
        "date": fields[26],
        "time": fields[27] if len(fields) > 27 else "",
        # Price
        "open": _f(1),
        "prev_close": _f(2),
        "current_price": _f(3),
        "high": _f(4),
        "low": _f(5),
        # Computed
        "change": round(_f(3) - _f(2), 4) if _f(3) and _f(2) else None,
        "change_pct": (
            round((_f(3) - _f(2)) / _f(2) * 100, 2)
            if _f(3) and _f(2) and _f(2) != 0
            else None
        ),
        "amplitude": (
            round((_f(4) - _f(5)) / _f(2) * 100, 2)
            if _f(4) and _f(5) and _f(2) and _f(2) != 0
            else None
        ),
        # Volume & turnover
        "volume": _i(8),       # 成交量 (shares)
        "amount": _f(9),       # 成交额 (yuan)
        # Bid side (买盘)
        "bid1_price": _f(6),   "bid1_volume": _i(7),
        "bid2_price": _f(8),   "bid2_volume": _i(9),
        "bid3_price": _f(10),  "bid3_volume": _i(11),
        "bid4_price": _f(12),  "bid4_volume": _i(13),
        "bid5_price": _f(14),  "bid5_volume": _i(15),
        # Ask side (卖盘)
        "ask1_price": _f(16),  "ask1_volume": _i(17),
        "ask2_price": _f(18),  "ask2_volume": _i(19),
        "ask3_price": _f(20),  "ask3_volume": _i(21),
        "ask4_price": _f(22),  "ask4_volume": _i(23),
        "ask5_price": _f(24),  "ask5_volume": _i(25),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_etf_hist(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch historical daily OHLCV data for a given ETF.

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
    prefixed = _detect_prefix(symbol)

    # --- Fetch from AKShare (Sina source) ---
    try:
        df = ak.fund_etf_hist_sina(symbol=prefixed)
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch history for '{prefixed}': {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data for '{symbol}'. Check the code is a valid ETF."
        )

    # --- Normalise column names ---
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data missing columns: {missing}. Available: {list(df.columns)}"
        )

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort ascending for correct computations, then descending at the end
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # --- Enrich with computed fields ---
    # Change % (day over day)
    df["change_pct"] = df["close"].pct_change() * 100
    df["change_pct"] = df["change_pct"].round(2)

    # Amplitude (振幅) = (high - low) / prev_close * 100
    df["prev_close"] = df["close"].shift(1)
    df["amplitude"] = ((df["high"] - df["low"]) / df["prev_close"]) * 100
    df["amplitude"] = df["amplitude"].round(2)

    # Moving averages
    df["ma5"] = df["close"].rolling(window=5).mean().round(4)
    df["ma10"] = df["close"].rolling(window=10).mean().round(4)
    df["ma20"] = df["close"].rolling(window=20).mean().round(4)

    # Drop helper column
    df = df.drop(columns=["prev_close"])

    # --- Date filtering ---
    if start_date:
        start = pd.to_datetime(start_date)
        df = df[df["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df["date"] <= end]

    # Final: most recent first
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def fetch_etf_info(symbol: str) -> dict:
    """Fetch real-time quote with 5-level bid/ask depth from Sina.

    Args:
        symbol: ETF ticker code, e.g. "510300".

    Returns:
        dict with 30+ keys: name, date, time, open, prev_close,
        current_price, high, low, change, change_pct, amplitude,
        volume, amount, and bid1-5 / ask1-5 price+volume pairs.
        Values are None when unavailable.

    Raises:
        ValueError: If symbol is empty.
    """
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()
    data = _fetch_sina_realtime(symbol)

    if not data:
        # Return graceful empty result so UI still renders
        return {
            "name": f"ETF {symbol}",
            "date": None, "time": None,
            "open": None, "prev_close": None,
            "current_price": None, "high": None, "low": None,
            "change": None, "change_pct": None, "amplitude": None,
            "volume": None, "amount": None,
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

    return data


def get_available_etfs() -> pd.DataFrame:
    """Get list of all available ETFs from the market.

    Returns:
        DataFrame with ETF codes and names.
    """
    try:
        df = ak.fund_etf_category_sina()
    except Exception:
        try:
            df = ak.fund_etf_spot_em()
        except Exception:
            return pd.DataFrame(columns=["代码", "名称"])

    return df
