"""ETF data fetching via AKShare.

Provides functions to fetch ETF historical daily data, real-time info,
and the full list of available ETFs from Chinese markets.

Exchange prefix auto-detection:
- Shanghai (6xx...)  -> "sh" prefix
- Shenzhen (0xx..., 1xx..., 3xx...) -> "sz" prefix
- Already-prefixed symbols (shXXXXXX, szXXXXXX) are used as-is.
"""

import pandas as pd
import akshare as ak


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _detect_prefix(symbol: str) -> str:
    """Detect and prepend the exchange prefix (sh/sz) for a bare ETF code.

    Args:
        symbol: ETF ticker code, e.g. "510300" or "sh510300".

    Returns:
        Prefixed symbol string, e.g. "sh510300".
    """
    symbol = symbol.strip().lower()

    # Already prefixed – return as-is
    if symbol.startswith("sh") or symbol.startswith("sz"):
        return symbol

    # Character-by-character auto-detection
    first_char = symbol[0]
    if first_char == "6":
        return f"sh{symbol}"
    if first_char in ("0", "1", "3"):
        return f"sz{symbol}"

    # Unknown exchange — try sh first as default
    return f"sh{symbol}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_etf_hist(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch historical daily data for a given ETF.

    Args:
        symbol: ETF ticker code, e.g. "510300" for 沪深300ETF.
                Exchange prefix (sh/sz) is auto-detected when omitted.
        start_date: Optional start date in "YYYYMMDD" format.
        end_date: Optional end date in "YYYYMMDD" format.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
        Sorted by date descending (most recent first).

    Raises:
        ValueError: If symbol is empty or no data could be fetched.
    """
    # --- Validation ---
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()
    prefixed = _detect_prefix(symbol)

    # --- Fetch ---
    try:
        df = ak.fund_etf_hist_sina(symbol=prefixed)
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch data for symbol '{prefixed}': {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for symbol '{symbol}' (prefixed: '{prefixed}'). "
            f"Check that the ETF code is valid."
        )

    # --- Normalise column names ---
    # fund_etf_hist_sina returns English columns: date, open, high, low,
    # close, volume.  The renaming loop below handles both English and
    # Chinese column names (older AKShare versions or different funcs).
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

    # --- Validate required columns ---
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the columns we care about
    df = df[required].copy()

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- Date filtering (AKShare does not support start/end natively) ---
    if start_date:
        start = pd.to_datetime(start_date)
        df = df[df["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df["date"] <= end]

    # Sort most-recent-first (even if empty — caller handles that)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def fetch_etf_info(symbol: str) -> dict:
    """Fetch basic info and latest quote for an ETF.

    Args:
        symbol: ETF ticker code, e.g. "510300".
                Exchange prefix is auto-detected when omitted.

    Returns:
        dict with keys: name, current_price, change_pct, volume.
        Values may be None if the symbol is not found.
    """
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()
    prefixed = _detect_prefix(symbol)

    try:
        df_spot = ak.fund_etf_spot_em()
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch ETF info for '{symbol}': {exc}"
        ) from exc

    # Match by exact code (codes in spot data already include prefix)
    row = df_spot[df_spot["代码"] == prefixed]
    if row.empty:
        # Try bare code match as fallback
        row = df_spot[df_spot["代码"] == symbol]

    if row.empty:
        return {
            "name": f"ETF {symbol}",
            "current_price": None,
            "change_pct": None,
            "volume": None,
        }

    row = row.iloc[0]
    return {
        "name": row.get("名称", f"ETF {symbol}"),
        "current_price": (
            float(row["最新价"])
            if "最新价" in row.index and pd.notna(row["最新价"])
            else None
        ),
        "change_pct": (
            float(row["涨跌幅"])
            if "涨跌幅" in row.index and pd.notna(row["涨跌幅"])
            else None
        ),
        "volume": (
            float(row["成交量"])
            if "成交量" in row.index and pd.notna(row["成交量"])
            else None
        ),
    }


def get_available_etfs() -> pd.DataFrame:
    """Get list of all available ETFs from the market.

    Returns:
        DataFrame with ETF codes and names.
    """
    try:
        df = ak.fund_etf_category_sina()
    except Exception:
        # Fallback: use spot data
        try:
            df = ak.fund_etf_spot_em()
        except Exception:
            # Last resort: empty DataFrame with expected columns
            return pd.DataFrame(columns=["代码", "名称"])

    # Return the full DataFrame – callers can pick columns they need.
    # The minimum expected columns are 代码 (code) and 名称 (name).
    return df
