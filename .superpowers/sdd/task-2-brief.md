### Task 2: ETF Data Fetcher Module

**Files:**
- Create: `src/data/fetcher.py`
- Create: `tests/test_fetcher.py`

**Interfaces:**
- Consumes: `akshare` (from Task 1)
- Produces:
  - `fetch_etf_hist(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame`
    - Returns DataFrame with columns: date, open, high, low, close, volume
  - `fetch_etf_info(symbol: str) -> dict`
    - Returns dict with keys: name, current_price, change_pct, volume
  - `get_available_etfs() -> pd.DataFrame`
    - Returns DataFrame of all available ETFs

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetcher.py`:

```python
import pytest
import pandas as pd
from src.data.fetcher import fetch_etf_hist, fetch_etf_info, get_available_etfs


def test_fetch_etf_hist_returns_dataframe():
    """fetch_etf_hist should return a DataFrame with expected columns."""
    df = fetch_etf_hist(symbol="510300")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {"date", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns))


def test_fetch_etf_hist_empty_symbol():
    """fetch_etf_hist with invalid symbol should raise ValueError."""
    with pytest.raises(ValueError, match="symbol"):
        fetch_etf_hist(symbol="")


def test_fetch_etf_info_returns_dict():
    """fetch_etf_info should return a dict with name and price info."""
    info = fetch_etf_info(symbol="510300")
    assert isinstance(info, dict)
    assert "name" in info
    assert "current_price" in info


def test_get_available_etfs_returns_dataframe():
    """get_available_etfs should return a non-empty DataFrame."""
    df = get_available_etfs()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_fetcher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.fetcher'`

- [ ] **Step 3: Write minimal implementation**

Create `src/data/fetcher.py`:

```python
"""ETF data fetching via AKShare.

Provides functions to fetch ETF historical daily data, real-time info,
and the full list of available ETFs from Chinese markets.
"""

import pandas as pd
import akshare as ak


def fetch_etf_hist(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical daily data for a given ETF.

    Args:
        symbol: ETF ticker code, e.g. "510300" for 沪深300ETF.
        start_date: Optional start date in "YYYYMMDD" format.
        end_date: Optional end date in "YYYYMMDD" format.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.

    Raises:
        ValueError: If symbol is empty or invalid.
    """
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()

    try:
        df = ak.fund_etf_hist_sina(symbol=symbol)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for symbol '{symbol}': {e}")

    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'")

    # Normalize column names — fund_etf_hist_sina returns Chinese column names
    column_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    # Try to detect and map Chinese columns
    chinese_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "日期" in str(col) or "date" in col_lower:
            chinese_map[col] = "date"
        elif "开盘" in str(col) or "open" in col_lower:
            chinese_map[col] = "open"
        elif "最高" in str(col) or "high" in col_lower:
            chinese_map[col] = "high"
        elif "最低" in str(col) or "low" in col_lower:
            chinese_map[col] = "low"
        elif "收盘" in str(col) or "close" in col_lower:
            chinese_map[col] = "close"
        elif "成交" in str(col) or "volume" in col_lower:
            chinese_map[col] = "volume"

    if chinese_map:
        df = df.rename(columns=chinese_map)

    # Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Select only required columns
    df = df[required].copy()

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by date range if provided
    if start_date:
        start = pd.to_datetime(start_date)
        df = df[df["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df["date"] <= end]

    # Sort by date descending (most recent first)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def fetch_etf_info(symbol: str) -> dict:
    """
    Fetch basic info and latest quote for an ETF.

    Args:
        symbol: ETF ticker code, e.g. "510300".

    Returns:
        dict with keys: name, current_price, change_pct, volume.

    Raises:
        ValueError: If symbol is empty or data unavailable.
    """
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    symbol = symbol.strip()

    try:
        df_spot = ak.fund_etf_spot_em()
    except Exception as e:
        raise ValueError(f"Failed to fetch ETF spot data: {e}")

    # Find the matching ETF
    row = df_spot[df_spot["代码"] == symbol]
    if row.empty:
        # Try with .SH or .SZ suffix if not found
        row = df_spot[
            (df_spot["代码"].str[:6] == symbol) | (df_spot["代码"] == symbol)
        ]

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
        "current_price": float(row.get("最新价", 0)) if pd.notna(row.get("最新价")) else None,
        "change_pct": float(row.get("涨跌幅", 0)) if pd.notna(row.get("涨跌幅")) else None,
        "volume": float(row.get("成交量", 0)) if pd.notna(row.get("成交量")) else None,
    }


def get_available_etfs() -> pd.DataFrame:
    """
    Get list of all available ETFs from the market.

    Returns:
        DataFrame with ETF codes and names.
    """
    try:
        df = ak.fund_etf_category_sina()
        # Keep only the most useful columns
        keep_cols = []
        for col in df.columns:
            col_str = str(col).lower()
            if any(k in col_str for k in ["代码", "名称", "code", "name", "symbol"]):
                keep_cols.append(col)
        if keep_cols:
            df = df[keep_cols]
        return df
    except Exception:
        # Fallback: return spot data as ETF list
        df = ak.fund_etf_spot_em()
        if "代码" in df.columns and "名称" in df.columns:
            return df[["代码", "名称"]].drop_duplicates(subset="代码")
        return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_fetcher.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add ETF data fetcher module with AKShare integration"
```

---

