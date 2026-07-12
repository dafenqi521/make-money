"""PE history loader & percentile calculator.

Reads cached index PE history from local parquet files, computes
PE percentile (where does current PE sit in the historical distribution?),
and classifies valuation zones.

Data source: legulegu.com API → parquet cache (see fetch_pe_data.py).
Cache location: ``src/data/pe_cache/<cache_key>.parquet``.

Usage::

    from src.data.pe_history import get_etf_pe_percentile

    pp = get_etf_pe_percentile("510300", current_pe=12.5)
    if pp:
        print(f"PE分位: {pp.pe_percentile:.1f}%, 区间: {pp.zone_label}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.index_map import get_index_for_etf, has_pe_data

# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "pe_cache"


def _resolve_cache_path(cache_key: str) -> Path:
    """Return the full path to a cached PE history parquet file."""
    return _CACHE_DIR / f"{cache_key}.parquet"


# ---------------------------------------------------------------------------
# PEPercentile — computed result
# ---------------------------------------------------------------------------


@dataclass
class PEPercentile:
    """Summary of where current PE sits in the index's valuation history.

    All PE values are raw floats from the data source.  The "zone" fields
    use standard valuation thresholds based on historical percentile.
    """

    current_pe: float | None = None
    pe_percentile: float | None = None  # 0-100
    pe_mean: float | None = None        # historical average
    pe_median: float | None = None      # historical median (50th percentile)
    pe_plus_1std: float | None = None   # +1 standard deviation
    pe_minus_1std: float | None = None  # -1 standard deviation
    pe_min_5yr: float | None = None     # lowest PE in last 5 years
    pe_max_5yr: float | None = None     # highest PE in last 5 years
    data_points: int = 0                # number of historical observations
    date_range: str = ""                # e.g. "2005-04-08 → 2026-07-10"
    index_name: str = ""                # Chinese display name

    # ── Derived valuation zone ──
    zone_label: str = ""                # "极度低估" / "低估" / "合理" / "高估" / "极度高估"
    zone_color: str = ""                # CSS color

    def to_dict(self) -> dict:
        """Serialize to a plain dict for UI / session_state."""
        return {
            "current_pe": self.current_pe,
            "pe_percentile": (
                round(self.pe_percentile, 1) if self.pe_percentile is not None else None
            ),
            "pe_mean": round(self.pe_mean, 2) if self.pe_mean is not None else None,
            "pe_median": round(self.pe_median, 2) if self.pe_median is not None else None,
            "pe_plus_1std": round(self.pe_plus_1std, 2) if self.pe_plus_1std is not None else None,
            "pe_minus_1std": round(self.pe_minus_1std, 2) if self.pe_minus_1std is not None else None,
            "pe_min_5yr": round(self.pe_min_5yr, 2) if self.pe_min_5yr is not None else None,
            "pe_max_5yr": round(self.pe_max_5yr, 2) if self.pe_max_5yr is not None else None,
            "data_points": self.data_points,
            "date_range": self.date_range,
            "index_name": self.index_name,
            "zone_label": self.zone_label,
            "zone_color": self.zone_color,
        }


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------

# (label, color) keyed by percentile range
_ZONES: list[tuple[float, str, str]] = [
    (10, "极度低估", "#166534"),   # < 10%
    (30, "低估",     "#22c55e"),   # 10-30%
    (70, "合理",     "#6b7280"),   # 30-70%
    (90, "高估",     "#f97316"),   # 70-90%
    (float("inf"), "极度高估", "#ef4444"),  # > 90%
]


def _classify_zone(percentile: float) -> tuple[str, str]:
    """Map a PE percentile (0-100) to (zone_label, zone_color)."""
    for threshold, label, color in _ZONES:
        if percentile < threshold:
            return label, color
    return "极度高估", "#ef4444"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_pe_history(cache_key: str) -> pd.DataFrame | None:
    """Load cached PE history for a given cache key.

    Args:
        cache_key: e.g. "000300_SH_沪深300".

    Returns:
        DataFrame with columns [date, index_close, pe_ttm, pe_ttm_mean,
        pe_ttm_plus_1std, pe_static, ...] sorted date-ascending,
        or None if the cache file is missing / unreadable.
    """
    path = _resolve_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        # Ensure date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def list_cached_indices() -> dict[str, str]:
    """Return {cache_key: display_name} for all cached PE history files."""
    result: dict[str, str] = {}
    if not _CACHE_DIR.exists():
        return result
    for f in sorted(_CACHE_DIR.glob("*.parquet")):
        key = f.stem  # e.g. "000300_SH_沪深300"
        # Derive a display name from the filename
        parts = key.rsplit("_", 1)
        name = parts[1] if len(parts) > 1 else key
        result[key] = name
    return result


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------


def compute_pe_percentile(
    df: pd.DataFrame,
    current_pe: float | None,
) -> PEPercentile:
    """Compute PE percentile and valuation zone from historical data.

    Args:
        df: PE history DataFrame from :func:`load_pe_history`.
        current_pe: Current PE(TTM) value (from Tencent or other source).

    Returns:
        PEPercentile with all computed fields.  Zone fields are populated
        even when *current_pe* is None (using the latest historical PE).
    """
    pp = PEPercentile()

    # ── Basic metadata ──
    pp.data_points = len(df)
    if "date" in df.columns and len(df) > 0:
        dmin = df["date"].min()
        dmax = df["date"].max()
        if pd.notna(dmin) and pd.notna(dmax):
            pp.date_range = f"{dmin.strftime('%Y-%m-%d')} → {dmax.strftime('%Y-%m-%d')}"

    # ── PE series ──
    pe_col = "pe_ttm" if "pe_ttm" in df.columns else None
    if pe_col is None:
        return pp  # no data

    pe_series = df[pe_col].dropna()
    if len(pe_series) == 0:
        return pp

    # ── Sanity-check current_pe against historical range ──
    # Tencent and legulegu use different PE calculation methodologies,
    # so Tencent's PE may fall far outside legulegu's historical range.
    # When that happens, use the latest legulegu PE as the reference point.
    hist_min = float(pe_series.min())
    hist_max = float(pe_series.max())

    if current_pe is not None and current_pe > 0:
        # If current_pe is wildly outside historical range (less than 0.3x min
        # or more than 3x max), it's likely from a different methodology —
        # fall back to latest legulegu PE.
        if current_pe < hist_min * 0.3 or current_pe > hist_max * 3.0:
            pe_val = float(pe_series.iloc[-1])
        else:
            pe_val = float(current_pe)
    else:
        pe_val = float(pe_series.iloc[-1])

    pp.current_pe = round(pe_val, 2)

    # ── Statistics ──
    pp.pe_mean = round(float(pe_series.mean()), 2)
    pp.pe_median = round(float(pe_series.median()), 2)
    pp.pe_plus_1std = round(float(pe_series.mean() + pe_series.std()), 2)
    pp.pe_minus_1std = round(float(pe_series.mean() - pe_series.std()), 2)

    # ── Percentile ──
    pp.pe_percentile = round(
        (pe_series < pe_val).sum() / len(pe_series) * 100, 1
    )

    # ── 5-year extremes ──
    if "date" in df.columns:
        cutoff = date.today() - timedelta(days=365 * 5)
        mask_5yr = df["date"] >= pd.Timestamp(cutoff)
        pe_5yr = df.loc[mask_5yr, pe_col].dropna()
        if len(pe_5yr) > 0:
            pp.pe_min_5yr = round(float(pe_5yr.min()), 2)
            pp.pe_max_5yr = round(float(pe_5yr.max()), 2)

    # ── Zone ──
    pp.zone_label, pp.zone_color = _classify_zone(pp.pe_percentile)

    return pp


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------


def get_etf_pe_percentile(
    etf_code: str,
    current_pe: float | None = None,
) -> PEPercentile | None:
    """Load PE history for the index tracked by *etf_code* and compute percentile.

    This is the primary public API — call it from strategy live-signal
    methods and UI code.

    Args:
        etf_code: ETF ticker, e.g. "510300".
        current_pe: Current PE(TTM) from real-time quote.  If None,
            the latest historical PE is used.

    Returns:
        PEPercentile if cache data exists for this ETF, else None.
    """
    mapping = get_index_for_etf(etf_code)
    if mapping is None:
        return None

    cache_key = mapping["cache_key"]
    df = load_pe_history(cache_key)
    if df is None or df.empty:
        return None

    pp = compute_pe_percentile(df, current_pe)
    pp.index_name = mapping["index_name"]
    return pp


def get_pe_band_data(etf_code: str) -> pd.DataFrame | None:
    """Return PE history DataFrame suitable for PE Band chart rendering.

    Returns the full daily PE history with mean and ±1σ columns,
    or None if no cache data exists.
    """
    mapping = get_index_for_etf(etf_code)
    if mapping is None:
        return None

    df = load_pe_history(mapping["cache_key"])
    if df is None or df.empty:
        return None

    # Ensure we have the columns the chart needs
    if "pe_ttm" not in df.columns:
        return None

    return df


def refresh_pe_cache() -> dict[str, bool]:
    """Attempt to refresh all cached PE history files from legulegu API.

    Returns:
        {cache_key: success} dict.
    """
    try:
        import py_mini_racer
        import requests
        import akshare.stock_feature.stock_a_pe_and_pb as pe_mod
        from akshare.stock_feature.stock_a_indicator import get_cookie_csrf
    except ImportError:
        return {}

    js = py_mini_racer.MiniRacer()
    js.eval(pe_mod.hash_code)

    today_str = date.today().isoformat()
    token = js.call("hex", today_str).lower()

    TZ_CN = timezone(timedelta(hours=8))
    results: dict[str, bool] = {}
    cached = list_cached_indices()

    for cache_key in cached:
        # Extract index_code from cache key: "000300_SH_沪深300" → "000300.SH"
        parts = cache_key.split("_")
        if len(parts) >= 2:
            code = parts[0]
            exchange = parts[1] if parts[1] in ("SH", "SZ") else "SH"
            index_code = f"{code}.{exchange}"
        else:
            continue

        try:
            ck = get_cookie_csrf(url="https://legulegu.com/stockdata/sz50-ttm-lyr") or {}
            extra_headers = ck.pop('headers', {}) if 'headers' in ck else {}

            r = requests.get(
                "https://legulegu.com/api/stockdata/index-basic-pe",
                params={"token": token, "indexCode": index_code},
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Referer': 'https://legulegu.com/stockdata/sz50-ttm-lyr',
                    **extra_headers,
                },
                timeout=30,
                **ck,
            )

            if r.status_code == 200 and len(r.text) > 100:
                data = r.json()
                items = data.get("data", [])
                if items:
                    rows = []
                    for item in items:
                        date_val = item.get('date')
                        try:
                            if isinstance(date_val, str):
                                d = datetime.fromtimestamp(int(date_val) / 1000, tz=TZ_CN).date()
                            else:
                                d = datetime.fromtimestamp(date_val / 1000, tz=TZ_CN).date()
                        except (ValueError, TypeError):
                            continue
                        rows.append({
                            'date': d,
                            'index_close': float(item.get('close', 0)) if item.get('close') else None,
                            'pe_ttm': float(item.get('ttmPe', 0)) if item.get('ttmPe') else None,
                            'pe_ttm_mean': float(item.get('middleTtmPe', 0)) if item.get('middleTtmPe') else None,
                            'pe_ttm_plus_1std': float(item.get('addTtmPe', 0)) if item.get('addTtmPe') else None,
                            'pe_static': float(item.get('lyrPe', 0)) if item.get('lyrPe') else None,
                            'pe_static_mean': float(item.get('middleLyrPe', 0)) if item.get('middleLyrPe') else None,
                            'pe_static_plus_1std': float(item.get('addLyrPe', 0)) if item.get('addLyrPe') else None,
                        })

                    df_new = pd.DataFrame(rows)
                    df_new = df_new.sort_values('date').reset_index(drop=True)
                    path = _resolve_cache_path(cache_key)
                    df_new.to_parquet(path, index=False)
                    results[cache_key] = True
                else:
                    results[cache_key] = False
            else:
                results[cache_key] = False
        except Exception:
            results[cache_key] = False

    return results
