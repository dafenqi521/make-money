"""Application service for scanning the single supported ETF strategy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable

import pandas as pd

from src.data.fetcher import fetch_etf_hist, fetch_multi_etf_info
from src.strategy.etf_rotation import (
    RotationConfig,
    classify_etf,
    rank_etfs,
    select_targets,
)


# The authoritative starter universe for the standalone dashboard. Users can
# replace it with their own six-digit exchange-traded ETF codes in the app.
DEFAULT_ETF_POOL: tuple[dict, ...] = (
    {"code": "510300", "name": "沪深300ETF", "category": "domestic_broad"},
    {"code": "510050", "name": "上证50ETF", "category": "domestic_broad"},
    {"code": "510500", "name": "中证500ETF", "category": "domestic_broad"},
    {"code": "512100", "name": "中证1000ETF", "category": "domestic_broad"},
    {"code": "159915", "name": "创业板ETF", "category": "domestic_broad"},
    {"code": "588000", "name": "科创50ETF", "category": "domestic_broad"},
    {"code": "510880", "name": "红利ETF", "category": "domestic_broad"},
    {"code": "512480", "name": "半导体ETF", "category": "domestic_sector"},
    {"code": "512880", "name": "证券ETF", "category": "domestic_sector"},
    {"code": "512660", "name": "军工ETF", "category": "domestic_sector"},
    {"code": "516160", "name": "新能源ETF", "category": "domestic_sector"},
    {"code": "512010", "name": "医药ETF", "category": "domestic_sector"},
    {"code": "512800", "name": "银行ETF", "category": "domestic_sector"},
    {"code": "513100", "name": "纳指ETF", "category": "overseas_equity"},
    {"code": "513500", "name": "标普500ETF", "category": "overseas_equity"},
    {"code": "159920", "name": "恒生ETF", "category": "overseas_equity"},
    {"code": "518880", "name": "黄金ETF", "category": "commodity"},
    {"code": "511010", "name": "国债ETF", "category": "bond"},
    {"code": "511260", "name": "十年国债ETF", "category": "bond"},
)


@dataclass
class RotationScanResult:
    """Result returned by a local ETF-pool scan."""

    rankings: pd.DataFrame
    targets: pd.DataFrame
    histories: dict[str, pd.DataFrame]
    errors: dict[str, str]
    as_of: date | None

    @property
    def scanned_count(self) -> int:
        return len(self.histories)

    @property
    def eligible_count(self) -> int:
        if self.rankings.empty:
            return 0
        return int(self.rankings["eligible"].sum())


def normalise_pool(pool: Iterable[dict | str]) -> list[dict]:
    """Validate, de-duplicate, and normalise user-supplied ETF entries."""

    result: list[dict] = []
    seen: set[str] = set()
    for item in pool:
        if isinstance(item, str):
            item = {"code": item}
        code = str(item.get("code", "")).strip()
        if len(code) != 6 or not code.isdigit() or code in seen:
            continue
        name = str(item.get("name") or f"ETF {code}")
        result.append(
            {
                "code": code,
                "name": name,
                "category": str(item.get("category") or classify_etf(name)),
            }
        )
        seen.add(code)
    return result


def _listed_days(history: pd.DataFrame) -> int:
    if history.empty or "date" not in history.columns:
        return 0
    dates = pd.to_datetime(history["date"], errors="coerce").dropna()
    if dates.empty:
        return 0
    return max(0, (dates.max().date() - dates.min().date()).days)


def scan_etf_pool(
    pool: Iterable[dict | str] = DEFAULT_ETF_POOL,
    config: RotationConfig | None = None,
    history_fetcher: Callable[[str], pd.DataFrame] = fetch_etf_hist,
    quote_fetcher: Callable[[list[str]], dict[str, dict]] = fetch_multi_etf_info,
    max_workers: int = 8,
) -> RotationScanResult:
    """Fetch, rank, and allocate a local ETF candidate pool.

    Network failures are explicit in ``errors``.  Failed symbols are never
    silently treated as weak candidates or converted into buy signals.
    """

    config = config or RotationConfig()
    entries = normalise_pool(pool)
    histories: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    if not entries:
        return RotationScanResult(pd.DataFrame(), pd.DataFrame(), {}, {}, None)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(entries)))) as executor:
        future_map = {
            executor.submit(history_fetcher, entry["code"]): entry["code"]
            for entry in entries
        }
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                history = future.result()
                if history is None or history.empty:
                    raise ValueError("历史行情为空")
                histories[code] = history.copy()
            except Exception as error:
                errors[code] = str(error)

    codes = [entry["code"] for entry in entries]
    try:
        quotes = quote_fetcher(codes) or {}
    except Exception:
        quotes = {}

    metadata = {}
    for entry in entries:
        code = entry["code"]
        quote_name = str((quotes.get(code) or {}).get("name") or "").strip()
        name = quote_name or entry["name"]
        category = entry["category"]
        if category == "other" or entry["name"].startswith("ETF "):
            category = classify_etf(name)
        metadata[code] = {
            "name": name,
            "category": category,
            "listed_days": _listed_days(histories[code]) if code in histories else 0,
        }

    rankings = rank_etfs(histories, metadata, config) if histories else pd.DataFrame()
    targets = select_targets(rankings, histories, config) if not rankings.empty else pd.DataFrame()

    all_dates = []
    for history in histories.values():
        if "date" in history.columns:
            all_dates.extend(pd.to_datetime(history["date"], errors="coerce").dropna().tolist())
    as_of = max(all_dates).date() if all_dates else None
    return RotationScanResult(rankings, targets, histories, errors, as_of)
