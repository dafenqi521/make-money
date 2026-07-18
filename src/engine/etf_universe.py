"""Automatic discovery and quality control for the exchange-traded ETF universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Mapping

import pandas as pd

from src.engine.trading_schedule import shanghai_now
from src.strategy.etf_rotation import RotationConfig, classify_etf


MIN_SOURCE_ROWS = 50
UNIVERSE_COLUMNS = [
    "code",
    "name",
    "exchange",
    "category",
    "price",
    "amount",
    "listed_date",
    "eligible",
    "exclusion_reason",
    "source",
    "refreshed_at",
]


@dataclass(frozen=True)
class UniverseRefreshResult:
    """Normalised universe plus source-level quality evidence."""

    entries: list[dict]
    source: str
    refreshed_at: str
    errors: dict[str, str]

    @property
    def total_count(self) -> int:
        return len(self.entries)

    @property
    def eligible_count(self) -> int:
        return sum(bool(row.get("eligible")) for row in self.entries)


def _column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    names = {str(value).strip().lower(): str(value) for value in frame.columns}
    for candidate in candidates:
        match = names.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _exchange(code: str) -> str:
    return "SZSE" if code.startswith(("15", "16")) else "SSE"


def _clean_code(value: object) -> str:
    code = str(value or "").strip().lower()
    if code.startswith(("sh", "sz")):
        code = code[2:]
    if code.endswith((".sh", ".sz")):
        code = code[:-3]
    return code.zfill(6) if code.isdigit() and len(code) < 6 else code


def normalise_universe_frame(
    frame: pd.DataFrame,
    source: str,
    refreshed_at: datetime | None = None,
    config: RotationConfig | None = None,
) -> list[dict]:
    """Normalise Eastmoney/Sina-like ETF spot tables into one stable grain."""

    if frame is None or frame.empty:
        raise ValueError("ETF目录为空")
    config = config or RotationConfig()
    code_col = _column(frame, ("代码", "基金代码", "symbol", "code"))
    name_col = _column(frame, ("名称", "基金简称", "name"))
    if code_col is None or name_col is None:
        raise ValueError("ETF目录缺少代码或名称列")
    price_col = _column(frame, ("最新价", "现价", "price", "trade"))
    amount_col = _column(frame, ("成交额", "amount", "turnover"))
    listed_col = _column(frame, ("上市日期", "list_date", "listed_date"))
    stamp = shanghai_now(refreshed_at).isoformat(timespec="seconds")
    excluded = tuple(value.upper() for value in config.excluded_name_keywords) + (
        "LOF",
        "封闭",
    )

    records: list[dict] = []
    seen: set[str] = set()
    for _, raw in frame.iterrows():
        code = _clean_code(raw.get(code_col))
        name = str(raw.get(name_col) or "").strip()
        if len(code) != 6 or not code.isdigit() or code in seen or not name:
            continue
        price = (
            pd.to_numeric(raw.get(price_col), errors="coerce")
            if price_col
            else None
        )
        amount = (
            pd.to_numeric(raw.get(amount_col), errors="coerce")
            if amount_col
            else None
        )
        listed = (
            pd.to_datetime(raw.get(listed_col), errors="coerce")
            if listed_col
            else None
        )
        reasons = []
        upper_name = name.upper()
        matched_keyword = next((word for word in excluded if word in upper_name), None)
        if matched_keyword:
            reasons.append(f"名称包含排除词：{matched_keyword}")
        if price_col and pd.isna(price):
            reasons.append("缺少有效交易价格")
        elif pd.notna(price) and float(price) <= 0:
            reasons.append("无有效交易价格")
        records.append(
            {
                "code": code,
                "name": name,
                "exchange": _exchange(code),
                "category": classify_etf(name),
                "price": float(price) if pd.notna(price) else None,
                "amount": float(amount) if pd.notna(amount) else None,
                "listed_date": listed.date().isoformat() if pd.notna(listed) else None,
                "eligible": not reasons,
                "exclusion_reason": "；".join(reasons),
                "source": source,
                "refreshed_at": stamp,
            }
        )
        seen.add(code)
    if len(records) < MIN_SOURCE_ROWS:
        raise ValueError(f"ETF目录仅{len(records)}条，低于质量门槛{MIN_SOURCE_ROWS}条")
    return records


def _eastmoney_fetcher() -> pd.DataFrame:
    import akshare as ak

    return ak.fund_etf_spot_em()


def _sina_fetcher() -> pd.DataFrame:
    import akshare as ak

    return ak.fund_etf_category_sina(symbol="ETF基金")


def discover_etf_universe(
    fetchers: Mapping[str, Callable[[], pd.DataFrame]] | None = None,
    refreshed_at: datetime | None = None,
    config: RotationConfig | None = None,
) -> UniverseRefreshResult:
    """Try independent sources in order and return the first valid catalogue."""

    source_fetchers = fetchers or {
        "东方财富/AKShare": _eastmoney_fetcher,
        "新浪财经/AKShare": _sina_fetcher,
    }
    errors: dict[str, str] = {}
    stamp = shanghai_now(refreshed_at)
    for source, fetcher in source_fetchers.items():
        try:
            entries = normalise_universe_frame(
                fetcher(), source, refreshed_at=stamp, config=config
            )
            return UniverseRefreshResult(
                entries=entries,
                source=source,
                refreshed_at=stamp.isoformat(timespec="seconds"),
                errors=errors,
            )
        except Exception as error:
            errors[source] = str(error)
    raise RuntimeError("；".join(f"{key}: {value}" for key, value in errors.items()))


def select_scan_pool(
    entries: Iterable[dict],
    minimum_spot_amount: float | None = None,
    max_count: int | None = None,
    always_include: Iterable[str] = (),
) -> list[dict]:
    """Select strategy scan inputs while preserving a full catalogue snapshot."""

    include_codes = {str(value) for value in always_include}
    candidates = []
    for raw in entries:
        row = dict(raw)
        code = str(row.get("code") or "")
        if not row.get("eligible") and code not in include_codes:
            continue
        amount = pd.to_numeric(row.get("amount"), errors="coerce")
        if (
            minimum_spot_amount is not None
            and pd.notna(amount)
            and float(amount) < minimum_spot_amount
            and code not in include_codes
        ):
            continue
        candidates.append(
            {
                "code": code,
                "name": str(row.get("name") or f"ETF {code}"),
                "category": str(row.get("category") or classify_etf(row.get("name", ""))),
                "amount": float(amount) if pd.notna(amount) else None,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["code"] not in include_codes,
            -(row["amount"] if row["amount"] is not None else -1.0),
            row["code"],
        )
    )
    if max_count and max_count > 0:
        candidates = candidates[:max_count]
    return [
        {key: row[key] for key in ("code", "name", "category")}
        for row in candidates
    ]
