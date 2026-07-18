"""Deterministic persistence format for ETF rotation signals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Iterable

import pandas as pd

from src.engine.rotation_scanner import RotationScanResult
from src.strategy.etf_rotation import RotationConfig


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_hash(config: RotationConfig) -> str:
    return _stable_hash(asdict(config))


def pool_hash(pool: Iterable[dict | str]) -> str:
    rows = []
    for item in pool:
        if isinstance(item, str):
            rows.append({"code": item})
        else:
            rows.append(
                {
                    "code": str(item.get("code") or ""),
                    "name": str(item.get("name") or ""),
                    "category": str(item.get("category") or ""),
                }
            )
    rows.sort(key=lambda row: row["code"])
    return _stable_hash(rows)


def batch_id_for(
    scan: RotationScanResult,
    config: RotationConfig,
    pool: Iterable[dict | str],
) -> str:
    signal_date = scan.as_of.isoformat() if scan.as_of else "none"
    return _stable_hash(
        {
            "signal_date": signal_date,
            "config_hash": config_hash(config),
            "pool_hash": pool_hash(pool),
        }
    )


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def serialize_scan_result(scan: RotationScanResult) -> dict:
    """Store rankings/targets and only the history dates needed by exits."""

    history_dates = {}
    for code, history in scan.histories.items():
        if history is None or history.empty or "date" not in history.columns:
            history_dates[str(code)] = []
            continue
        dates = (
            pd.to_datetime(history["date"], errors="coerce")
            .dropna()
            .sort_values()
            .tail(260)
        )
        history_dates[str(code)] = [value.date().isoformat() for value in dates]
    return {
        "schema_version": 1,
        "as_of": scan.as_of.isoformat() if scan.as_of else None,
        "rankings": _records(scan.rankings),
        "targets": _records(scan.targets),
        "history_dates": history_dates,
        "errors": {str(key): str(value) for key, value in scan.errors.items()},
    }


def deserialize_scan_result(payload: dict) -> RotationScanResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported signal payload")
    rankings = pd.DataFrame(payload.get("rankings") or [])
    targets = pd.DataFrame(payload.get("targets") or [])
    if "eligible" in rankings.columns:
        rankings["eligible"] = rankings["eligible"].astype(bool)
    histories = {
        str(code): pd.DataFrame(
            {"date": pd.to_datetime(dates or [], errors="coerce")}
        )
        for code, dates in (payload.get("history_dates") or {}).items()
    }
    as_of = pd.to_datetime(payload.get("as_of"), errors="coerce")
    return RotationScanResult(
        rankings=rankings,
        targets=targets,
        histories=histories,
        errors={
            str(key): str(value)
            for key, value in (payload.get("errors") or {}).items()
        },
        as_of=as_of.date() if pd.notna(as_of) else None,
    )
