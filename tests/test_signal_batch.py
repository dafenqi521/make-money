"""Tests for deterministic persisted rotation signals."""

from datetime import date

import pandas as pd

from src.engine.rotation_scanner import RotationScanResult
from src.engine.signal_batch import (
    batch_id_for,
    deserialize_scan_result,
    serialize_scan_result,
)
from src.strategy.etf_rotation import RotationConfig



def _scan() -> RotationScanResult:
    history = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=130)})
    rankings = pd.DataFrame(
        [{"code": "510300", "name": "沪深300ETF", "eligible": True, "close": 4.0}]
    )
    targets = pd.DataFrame(
        [{"code": "510300", "name": "沪深300ETF", "target_weight": 0.3}]
    )
    return RotationScanResult(
        rankings,
        targets,
        {"510300": history},
        {},
        date(2026, 7, 17),
    )


def test_signal_batch_id_is_deterministic_and_payload_roundtrips():
    scan = _scan()
    pool = [{"code": code} for code in scan.histories]
    config = RotationConfig(min_avg_amount=1, min_daily_amount=1)

    first = batch_id_for(scan, config, pool)
    second = batch_id_for(scan, config, list(reversed(pool)))
    restored = deserialize_scan_result(serialize_scan_result(scan))

    assert first == second
    assert restored.as_of == scan.as_of
    assert restored.scanned_count == scan.scanned_count
    assert set(restored.rankings["code"]) == set(scan.rankings["code"])
