"""Tests for close-scan persistence and next-session reminder jobs."""

from datetime import date, datetime

import pandas as pd

from src.data.portfolio_db import PortfolioDB
from src.engine.rotation_scanner import RotationScanResult
from src.jobs.daily_signal import refresh_universe, run_reminder_job, run_scan_job


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "代码": [f"51{index:04d}" for index in range(60)],
            "名称": [f"测试{index}ETF" for index in range(60)],
            "最新价": [1.0] * 60,
            "成交额": [100_000_000] * 60,
        }
    )


def _scanner(**kwargs) -> RotationScanResult:
    history = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=130)})
    rankings = pd.DataFrame(
        [{"code": "510001", "name": "测试ETF", "eligible": True, "close": 1.0}]
    )
    targets = pd.DataFrame(
        [{"code": "510001", "name": "测试ETF", "target_weight": 0.3}]
    )
    return RotationScanResult(
        rankings,
        targets,
        {"510001": history},
        {},
        date(2026, 7, 20),
    )


def test_background_scan_saves_signal_and_reminder_uses_next_session(tmp_path):
    db = PortfolioDB(db_path=tmp_path / "job.sqlite3")
    scan_result = run_scan_job(
        db,
        now=datetime(2026, 7, 20, 15, 20),
        universe_fetchers={"mock": _universe_frame},
        scanner=_scanner,
    )
    reminder = run_reminder_job(
        db, now=datetime(2026, 7, 21, 9, 35)
    )

    assert scan_result["status"] == "success"
    assert scan_result["universe_count"] == 60
    assert db.get_latest_signal_batch()["batch_id"] == scan_result["batch_id"]
    assert reminder["status"] == "success"


def test_background_scan_skips_exchange_holiday(tmp_path):
    db = PortfolioDB(db_path=tmp_path / "holiday.sqlite3")

    result = run_scan_job(
        db,
        now=datetime(2026, 10, 1, 15, 20),
        universe_fetchers={"mock": _universe_frame},
        scanner=_scanner,
    )

    assert result == {"status": "skipped", "reason": "休市日"}


def test_universe_refresh_rejects_large_row_count_drop(tmp_path):
    db = PortfolioDB(db_path=tmp_path / "drift.sqlite3")
    full, first_status = refresh_universe(
        db,
        now=datetime(2026, 7, 20, 15, 20),
        fetchers={"full": lambda: _universe_frame().loc[
            _universe_frame().index.repeat(2)
        ].assign(代码=lambda frame: [f"51{i:04d}" for i in range(len(frame))])},
    )
    retained, second_status = refresh_universe(
        db,
        now=datetime(2026, 7, 21, 15, 20),
        fetchers={"partial": _universe_frame},
    )

    assert len(full) == 120
    assert first_status["status"] == "success"
    assert len(retained) == 120
    assert second_status["status"] == "failed"
