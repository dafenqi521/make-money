"""Tests for automatic full-market ETF universe maintenance."""

from datetime import datetime

import pandas as pd

from src.engine.etf_universe import (
    discover_etf_universe,
    normalise_universe_frame,
    select_scan_pool,
)


def _spot_frame(count: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "代码": [f"51{index:04d}" for index in range(count)],
            "名称": ["货币ETF" if index == 0 else f"测试{index}ETF" for index in range(count)],
            "最新价": [1.0 + index / 100 for index in range(count)],
            "成交额": [index * 1_000_000 for index in range(count)],
        }
    )


def test_normalise_universe_has_unique_grain_and_exclusion_evidence():
    records = normalise_universe_frame(
        _spot_frame(), "mock", refreshed_at=datetime(2026, 7, 20, 15, 20)
    )

    assert len(records) == 60
    assert len({row["code"] for row in records}) == 60
    assert not records[0]["eligible"]
    assert "货币" in records[0]["exclusion_reason"]
    assert records[1]["source"] == "mock"


def test_discovery_falls_back_to_second_source():
    result = discover_etf_universe(
        fetchers={
            "broken": lambda: (_ for _ in ()).throw(RuntimeError("down")),
            "working": _spot_frame,
        },
        refreshed_at=datetime(2026, 7, 20, 15, 20),
    )

    assert result.source == "working"
    assert result.total_count == 60
    assert "broken" in result.errors


def test_scan_pool_applies_liquidity_limit_and_always_keeps_holdings():
    records = normalise_universe_frame(_spot_frame(), "mock")
    selected = select_scan_pool(
        records,
        minimum_spot_amount=50_000_000,
        max_count=5,
        always_include=["510001"],
    )

    assert selected[0]["code"] == "510001"
    assert len(selected) == 5
    assert all(len(row["code"]) == 6 for row in selected)
