"""Tests for the focused ETF rotation scanner service."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.rotation_scanner import normalise_pool, scan_etf_pool
from src.strategy.etf_rotation import RotationConfig


def _history(code: str) -> pd.DataFrame:
    seed = int(code[-2:])
    rng = np.random.default_rng(seed)
    returns = 0.001 + rng.normal(0, 0.0005, 150)
    close = np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=150),
            "close": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "volume": 100_000_000 / close,
            "money": 100_000_000,
        }
    )


def test_normalise_pool_deduplicates_and_rejects_bad_codes():
    pool = normalise_pool(["510300", "510300", "abc", {"code": "513100", "name": "纳指ETF"}])
    assert [item["code"] for item in pool] == ["510300", "513100"]


def test_scan_etf_pool_returns_rankings_and_targets():
    pool = [
        {"code": "510300", "name": "沪深300ETF", "category": "domestic_broad"},
        {"code": "513100", "name": "纳指ETF", "category": "overseas_equity"},
        {"code": "518880", "name": "黄金ETF", "category": "commodity"},
    ]
    result = scan_etf_pool(
        pool,
        config=RotationConfig(min_avg_amount=1, min_daily_amount=1),
        history_fetcher=_history,
        quote_fetcher=lambda codes: {},
        max_workers=2,
    )
    assert result.scanned_count == 3
    assert result.eligible_count == 3
    assert not result.targets.empty
    assert result.targets["target_weight"].sum() <= 0.90 + 1e-8


def test_scan_failure_is_explicit_and_not_ranked():
    def failing_fetcher(code: str) -> pd.DataFrame:
        if code == "510300":
            raise RuntimeError("network down")
        return _history(code)

    result = scan_etf_pool(
        ["510300", "513100"],
        config=RotationConfig(min_avg_amount=1, min_daily_amount=1),
        history_fetcher=failing_fetcher,
        quote_fetcher=lambda codes: {},
    )
    assert "510300" in result.errors
    assert "510300" not in set(result.rankings["code"])

