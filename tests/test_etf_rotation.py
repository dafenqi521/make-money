"""Tests for the exchange-traded ETF trend-rotation core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.etf_rotation import (
    PositionState,
    RotationConfig,
    classify_etf,
    compute_etf_features,
    evaluate_exit,
    rank_etfs,
    select_targets,
)


def _history(
    daily_return: float = 0.001,
    days: int = 150,
    money: float = 80_000_000,
    noise_scale: float = 0.0005,
) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = daily_return + rng.normal(0, noise_scale, size=days)
    close = 1.0 * np.cumprod(1.0 + returns)
    dates = pd.bdate_range("2025-01-01", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "volume": money / close,
            "money": [money] * days,
        }
    )


def test_classify_etf_categories():
    assert classify_etf("沪深300ETF") == "domestic_broad"
    assert classify_etf("纳指ETF") == "overseas_equity"
    assert classify_etf("黄金ETF") == "commodity"
    assert classify_etf("国债ETF") == "bond"
    assert classify_etf("半导体ETF") == "domestic_sector"


def test_feature_filter_accepts_liquid_uptrend():
    feature = compute_etf_features(
        _history(),
        {"name": "沪深300ETF", "listed_days": 1000},
    )
    assert feature["eligible"] is True
    assert feature["return20"] > 0
    assert feature["avg_amount20"] == pytest.approx(80_000_000)


def test_feature_filter_rejects_illiquid_etf():
    feature = compute_etf_features(
        _history(money=1_000_000),
        {"name": "冷门ETF", "listed_days": 1000},
    )
    assert feature["eligible"] is False
    assert "20日平均成交额不足" in feature["rejection_reasons"]


def test_rank_etfs_prefers_stronger_momentum():
    histories = {
        "510001": _history(daily_return=0.0005, noise_scale=0.0002),
        "510002": _history(daily_return=0.0015, noise_scale=0.0002),
    }
    metadata = {
        "510001": {"name": "宽基一ETF", "listed_days": 1000, "category": "other"},
        "510002": {"name": "宽基二ETF", "listed_days": 1000, "category": "other"},
    }
    ranked = rank_etfs(histories, metadata)
    assert ranked.iloc[0]["code"] == "510002"
    assert ranked.iloc[0]["score"] > ranked.iloc[1]["score"]


def test_select_targets_deduplicates_high_correlation():
    base = _history(daily_return=0.001)
    histories = {
        "510001": base,
        "510002": base.assign(close=base["close"] * 1.01),
        "513001": _history(daily_return=0.0012, noise_scale=0.002),
    }
    metadata = {
        "510001": {"name": "沪深300ETF甲", "listed_days": 1000},
        "510002": {"name": "沪深300ETF乙", "listed_days": 1000},
        "513001": {"name": "纳指ETF", "listed_days": 1000},
    }
    ranked = rank_etfs(histories, metadata)
    targets = select_targets(ranked, histories)
    assert len(set(targets["code"]).intersection({"510001", "510002"})) == 1


def test_target_weights_respect_cash_and_position_caps():
    histories = {
        f"51000{i}": _history(daily_return=0.0008 + i * 0.0001, noise_scale=0.001 * i)
        for i in range(1, 6)
    }
    metadata = {
        code: {"name": f"测试ETF{code}", "listed_days": 1000, "category": "other"}
        for code in histories
    }
    config = RotationConfig(
        category_position_limits={"other": 4},
        category_weight_caps={"other": 0.90},
    )
    targets = select_targets(rank_etfs(histories, metadata, config), histories, config)
    assert len(targets) <= 4
    assert targets["target_weight"].max() <= 0.30
    assert targets["target_weight"].sum() <= 0.90 + 1e-8


def test_exit_hard_stop_and_t1_guard():
    feature = {"atr20_pct": 0.015, "ma5": 1.0, "ma10": 0.99}
    position = PositionState("510300", 1.0, 1.02, holding_days=2)
    decision = evaluate_exit(position, feature, 0.95, 3, 10)
    assert decision.should_exit is True
    assert decision.reason_code == "hard_stop"

    blocked = evaluate_exit(
        PositionState("510300", 1.0, 1.02, holding_days=0, closeable=False),
        feature,
        0.90,
        3,
        10,
    )
    assert blocked.should_exit is False
    assert blocked.reason_code == "not_closeable"


def test_exit_time_stop_for_stale_position():
    position = PositionState("510300", 1.0, 1.01, holding_days=10)
    feature = {"atr20_pct": 0.01, "ma5": 1.01, "ma10": 1.00}
    decision = evaluate_exit(position, feature, 1.005, 9, 12)
    assert decision.should_exit is True
    assert decision.reason_code == "time_stop"

