"""Tests for scan-driven local paper trading and rebalance safety gates."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.engine.paper_trading import build_rebalance_plan, execute_rebalance_plan
from src.engine.portfolio import PortfolioManager
from src.engine.rotation_scanner import RotationScanResult, scan_etf_pool
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


def _scan() -> RotationScanResult:
    pool = [
        {"code": "510300", "name": "沪深300ETF", "category": "domestic_broad"},
        {"code": "513100", "name": "纳指ETF", "category": "overseas_equity"},
        {"code": "518880", "name": "黄金ETF", "category": "commodity"},
    ]
    return scan_etf_pool(
        pool,
        config=RotationConfig(min_avg_amount=1, min_daily_amount=1),
        history_fetcher=_history,
        quote_fetcher=lambda codes: {},
    )


def test_build_plan_creates_lot_sized_buys_from_scan_targets():
    scan = _scan()
    pm = PortfolioManager(initial_capital=100_000)
    trade_date = scan.as_of.isoformat()

    plan = build_rebalance_plan(
        pm,
        scan,
        RotationConfig(min_avg_amount=1, min_daily_amount=1),
        trade_date=trade_date,
    )

    buys = plan.orders[plan.orders["action"] == "buy"]
    assert not buys.empty
    assert (buys["delta_shares"] % 100 == 0).all()
    assert (buys["target_weight"] > 0).all()
    assert not plan.errors


def test_execute_plan_applies_slippage_and_updates_account():
    scan = _scan()
    pm = PortfolioManager(initial_capital=100_000)
    trade_date = scan.as_of.isoformat()
    plan = build_rebalance_plan(
        pm,
        scan,
        RotationConfig(min_avg_amount=1, min_daily_amount=1),
        trade_date=trade_date,
    )

    execution = execute_rebalance_plan(
        pm, plan, trade_date=trade_date, slippage_pct=0.001
    )

    assert execution.trades
    assert not execution.errors
    assert pm.total_trades == len(execution.trades)
    for trade in execution.trades:
        reference = float(
            plan.orders.loc[plan.orders["code"] == trade.code, "reference_price"].iloc[0]
        )
        assert trade.price == reference * 1.001
        assert trade.shares % 100 == 0


def test_same_day_bought_shares_are_not_sellable():
    pm = PortfolioManager(initial_capital=100_000)
    assert pm.buy("510300", 4.0, 1000, trade_date="2025-07-01") is not None

    assert pm.available_shares("510300", "2025-07-01") == 0
    assert pm.sell("510300", 4.1, 1000, trade_date="2025-07-01") is None
    assert pm.sell("510300", 4.1, 1000, trade_date="2025-07-02") is not None


def test_scan_missing_held_symbol_freezes_that_position():
    scan = _scan()
    pm = PortfolioManager(initial_capital=100_000)
    pm.buy("512480", 1.0, 1000, name="半导体ETF", trade_date="2025-01-02")

    plan = build_rebalance_plan(
        pm,
        scan,
        RotationConfig(min_avg_amount=1, min_daily_amount=1),
        trade_date=scan.as_of.isoformat(),
    )

    row = plan.orders.loc[plan.orders["code"] == "512480"].iloc[0]
    assert row["action"] == "hold"
    assert "512480" in plan.errors


def test_stale_scan_freezes_all_automatic_orders():
    scan = _scan()
    pm = PortfolioManager(initial_capital=100_000)

    plan = build_rebalance_plan(
        pm,
        scan,
        RotationConfig(min_avg_amount=1, min_daily_amount=1),
        trade_date="2026-07-19",
    )

    assert plan.actionable_count == 0
    assert "freshness" in plan.errors


def test_future_dated_scan_freezes_all_automatic_orders():
    scan = _scan()
    pm = PortfolioManager(initial_capital=100_000)

    plan = build_rebalance_plan(
        pm,
        scan,
        RotationConfig(min_avg_amount=1, min_daily_amount=1),
        trade_date=(scan.as_of - timedelta(days=1)).isoformat(),
    )

    assert plan.actionable_count == 0
    assert "freshness" in plan.errors


def test_hard_stop_generates_sell_for_existing_holding():
    history = _history("510300")
    as_of = history["date"].max().date()
    ranking = pd.DataFrame(
        [
            {
                "code": "510300",
                "name": "沪深300ETF",
                "category": "domestic_broad",
                "eligible": False,
                "score": np.nan,
                "close": 0.90,
                "ma5": 0.95,
                "ma10": 0.96,
                "ma20": 0.98,
                "atr20_pct": 0.01,
            }
        ]
    )
    scan = RotationScanResult(
        rankings=ranking,
        targets=pd.DataFrame(),
        histories={"510300": history},
        errors={},
        as_of=as_of,
    )
    pm = PortfolioManager(initial_capital=100_000)
    pm.buy("510300", 1.0, 1000, name="沪深300ETF", trade_date="2025-01-02")

    plan = build_rebalance_plan(pm, scan, trade_date=as_of.isoformat())

    row = plan.orders.iloc[0]
    assert row["action"] == "sell"
    assert row["delta_shares"] == -1000
    assert "止损" in row["reason"]
