"""Tests for the look-ahead-safe ETF rotation backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.backtest import (
    BacktestSettings,
    fetch_backtest_histories,
    run_parameter_sweep,
    run_rotation_backtest,
)
from src.strategy.etf_rotation import RotationConfig


def _history(code: str, days: int = 360) -> pd.DataFrame:
    seed = int(code[-2:])
    rng = np.random.default_rng(seed)
    drift = 0.0005 + (seed % 5) * 0.00012
    returns = drift + rng.normal(0, 0.002, days)
    close = np.cumprod(1.0 + returns)
    dates = pd.bdate_range("2023-01-02", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1.0 + rng.normal(0, 0.0005, days)),
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "volume": 100_000_000 / close,
            "money": 100_000_000.0,
        }
    )


def _inputs():
    codes = ("510301", "510302", "510303", "510304")
    histories = {code: _history(code) for code in codes}
    metadata = {
        code: {
            "name": f"测试宽基ETF{code}",
            "category": "domestic_broad",
        }
        for code in codes
    }
    dates = histories[codes[0]]["date"]
    settings = BacktestSettings(
        start_date=dates.iloc[150].date(),
        end_date=dates.iloc[-1].date(),
        initial_capital=100_000,
        slippage_pct=0.001,
        benchmark_code=codes[0],
    )
    config = RotationConfig(
        min_avg_amount=1,
        min_daily_amount=1,
        max_positions=2,
        category_position_limits={"domestic_broad": 4},
        category_weight_caps={"domestic_broad": 0.90},
    )
    return histories, metadata, config, settings


def test_backtest_executes_signals_on_next_session_open():
    histories, metadata, config, settings = _inputs()

    result = run_rotation_backtest(histories, metadata, config, settings)

    assert len(result.equity_curve) > 150
    assert not result.signal_log.empty
    assert not result.trades.empty
    first_trade = result.trades["date"].min()
    first_actionable = result.signal_log.loc[
        result.signal_log["actionable_count"] > 0
    ].iloc[0]
    assert first_trade >= first_actionable["execution_date"]
    assert first_actionable["execution_date"] > first_actionable["signal_date"]
    assert result.metrics["trade_count"] == len(result.trades)


def test_slippage_reduces_strategy_equity():
    histories, metadata, config, settings = _inputs()
    zero_slippage = run_rotation_backtest(
        histories,
        metadata,
        config,
        BacktestSettings(**{**settings.__dict__, "slippage_pct": 0.0}),
    )
    high_slippage = run_rotation_backtest(
        histories,
        metadata,
        config,
        BacktestSettings(**{**settings.__dict__, "slippage_pct": 0.005}),
    )

    assert high_slippage.equity_curve["equity"].iloc[-1] <= zero_slippage.equity_curve[
        "equity"
    ].iloc[-1]


def test_backtest_reports_benchmark_months_and_drawdown():
    histories, metadata, config, settings = _inputs()

    result = run_rotation_backtest(histories, metadata, config, settings)

    assert result.benchmark_code == settings.benchmark_code
    assert len(result.monthly_returns) >= 6
    assert result.equity_curve["drawdown"].max() <= 1e-12
    assert result.equity_curve["benchmark_drawdown"].max() <= 1e-12
    assert "annual_return" in result.benchmark_metrics


def test_fetch_result_keeps_failures_explicit():
    sample = _history("510301")

    def fetcher(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if code == "510302":
            raise RuntimeError("network down")
        return sample

    start = sample["date"].iloc[150].date()
    end = sample["date"].iloc[-1].date()
    data = fetch_backtest_histories(
        ["510301", "510302"],
        start,
        end,
        history_fetcher=fetcher,
        max_workers=2,
    )

    assert data.coverage == 0.5
    assert "510302" in data.errors
    assert "510302" not in data.histories


def test_parameter_sweep_returns_requested_variants():
    histories, metadata, config, settings = _inputs()
    weekly = BacktestSettings(
        **{**settings.__dict__, "signal_frequency": "weekly"}
    )

    sweep = run_parameter_sweep(
        histories,
        metadata,
        config,
        weekly,
        max_positions_values=(2, 3),
    )

    assert sweep["max_positions"].tolist() == [2, 3]
    assert set(("total_return", "max_drawdown", "sharpe_ratio")).issubset(sweep)


def test_missing_daily_bar_lowers_coverage_and_freezes_signal():
    histories, metadata, config, settings = _inputs()
    missing_date = histories["510304"]["date"].iloc[200]
    histories["510304"] = histories["510304"].loc[
        histories["510304"]["date"] != missing_date
    ]

    result = run_rotation_backtest(histories, metadata, config, settings)

    affected = result.signal_log.loc[result.signal_log["signal_date"] == missing_date]
    assert not affected.empty
    assert affected.iloc[0]["coverage"] == 0.75
    assert bool(affected.iloc[0]["frozen"])
    assert result.coverage == 0.75
