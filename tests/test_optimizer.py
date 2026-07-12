"""Tests for strategy parameter optimizer."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.optimizer import (
    OptimizationResult,
    OptimizationReport,
    _smart_sample,
    _cartesian_product,
    generate_param_grid,
    score_result,
    run_optimization,
)
from src.strategy.four_percent_dca import FourPercentDCAStrategy
from src.strategy.trend_following import TrendFollowingStrategy
from src.strategy.grid_trading import GridTradingStrategy
from src.strategy.value_averaging import ValueAveragingStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """OHLCV data suitable for all strategies."""
    dates = pd.date_range("2025-01-01", periods=120, freq="B")
    np.random.seed(42)
    # Mild upward trend with some noise
    close = 4.0 + np.cumsum(np.random.randn(120) * 0.03)
    close = np.clip(close, 2.5, 7.0)
    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(100000, 500000, 120),
        "ma5": close + np.random.randn(120) * 0.02,
        "ma10": close + np.random.randn(120) * 0.03,
        "ma20": close + np.random.randn(120) * 0.05,
        "change_pct": np.random.randn(120) * 0.5,
        "amplitude": np.random.uniform(0.5, 2.0, 120),
    })
    return df


# ---------------------------------------------------------------------------
# _smart_sample
# ---------------------------------------------------------------------------


class TestSmartSample:
    """Parameter value generation from param descriptions."""

    def test_select_type(self):
        desc = {"type": "select", "options": ["ma5", "ma10", "ma20"]}
        result = _smart_sample(desc, "ma5")
        assert result == ["ma5", "ma10", "ma20"]

    def test_select_type_with_booleans(self):
        desc = {"type": "select", "options": ["True", "False"]}
        result = _smart_sample(desc, "True")
        assert True in result
        assert False in result

    def test_slider_type(self):
        desc = {"type": "slider", "min": 0.02, "max": 0.10, "step": 0.02}
        result = _smart_sample(desc, 0.04, max_values=5)
        assert len(result) >= 2
        assert all(0.02 <= v <= 0.10 for v in result)

    def test_slider_includes_default(self):
        desc = {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.1}
        result = _smart_sample(desc, 0.35)
        assert 0.35 in result

    def test_number_type(self):
        desc = {"type": "number", "min": 3, "max": 20, "step": 1}
        result = _smart_sample(desc, 10)
        assert 10 in result
        assert all(isinstance(v, (int, float)) for v in result)

    def test_number_type_respects_bounds(self):
        desc = {"type": "number", "min": 5, "max": 15, "step": 1}
        result = _smart_sample(desc, 5)
        assert all(5 <= v <= 15 for v in result)


# ---------------------------------------------------------------------------
# _cartesian_product
# ---------------------------------------------------------------------------


class TestCartesianProduct:
    """Cartesian product generation."""

    def test_two_params(self):
        candidates = {"a": [1, 2], "b": ["x", "y"]}
        result = _cartesian_product(candidates)
        assert len(result) == 4
        assert {"a": 1, "b": "x"} in result
        assert {"a": 2, "b": "y"} in result

    def test_empty(self):
        assert _cartesian_product({}) == [{}]

    def test_single_param(self):
        candidates = {"a": [1, 2, 3]}
        result = _cartesian_product(candidates)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# generate_param_grid
# ---------------------------------------------------------------------------


class TestGenerateParamGrid:
    """Full parameter grid generation for each strategy."""

    def test_generates_for_4pct_dca(self):
        strat = FourPercentDCAStrategy()
        grid = generate_param_grid(strat, max_combinations=50)
        assert len(grid) >= 2
        assert len(grid) <= 55  # allow slight buffer
        for combo in grid:
            assert "total_portions" in combo
            assert "drop_threshold_pct" in combo

    def test_generates_for_trend_following(self):
        strat = TrendFollowingStrategy()
        grid = generate_param_grid(strat, max_combinations=50)
        assert len(grid) >= 2
        for combo in grid:
            assert "fast_ma" in combo or "slow_ma" in combo

    def test_generates_for_grid_trading(self):
        strat = GridTradingStrategy()
        grid = generate_param_grid(strat, max_combinations=50)
        assert len(grid) >= 2

    def test_generates_for_value_averaging(self):
        strat = ValueAveragingStrategy()
        grid = generate_param_grid(strat, max_combinations=50)
        assert len(grid) >= 2

    def test_respects_max_combinations(self):
        strat = FourPercentDCAStrategy()
        grid = generate_param_grid(strat, max_combinations=30)
        assert len(grid) <= 35

    def test_all_combos_are_dicts_with_same_keys(self):
        strat = TrendFollowingStrategy()
        grid = generate_param_grid(strat, max_combinations=30)
        keys = set(grid[0].keys())
        for combo in grid:
            assert set(combo.keys()) == keys


# ---------------------------------------------------------------------------
# score_result
# ---------------------------------------------------------------------------


class TestScoreResult:
    """Composite fitness scoring."""

    def test_good_strategy_scores_high(self):
        score = score_result(
            annual_return=0.15, sharpe_ratio=1.5,
            max_drawdown=0.15, win_rate=0.60, total_trades=20,
        )
        assert score > 0.4

    def test_bad_strategy_scores_low(self):
        score = score_result(
            annual_return=-0.10, sharpe_ratio=-1.0,
            max_drawdown=0.50, win_rate=0.30, total_trades=10,
        )
        assert score < 0.4

    def test_few_trades_penalty(self):
        good = score_result(0.15, 1.5, 0.15, 0.60, 20)
        few = score_result(0.15, 1.5, 0.15, 0.60, 3)
        assert few < good

    def test_deep_drawdown_penalty(self):
        good = score_result(0.15, 1.5, 0.15, 0.60, 20)
        deep = score_result(0.15, 1.5, 0.50, 0.60, 20)
        assert deep < good

    def test_score_in_range(self):
        score = score_result(0.10, 0.8, 0.20, 0.50, 10)
        assert 0.0 <= score <= 1.0

    def test_nan_safe(self):
        """Should not crash with NaN inputs."""
        score = score_result(float('nan'), float('nan'), float('nan'), float('nan'), 0)
        assert score is not None


# ---------------------------------------------------------------------------
# run_optimization integration
# ---------------------------------------------------------------------------


class TestRunOptimization:
    """End-to-end optimization runs."""

    def test_optimize_4pct_dca(self, sample_df):
        report = run_optimization(
            sample_df, FourPercentDCAStrategy,
            max_combinations=20, pe_value=12.0,
        )
        assert report.strategy_name == "4%定投法"
        assert report.total_combinations > 0
        assert report.elapsed_seconds > 0
        assert report.best is not None
        assert report.best.score > 0
        assert len(report.top_n) >= 1

    def test_optimize_trend_following(self, sample_df):
        report = run_optimization(
            sample_df, TrendFollowingStrategy,
            max_combinations=20,
        )
        assert report.strategy_name == "趋势跟随"
        assert report.best is not None

    def test_optimize_grid(self, sample_df):
        report = run_optimization(
            sample_df, GridTradingStrategy,
            max_combinations=20,
        )
        assert report.best is not None

    def test_results_are_ranked(self, sample_df):
        report = run_optimization(
            sample_df, FourPercentDCAStrategy,
            max_combinations=15, pe_value=12.0,
        )
        for i, r in enumerate(report.all_results):
            assert r.rank == i + 1
        # First result has highest score
        assert report.all_results[0].score >= report.all_results[-1].score

    def test_top_n_is_at_most_10(self, sample_df):
        report = run_optimization(
            sample_df, FourPercentDCAStrategy,
            max_combinations=40, pe_value=12.0,
        )
        assert len(report.top_n) <= 10

    def test_invalid_params_dont_crash(self, sample_df):
        """Invalid parameter combos are skipped, not crashed."""
        report = run_optimization(
            sample_df, TrendFollowingStrategy,
            max_combinations=30,
        )
        # Should complete without exception
        assert report.best is not None


# ---------------------------------------------------------------------------
# OptimizationResult dataclass
# ---------------------------------------------------------------------------


class TestOptimizationResult:
    """Result container."""

    def test_to_dict(self):
        r = OptimizationResult(
            params={"a": 1, "b": 2},
            annual_return=0.12, sharpe_ratio=1.2,
            max_drawdown=0.15, calmar_ratio=0.8,
            win_rate=0.55, total_trades=15,
            final_equity=112000, score=0.65, rank=3,
        )
        d = r.to_dict()
        assert d["param_a"] == 1
        assert d["param_b"] == 2
        assert d["Sharpe"] == 1.2
        assert d["rank"] == 3
        assert "年化收益" in d

    def test_default_values(self):
        r = OptimizationResult()
        assert r.score == 0.0
        assert r.rank == 0
        assert r.params == {}
        assert r.total_trades == 0
