"""Strategy parameter optimizer — grid search over parameter space.

Given an ETF's historical data and a strategy, sweeps through parameter
combinations and ranks them by a composite fitness score.  Designed to
run in-process (no external dependencies) with a configurable cap on
total combinations to keep things responsive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import pandas as pd

from src.engine.backtest import BacktestEngine
from src.engine.broker import Broker
from src.engine.risk import RiskManager
from src.strategy.base import BaseStrategy

if TYPE_CHECKING:
    from src.data.pe_history import PEPercentile


# =========================================================================
# Data containers
# =========================================================================


@dataclass
class OptimizationResult:
    """One parameter combination's backtest output + fitness score."""

    params: dict = field(default_factory=dict)
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    final_equity: float = 0.0
    score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            **{f"param_{k}": v for k, v in self.params.items()},
            "年化收益": f"{self.annual_return:+.1%}",
            "Sharpe": round(self.sharpe_ratio, 2),
            "最大回撤": f"{self.max_drawdown:.1%}",
            "Calmar": round(self.calmar_ratio, 2),
            "胜率": f"{self.win_rate:.0%}",
            "交易次数": self.total_trades,
            "最终权益": f"¥{self.final_equity:,.0f}",
            "score": round(self.score, 4),
            "rank": self.rank,
        }


@dataclass
class OptimizationReport:
    """Full optimization run summary."""

    strategy_name: str = ""
    total_combinations: int = 0
    elapsed_seconds: float = 0.0
    best: OptimizationResult | None = None
    top_n: list[OptimizationResult] = field(default_factory=list)
    all_results: list[OptimizationResult] = field(default_factory=list)
    etf_code: str = ""
    pe_value: float | None = None


# =========================================================================
# Parameter grid generation
# =========================================================================


def _smart_sample(
    param_info: dict,
    default_value,
    max_values: int = 5,
) -> list:
    """Generate candidate values for a single parameter.

    - **select**: all options
    - **slider**: evenly-spaced samples (up to *max_values*)
    - **number**: defaults ± offsets (3-5 values)
    """
    ptype = param_info.get("type", "")

    if ptype == "select":
        options = param_info.get("options", [])
        # Parse "True"/"False" strings back to booleans if that's the pattern
        result = []
        for o in options:
            if isinstance(o, str):
                lo = o.lower()
                if lo == "true":
                    result.append(True)
                elif lo == "false":
                    result.append(False)
                else:
                    result.append(o)
            else:
                result.append(o)
        return result

    elif ptype == "slider":
        lo = float(param_info.get("min", 0))
        hi = float(param_info.get("max", 1))
        step = float(param_info.get("step", 0.1))
        n = min(max_values, max(2, int((hi - lo) / step) + 1))
        values = []
        for i in range(n):
            v = lo + (hi - lo) * i / max(n - 1, 1)
            # Round to step precision
            decimals = max(0, -int(math.floor(math.log10(step))) if step > 0 else 0)
            values.append(round(v, decimals))
        # Always include default if not already present
        if default_value is not None and default_value not in values:
            values.append(default_value)
            values.sort()
        return values

    elif ptype == "number":
        if not isinstance(default_value, (int, float)):
            default_value = float(param_info.get("min", 1))
        lo = float(param_info.get("min", default_value * 0.5))
        hi = float(param_info.get("max", default_value * 2.0))
        step = float(param_info.get("step", 1))
        # Sample: default, default±step, default±2*step
        values = set()
        for offset in (-2, -1, 0, 1, 2):
            v = default_value + offset * step
            if lo <= v <= hi:
                if isinstance(default_value, int):
                    values.add(int(v))
                else:
                    decimals = max(0, -int(math.floor(math.log10(step))) if step > 0 else 0)
                    values.add(round(v, decimals))
        return sorted(values)

    # Fallback
    return [default_value]


def generate_param_grid(
    strategy: BaseStrategy,
    max_combinations: int = 150,
) -> list[dict]:
    """Generate parameter combinations for grid search.

    When a strategy has many parameters, only the most impactful ones
    (up to ~5) are varied; the rest stay at their defaults.  This keeps
    the grid size manageable while still exploring the most important
    dimensions of the parameter space.
    """
    defaults = strategy.get_default_params()
    descriptions = strategy.get_param_descriptions()
    param_names = list(defaults.keys())

    # --- Pick which params to vary (most impactful first, up to ~5) ---
    # Heuristic: boolean/select params are more impactful (structural),
    # then sliders (continuous tuning), then numbers (fine-tuning).
    # We also prefer params with more candidate values.
    param_impact = []
    for key in param_names:
        desc = descriptions.get(key, {})
        ptype = desc.get("type", "")
        # Impact score: select=3, slider=2, number=1
        if ptype == "select":
            impact = 3
        elif ptype == "slider":
            impact = 2
        else:
            impact = 1
        param_impact.append((key, impact))

    # Sort by impact descending, then by name for determinism
    param_impact.sort(key=lambda x: (-x[1], x[0]))

    # Calculate how many params we can vary given max_combinations
    # target: 2-3 values per param, so max_params ≈ log_2.5(max_combinations)
    max_vary = min(len(param_names), max(3, int(__import__('math').log(max_combinations, 2.2))))
    vary_keys = {k for k, _ in param_impact[:max_vary]}

    # Build candidate values
    candidates: dict[str, list] = {}
    for key in param_names:
        if key in vary_keys:
            desc = descriptions.get(key, {})
            default = defaults[key]
            candidates[key] = _smart_sample(desc, default)
        else:
            # Fixed at default
            candidates[key] = [defaults[key]]

    # --- Prune further if still too many ---
    total = 1
    for vals in candidates.values():
        total *= len(vals)

    # Target ~2.2 values per param on average
    target_per_param = max(2, int(max_combinations ** (1.0 / max(len(vary_keys), 1))))

    if total > max_combinations:
        for key in vary_keys:
            if total <= max_combinations:
                break
            vals = candidates[key]
            if len(vals) <= 2:
                continue
            new_n = max(2, min(target_per_param, len(vals) // 2))
            keep = [vals[0]]
            step = max(1, (len(vals) - 1) // (new_n - 1))
            for i in range(1, new_n - 1):
                idx = min(i * step, len(vals) - 2)
                keep.append(vals[idx])
            keep.append(vals[-1])
            seen = set()
            deduped = []
            for v in keep:
                if v not in seen:
                    seen.add(v)
                    deduped.append(v)
            candidates[key] = deduped
            total = 1
            for vals in candidates.values():
                total *= len(vals)

    # --- Fallback: random sampling if still too big ---
    import random
    all_combos = _cartesian_product(candidates)
    if len(all_combos) > max_combinations:
        random.seed(42)
        default_combo = dict(defaults)
        for k, v in default_combo.items():
            if isinstance(v, bool):
                default_combo[k] = v
        sampled = [default_combo] if default_combo in all_combos else []
        remaining = [c for c in all_combos if c != default_combo]
        n_sample = min(max_combinations - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, n_sample))
        return sampled

    return all_combos


def _cartesian_product(candidates: dict[str, list]) -> list[dict]:
    """Generate the Cartesian product of candidate values."""
    if not candidates:
        return [{}]

    keys = list(candidates.keys())
    result = [{}]

    for key in keys:
        new_result = []
        for d in result:
            for val in candidates[key]:
                new_d = dict(d)
                new_d[key] = val
                new_result.append(new_d)
        result = new_result

    return result


# =========================================================================
# Scoring
# =========================================================================


def score_result(
    annual_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    total_trades: int,
) -> float:
    """Composite fitness score.  Higher = better.

    Weights:
      - Sharpe ratio:  40%  (risk-adjusted return is king)
      - Calmar ratio:  25%  (penalises deep drawdowns)
      - Annual return: 20%  (absolute performance matters)
      - Win rate:      15%  (consistency bonus)

    Penalties:
      - < 5 trades: score × 0.5  (too few = overfit / unreliable)
      - max_drawdown > 40%: score × 0.7
    """
    # Calmar = annual_return / |max_drawdown|
    calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 and max_drawdown is not None else 0.0

    # Normalise each component to roughly 0-1 range for fair weighting
    # Sharpe: typical range -2 to +4 → map to 0-1
    sharpe_norm = max(0.0, min(1.0, (sharpe_ratio + 1.0) / 3.0))
    # Calmar: typical range 0 to 3 → map to 0-1
    calmar_norm = max(0.0, min(1.0, calmar / 2.0))
    # Annual return: -50% to +50% → map to 0-1
    ret_norm = max(0.0, min(1.0, (annual_return + 0.30) / 0.80))
    # Win rate: already 0-1
    wr_norm = max(0.0, min(1.0, win_rate)) if win_rate is not None else 0.5

    score = (
        sharpe_norm * 0.40
        + calmar_norm * 0.25
        + ret_norm * 0.20
        + wr_norm * 0.15
    )

    # Penalties
    if total_trades < 5:
        score *= 0.5
    if max_drawdown is not None and max_drawdown > 0.40:
        score *= 0.7

    return round(score, 4)


# =========================================================================
# Optimization runner
# =========================================================================


def _params_match_defaults(
    strategy_cls: type[BaseStrategy],
    params: dict,
) -> bool:
    """Check if *params* matches the strategy's defaults."""
    defaults = strategy_cls().get_default_params()
    for k, v in defaults.items():
        pv = params.get(k)
        # Normalize bool/string
        if isinstance(v, bool) and isinstance(pv, str):
            pv = pv.lower() in ("true", "1", "yes")
        if pv != v:
            return False
    return True


def run_optimization(
    df: pd.DataFrame,
    strategy_cls: type[BaseStrategy],
    max_combinations: int = 120,
    pe_value: float | None = None,
    pe_percentile: "PEPercentile | None" = None,
    initial_capital: float = 100_000,
    progress_callback=None,
) -> OptimizationReport:
    """Run grid-search parameter optimization for *strategy_cls* on *df*.

    Args:
        df: OHLCV DataFrame (date-ascending).
        strategy_cls: Strategy class (uninstantiated).
        max_combinations: Upper bound on param combos to test.
        pe_value: Optional PE snapshot for PE-aware strategies.
        pe_percentile: Optional PE percentile data.
        initial_capital: Starting capital for each backtest.
        progress_callback: Optional callable(completed, total) for UI updates.

    Returns:
        OptimizationReport with ranked results.
    """
    import time
    t0 = time.time()

    strategy = strategy_cls()
    strategy_name = strategy.name
    param_grid = generate_param_grid(strategy, max_combinations=max_combinations)

    results: list[OptimizationResult] = []
    total = len(param_grid)

    for i, params in enumerate(param_grid):
        # Build strategy + engine
        inst = strategy_cls()
        engine = BacktestEngine(
            initial_capital=initial_capital,
            broker=Broker(),
            risk_manager=RiskManager(),
        )

        try:
            btr = engine.run(
                df.copy(), inst,
                pe_value=pe_value,
                pe_percentile=pe_percentile,
                **params,
            )
        except Exception:
            # Skip combinations that cause errors (e.g. invalid MA columns)
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        score = score_result(
            annual_return=btr.annual_return,
            sharpe_ratio=btr.sharpe_ratio,
            max_drawdown=btr.max_drawdown,
            win_rate=btr.win_rate,
            total_trades=btr.total_trades,
        )

        results.append(OptimizationResult(
            params=params,
            annual_return=btr.annual_return,
            sharpe_ratio=btr.sharpe_ratio,
            max_drawdown=btr.max_drawdown,
            calmar_ratio=btr.calmar_ratio,
            win_rate=btr.win_rate,
            total_trades=btr.total_trades,
            final_equity=btr.final_equity,
            score=score,
        ))

        if progress_callback:
            progress_callback(i + 1, total)

    # Sort by score descending, assign ranks
    results.sort(key=lambda r: r.score, reverse=True)
    for rank, r in enumerate(results, 1):
        r.rank = rank

    elapsed = time.time() - t0

    # Find the best result that differs from defaults (if possible)
    best = results[0] if results else None
    for r in results:
        if not _params_match_defaults(strategy_cls, r.params):
            best = r
            break

    return OptimizationReport(
        strategy_name=strategy_name,
        total_combinations=len(param_grid),
        elapsed_seconds=round(elapsed, 2),
        best=best,
        top_n=results[:10],
        all_results=results,
        pe_value=pe_value,
    )
