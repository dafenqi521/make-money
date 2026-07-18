"""The project's single supported strategy: exchange-traded ETF rotation."""

from src.strategy.etf_rotation import (
    ExitDecision,
    PositionState,
    RotationConfig,
    classify_etf,
    compute_etf_features,
    evaluate_exit,
    rank_etfs,
    select_targets,
)

__all__ = [
    "ExitDecision",
    "PositionState",
    "RotationConfig",
    "classify_etf",
    "compute_etf_features",
    "evaluate_exit",
    "rank_etfs",
    "select_targets",
]
