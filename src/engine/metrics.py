"""Performance metrics — Sharpe, Max Drawdown, CAGR, win rate.

Pure functions — no side effects, no state.  All computations are
vectorized via numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: list,
    initial_capital: float,
    risk_free_rate: float = 0.03,
) -> dict:
    """Compute all performance metrics from equity curve and trade list.

    Args:
        equity_curve: DataFrame with columns [date, equity, cash, shares].
        trades: List of completed Trade objects.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate (default 3%).

    Returns:
        dict with keys: total_return, annual_return, annual_volatility,
        sharpe_ratio, max_drawdown, calmar_ratio, win_rate,
        total_trades, winning_trades, losing_trades, avg_holding_days.
    """
    result: dict = {
        "total_return": 0.0,
        "annual_return": 0.0,
        "annual_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "total_trades": len(trades),
        "winning_trades": 0,
        "losing_trades": 0,
        "avg_holding_days": 0.0,
    }

    if equity_curve.empty or equity_curve["equity"].iloc[-1] <= 0:
        return result

    final_equity = equity_curve["equity"].iloc[-1]

    # --- Total return ---
    result["total_return"] = (final_equity - initial_capital) / initial_capital

    # --- Daily returns ---
    equity = equity_curve.set_index("date")["equity"]
    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 2:
        return result

    # --- Annual return (CAGR) ---
    days = (equity.index[-1] - equity.index[0]).days
    if days > 0:
        years = days / 365.25
        if years > 0 and final_equity > 0 and initial_capital > 0:
            result["annual_return"] = (
                (final_equity / initial_capital) ** (1.0 / years) - 1.0
            )

    # --- Annual volatility ---
    daily_vol = daily_returns.std()
    result["annual_volatility"] = daily_vol * np.sqrt(252)

    # --- Sharpe ratio ---
    if result["annual_volatility"] > 0:
        result["sharpe_ratio"] = (
            result["annual_return"] - risk_free_rate
        ) / result["annual_volatility"]

    # --- Max drawdown ---
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    result["max_drawdown"] = float(drawdown.min())  # negative number

    # --- Calmar ratio ---
    if abs(result["max_drawdown"]) > 0:
        result["calmar_ratio"] = result["annual_return"] / abs(result["max_drawdown"])

    # --- Trade statistics ---
    closed_trades = [t for t in trades if t.pnl is not None]
    if closed_trades:
        winning = [t for t in closed_trades if t.pnl > 0]
        result["winning_trades"] = len(winning)
        result["losing_trades"] = len(closed_trades) - len(winning)
        result["win_rate"] = len(winning) / len(closed_trades)

        holding_days = [
            t.holding_days for t in closed_trades if t.holding_days is not None
        ]
        if holding_days:
            result["avg_holding_days"] = float(np.mean(holding_days))

    return result


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown at each point in time (0 = no drawdown, -0.2 = 20% DD)."""
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax.replace(0, np.nan)
    return drawdown.fillna(0.0)
