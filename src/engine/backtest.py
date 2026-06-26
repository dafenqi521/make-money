"""Lightweight backtesting engine.

Walks through historical data bar-by-bar, maintaining portfolio state,
and produces a BacktestResult with equity curve, trades, and metrics.

Design:
  - Strategies are stateless — they produce signals, not state.
  - The engine manages cash, shares, cost basis, and trade pairing.
  - Commission + slippage applied via Broker.
  - Risk checks applied via RiskManager.
"""

from __future__ import annotations

import pandas as pd

from src.engine.broker import Broker
from src.engine.risk import RiskManager
from src.engine.metrics import compute_metrics, compute_drawdown_series
from src.strategy.signals import Signal, Trade, Position, BacktestResult
from src.strategy.base import BaseStrategy


class BacktestEngine:
    """Lightweight vectorized backtesting engine.

    Usage::

        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(df, strategy, pe_value=None, **params)
        print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        broker: Broker | None = None,
        risk_manager: RiskManager | None = None,
    ):
        self.initial_capital = initial_capital
        self.broker = broker or Broker()
        self.risk = risk_manager or RiskManager()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        pe_value: float | None = None,
        **strategy_params,
    ) -> BacktestResult:
        """Execute a full backtest.

        Args:
            df: OHLCV DataFrame from fetch_etf_hist(), sorted date ASCENDING.
            strategy: Strategy instance.
            pe_value: Current PE snapshot (or None). Broadcast to all dates.
            **strategy_params: Passed to strategy.generate_signals().

        Returns:
            BacktestResult with equity curve, trades, and all metrics.
        """
        # --- 1. Generate signals ---
        params = {**strategy.get_default_params(), **strategy_params}
        sig_df = strategy.generate_signals(df, **params)

        # --- 2. Walk through bars chronologically ---
        sig_df = sig_df.sort_values("date", ascending=True).reset_index(drop=True)

        cash = self.initial_capital
        shares = 0
        cost_basis = 0.0
        grid_inventory: dict[int, dict] = {}  # level_idx → {shares, price}

        equity_rows: list[dict] = []
        signals_log: list[Signal] = []
        open_trade: Trade | None = None
        closed_trades: list[Trade] = []
        warnings_log: list[str] = []

        for i in range(len(sig_df)):
            row = sig_df.iloc[i]
            date = row["date"]
            close = row["close"]

            if pd.isna(close) or close <= 0:
                continue

            position_value = shares * close
            total_equity = cash + position_value

            signal_action = row.get("signal", "hold")
            signal_price = row.get("signal_price", close)
            signal_shares = int(row.get("signal_shares", 0) or 0)
            signal_reason = str(row.get("signal_reason", ""))

            # --- BUY ---
            if signal_action == "buy" and signal_shares > 0:
                requested = signal_price * signal_shares
                allowed, adj_amount, reason = self.risk.check_buy(
                    requested, position_value, total_equity, pe_value
                )

                if reason:
                    signals_log.append(Signal(
                        date=date, action="buy", price=signal_price,
                        shares=signal_shares, amount=requested, reason=reason,
                    ))

                if allowed:
                    adj_shares = max(1, int(signal_shares * adj_amount / requested)) if requested > 0 else signal_shares
                    fill_price, fill_shares, cost = self.broker.buy(
                        signal_price, adj_shares, cash
                    )
                    if fill_shares > 0:
                        cash -= cost
                        shares += fill_shares
                        cost_basis += cost

                        signals_log.append(Signal(
                            date=date, action="buy", price=fill_price,
                            shares=fill_shares, amount=cost,
                            reason=f"执行: {signal_reason}" if signal_reason else "执行买入",
                        ))

                        # Open a trade if none active
                        if open_trade is None:
                            open_trade = Trade(
                                entry_date=date, entry_price=fill_price,
                                shares=fill_shares, entry_amount=cost,
                            )
                        else:
                            # Averaging in — update cost basis on the open trade
                            total_shares = open_trade.shares + fill_shares
                            total_cost = open_trade.entry_amount + cost
                            open_trade.entry_price = total_cost / total_shares if total_shares > 0 else fill_price
                            open_trade.shares = total_shares
                            open_trade.entry_amount = total_cost

            # --- SELL ---
            elif signal_action == "sell" and signal_shares > 0 and shares > 0:
                signals_log.append(Signal(
                    date=date, action="sell", price=signal_price,
                    shares=signal_shares, amount=signal_price * signal_shares,
                    reason=signal_reason,
                ))

                fill_price, fill_shares, proceeds = self.broker.sell(
                    signal_price, signal_shares, shares
                )
                if fill_shares > 0:
                    cash += proceeds
                    shares -= fill_shares
                    # Proportionally reduce cost basis
                    if shares + fill_shares > 0:
                        cost_basis = cost_basis * shares / (shares + fill_shares)
                    else:
                        cost_basis = 0.0

                    # Close the open trade
                    if open_trade is not None:
                        avg_entry = open_trade.entry_amount / open_trade.shares if open_trade.shares > 0 else fill_price
                        open_trade.exit_date = date
                        open_trade.exit_price = fill_price
                        open_trade.exit_amount = proceeds
                        open_trade.pnl = proceeds - open_trade.entry_amount * (fill_shares / open_trade.shares)
                        open_trade.pnl_pct = open_trade.pnl / (open_trade.entry_amount * fill_shares / open_trade.shares) if open_trade.entry_amount > 0 else 0.0
                        open_trade.holding_days = (date - open_trade.entry_date).days
                        closed_trades.append(open_trade)
                        open_trade = None

            # --- HOLD ---
            else:
                if signal_action != "hold":
                    signals_log.append(Signal(
                        date=date, action="hold", price=close,
                        reason=f"忽略: {signal_reason}",
                    ))

            # --- Mark to market ---
            equity = cash + shares * close
            equity_rows.append({
                "date": date,
                "equity": equity,
                "cash": cash,
                "shares": shares,
                "position_value": shares * close,
            })

        # --- 3. Close any remaining open trade at final price ---
        if open_trade is not None and shares > 0:
            final_row = sig_df.iloc[-1]
            final_price = final_row["close"]
            open_trade.exit_date = final_row["date"]
            open_trade.exit_price = final_price
            unrealized = shares * final_price
            open_trade.exit_amount = unrealized
            cost_portion = open_trade.entry_amount * (shares / open_trade.shares) if open_trade.shares > 0 else open_trade.entry_amount
            open_trade.pnl = unrealized - cost_portion
            open_trade.pnl_pct = open_trade.pnl / cost_portion if cost_portion > 0 else 0.0
            open_trade.holding_days = (final_row["date"] - open_trade.entry_date).days
            closed_trades.append(open_trade)

        # --- 4. Compute metrics ---
        eq_df = pd.DataFrame(equity_rows)
        if eq_df.empty:
            return BacktestResult(
                initial_capital=self.initial_capital,
                strategy_name=strategy.name,
                strategy_params=params,
            )

        final_equity = eq_df["equity"].iloc[-1]
        metrics = compute_metrics(eq_df, closed_trades, self.initial_capital)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            annual_volatility=metrics["annual_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            calmar_ratio=metrics["calmar_ratio"],
            win_rate=metrics["win_rate"],
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            losing_trades=metrics["losing_trades"],
            avg_holding_days=metrics["avg_holding_days"],
            equity_curve=eq_df,
            trades=closed_trades,
            signals=signals_log,
            strategy_name=strategy.name,
            strategy_params=params,
        )
