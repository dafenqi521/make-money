"""Look-ahead-safe historical backtest for the ETF rotation strategy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_etf_hist
from src.engine.paper_trading import (
    RebalancePlan,
    build_rebalance_plan,
    execute_rebalance_plan,
)
from src.engine.portfolio import LOT_SIZE, PortfolioManager
from src.engine.rotation_scanner import RotationScanResult, normalise_pool
from src.strategy.etf_rotation import RotationConfig, rank_etfs, select_targets


@dataclass(frozen=True)
class BacktestSettings:
    """Execution and reporting settings for one historical simulation."""

    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.001
    signal_frequency: str = "daily"
    benchmark_code: str = "510300"
    risk_free_rate: float = 0.03

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.slippage_pct < 0 or self.slippage_pct > 0.05:
            raise ValueError("slippage_pct must be between 0 and 5%")
        if self.signal_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("unsupported signal_frequency")


@dataclass
class HistoricalDataResult:
    """Downloaded and normalised histories plus explicit failures."""

    histories: dict[str, pd.DataFrame]
    metadata: dict[str, dict]
    errors: dict[str, str]
    requested_count: int

    @property
    def coverage(self) -> float:
        if self.requested_count <= 0:
            return 0.0
        return len(self.histories) / self.requested_count


@dataclass
class BacktestResult:
    """Complete result used by tests and the Streamlit dashboard."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    monthly_returns: pd.DataFrame
    signal_log: pd.DataFrame
    metrics: dict[str, float | int]
    benchmark_metrics: dict[str, float | int]
    data_errors: dict[str, str]
    requested_count: int
    benchmark_code: str

    @property
    def coverage(self) -> float:
        successful = self.requested_count - len(self.data_errors)
        source_coverage = (
            successful / self.requested_count if self.requested_count else 0.0
        )
        if self.signal_log.empty or "coverage" not in self.signal_log:
            return source_coverage
        return min(source_coverage, float(self.signal_log["coverage"].min()))


def _normalise_market_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        raise ValueError("历史行情为空")
    if "date" not in history.columns or "close" not in history.columns:
        raise ValueError("历史行情缺少date或close列")
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "money", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0]
    if "open" not in frame.columns:
        frame["open"] = frame["close"]
    frame["open"] = frame["open"].fillna(frame["close"])
    if "high" not in frame.columns:
        frame["high"] = frame[["open", "close"]].max(axis=1)
    if "low" not in frame.columns:
        frame["low"] = frame[["open", "close"]].min(axis=1)
    frame["high"] = frame["high"].fillna(frame[["open", "close"]].max(axis=1))
    frame["low"] = frame["low"].fillna(frame[["open", "close"]].min(axis=1))
    return (
        frame.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )


def fetch_backtest_histories(
    pool: Iterable[dict | str],
    start_date: date,
    end_date: date,
    history_fetcher: Callable[..., pd.DataFrame] = fetch_etf_hist,
    max_workers: int = 8,
    warmup_calendar_days: int = 260,
) -> HistoricalDataResult:
    """Fetch a bounded window including enough pre-start indicator warmup."""

    entries = normalise_pool(pool)
    histories: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    metadata = {
        entry["code"]: {
            "name": entry["name"],
            "category": entry["category"],
        }
        for entry in entries
    }
    fetch_start = start_date - timedelta(days=warmup_calendar_days)

    def fetch_one(code: str) -> pd.DataFrame:
        return history_fetcher(
            code,
            start_date=fetch_start.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )

    if entries:
        worker_count = max(1, min(max_workers, len(entries)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(fetch_one, entry["code"]): entry["code"]
                for entry in entries
            }
            for future in as_completed(futures):
                code = futures[future]
                try:
                    frame = _normalise_market_history(future.result())
                    frame = frame[
                        (frame["date"].dt.date >= fetch_start)
                        & (frame["date"].dt.date <= end_date)
                    ].reset_index(drop=True)
                    if frame.empty:
                        raise ValueError("指定区间没有历史行情")
                    histories[code] = frame
                except Exception as error:
                    errors[code] = str(error)

    return HistoricalDataResult(histories, metadata, errors, len(entries))


def _signal_due(index: int, trading_dates: list[pd.Timestamp], frequency: str) -> bool:
    if index >= len(trading_dates) - 1:
        return False
    if frequency == "daily":
        return True
    current = trading_dates[index]
    following = trading_dates[index + 1]
    if frequency == "weekly":
        return current.isocalendar()[:2] != following.isocalendar()[:2]
    return (current.year, current.month) != (following.year, following.month)


def _scan_at_close(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, object]],
    signal_date: pd.Timestamp,
    config: RotationConfig,
) -> RotationScanResult:
    available: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    dated_metadata: dict[str, dict] = {}
    for code in metadata:
        history = histories.get(code)
        if history is None:
            errors[code] = "历史行情读取失败"
            continue
        sliced = history[history["date"] <= signal_date]
        if sliced.empty or sliced["date"].iloc[-1] != signal_date:
            errors[code] = "信号日行情缺失"
            continue
        available[code] = sliced.reset_index(drop=True)
        first_date = sliced["date"].iloc[0]
        dated_metadata[code] = {
            **metadata.get(code, {}),
            "listed_days": max(0, (signal_date - first_date).days),
        }
    rankings = (
        rank_etfs(available, dated_metadata, config) if available else pd.DataFrame()
    )
    targets = (
        select_targets(rankings, available, config)
        if not rankings.empty
        else pd.DataFrame()
    )
    return RotationScanResult(
        rankings=rankings,
        targets=targets,
        histories=available,
        errors=errors,
        as_of=signal_date.date(),
    )


def _prices_on(
    histories: Mapping[str, pd.DataFrame],
    trading_date: pd.Timestamp,
    column: str,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for code, history in histories.items():
        row = history.loc[history["date"] == trading_date]
        if row.empty:
            continue
        value = pd.to_numeric(row.iloc[-1].get(column), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            prices[code] = float(value)
    return prices


def _reprice_plan_at_open(
    plan: RebalancePlan,
    portfolio: PortfolioManager,
    open_prices: Mapping[str, float],
    execution_date: str,
) -> RebalancePlan:
    """Use next-session opens for sizing/fills without changing prior signals."""

    orders = plan.orders.copy()
    errors = dict(plan.errors)
    portfolio.update_prices(
        {code: price for code, price in open_prices.items() if code in portfolio.holdings}
    )
    equity_at_open = portfolio.total_equity
    for index, order in orders.iterrows():
        original_action = str(order["action"])
        if original_action not in {"buy", "sell"}:
            continue
        code = str(order["code"])
        price = float(open_prices.get(code, 0.0))
        if price <= 0:
            orders.loc[index, ["action", "delta_shares", "estimated_amount"]] = [
                "hold",
                0,
                0.0,
            ]
            orders.loc[index, "target_shares"] = int(order["current_shares"])
            orders.loc[index, "reason"] = "次日开盘价缺失，未模拟成交"
            errors[code] = "次日开盘价缺失"
            continue

        current_shares = (
            portfolio.holdings[code].shares if code in portfolio.holdings else 0
        )
        target_weight = max(0.0, float(order["target_weight"]))
        target_shares = int(
            np.floor(equity_at_open * target_weight / (price * LOT_SIZE)) * LOT_SIZE
        )
        if target_weight == 0:
            target_shares = 0
        delta = target_shares - current_shares
        if original_action == "buy" and delta <= 0:
            delta = 0
            target_shares = current_shares
        elif original_action == "sell" and delta >= 0:
            delta = 0
            target_shares = current_shares
        elif delta < 0:
            available = portfolio.available_shares(code, execution_date)
            delta = -min(abs(delta), available)
            target_shares = current_shares + delta

        action = "buy" if delta > 0 else "sell" if delta < 0 else "hold"
        orders.loc[index, "reference_price"] = price
        orders.loc[index, "current_shares"] = current_shares
        orders.loc[index, "target_shares"] = target_shares
        orders.loc[index, "delta_shares"] = delta
        orders.loc[index, "estimated_amount"] = abs(delta) * price
        orders.loc[index, "action"] = action
    return RebalancePlan(orders, errors, plan.as_of, equity_at_open)


def _performance_metrics(
    equity_curve: pd.DataFrame,
    initial_capital: float,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = 0.03,
) -> dict[str, float | int]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "annual_turnover": 0.0,
            "average_exposure": 0.0,
        }
    equity = equity_curve.set_index("date")["equity"].astype(float)
    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    elapsed_days = max(1, (equity.index[-1] - equity.index[0]).days)
    years = elapsed_days / 365.25
    annual_return = (
        float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
        if years > 0 and equity.iloc[-1] > 0
        else 0.0
    )
    annual_volatility = (
        float(daily_returns.std(ddof=0) * np.sqrt(252))
        if len(daily_returns) >= 2
        else 0.0
    )
    annualized_mean = float(daily_returns.mean() * 252) if len(daily_returns) else 0.0
    sharpe = (
        (annualized_mean - risk_free_rate) / annual_volatility
        if annual_volatility > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    trade_count = 0 if trades is None else len(trades)
    win_rate = 0.0
    annual_turnover = 0.0
    if trades is not None and not trades.empty:
        sells = trades[(trades["action"] == "sell") & trades["pnl"].notna()]
        win_rate = float((sells["pnl"] > 0).mean()) if not sells.empty else 0.0
        years_observed = max(len(daily_returns) / 252.0, 1 / 252.0)
        annual_turnover = float(
            trades["amount"].abs().sum() / max(equity.mean(), 1.0) / years_observed
        )
    exposure = (
        float(equity_curve["exposure"].mean())
        if "exposure" in equity_curve and not equity_curve.empty
        else 1.0
    )
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "trade_count": int(trade_count),
        "win_rate": win_rate,
        "annual_turnover": annual_turnover,
        "average_exposure": exposure,
    }


def _monthly_returns(curve: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame(columns=["month", "strategy_return", "benchmark_return"])
    indexed = curve.set_index("date")
    monthly = indexed[["equity", "benchmark_equity"]].resample("ME").last()
    result = monthly.pct_change()
    result.iloc[0, result.columns.get_loc("equity")] = (
        monthly["equity"].iloc[0] / initial_capital - 1.0
    )
    result.iloc[0, result.columns.get_loc("benchmark_equity")] = (
        monthly["benchmark_equity"].iloc[0] / initial_capital - 1.0
    )
    return result.rename(
        columns={"equity": "strategy_return", "benchmark_equity": "benchmark_return"}
    ).reset_index(names="month")


def run_rotation_backtest(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, object]],
    rotation_config: RotationConfig,
    settings: BacktestSettings,
    data_errors: Mapping[str, str] | None = None,
    requested_count: int | None = None,
) -> BacktestResult:
    """Run close-signal/next-open execution with daily mark-to-market."""

    normalised = {
        code: _normalise_market_history(history)
        for code, history in histories.items()
    }
    errors = dict(data_errors or {})
    requested = requested_count if requested_count is not None else len(normalised) + len(errors)
    calendar_values = sorted(
        {
            timestamp
            for history in normalised.values()
            for timestamp in history["date"]
            if settings.start_date <= timestamp.date() <= settings.end_date
        }
    )
    trading_dates = [pd.Timestamp(value).normalize() for value in calendar_values]
    empty_columns = [
        "date",
        "equity",
        "cash",
        "market_value",
        "exposure",
        "benchmark_equity",
        "strategy_return",
        "benchmark_return",
        "drawdown",
        "benchmark_drawdown",
    ]
    if len(trading_dates) < 2:
        empty = pd.DataFrame(columns=empty_columns)
        return BacktestResult(
            empty,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            _performance_metrics(empty, settings.initial_capital),
            _performance_metrics(empty, settings.initial_capital),
            errors,
            requested,
            settings.benchmark_code,
        )

    benchmark_code = None
    candidates = [settings.benchmark_code, *normalised.keys()]
    for candidate in dict.fromkeys(candidates):
        history = normalised.get(candidate)
        if history is None:
            continue
        in_period = history[
            (history["date"].dt.date >= settings.start_date)
            & (history["date"].dt.date <= settings.end_date)
        ]
        if not in_period.empty:
            benchmark_code = candidate
            break
    if benchmark_code is None:
        empty = pd.DataFrame(columns=empty_columns)
        return BacktestResult(
            empty,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            _performance_metrics(empty, settings.initial_capital),
            _performance_metrics(empty, settings.initial_capital),
            errors,
            requested,
            settings.benchmark_code,
        )
    if benchmark_code != settings.benchmark_code:
        errors[settings.benchmark_code] = f"基准不可用，改用{benchmark_code}"
    benchmark = normalised[benchmark_code].set_index("date")["close"].reindex(
        trading_dates
    ).ffill()
    first_benchmark_date = benchmark.first_valid_index()
    trading_dates = [value for value in trading_dates if value >= first_benchmark_date]
    if len(trading_dates) < 2:
        empty = pd.DataFrame(columns=empty_columns)
        return BacktestResult(
            empty,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            _performance_metrics(empty, settings.initial_capital),
            _performance_metrics(empty, settings.initial_capital),
            errors,
            requested,
            benchmark_code,
        )
    benchmark = benchmark.reindex(trading_dates).ffill()
    benchmark_start = float(benchmark.iloc[0])

    portfolio = PortfolioManager(initial_capital=settings.initial_capital)
    curve_rows: list[dict] = []
    signal_rows: list[dict] = []
    pending_plan: RebalancePlan | None = None

    for index, trading_date in enumerate(trading_dates):
        execution_date = trading_date.date().isoformat()
        open_prices = _prices_on(normalised, trading_date, "open")
        if pending_plan is not None:
            execution_plan = _reprice_plan_at_open(
                pending_plan,
                portfolio,
                open_prices,
                execution_date,
            )
            execute_rebalance_plan(
                portfolio,
                execution_plan,
                trade_date=execution_date,
                slippage_pct=settings.slippage_pct,
            )
            pending_plan = None

        close_prices = _prices_on(normalised, trading_date, "close")
        portfolio.update_prices(close_prices)
        equity = portfolio.total_equity
        market_value = portfolio.total_market_value
        benchmark_price = float(benchmark.loc[trading_date])
        benchmark_equity = settings.initial_capital * benchmark_price / benchmark_start
        curve_rows.append(
            {
                "date": trading_date,
                "equity": equity,
                "cash": portfolio.cash,
                "market_value": market_value,
                "exposure": market_value / equity if equity > 0 else 0.0,
                "benchmark_equity": benchmark_equity,
            }
        )

        if _signal_due(index, trading_dates, settings.signal_frequency):
            scan = _scan_at_close(normalised, metadata, trading_date, rotation_config)
            next_date = trading_dates[index + 1]
            plan = build_rebalance_plan(
                portfolio,
                scan,
                rotation_config,
                trade_date=next_date.date().isoformat(),
            )
            pending_plan = plan
            total = scan.scanned_count + len(scan.errors)
            coverage = scan.scanned_count / total if total else 0.0
            signal_rows.append(
                {
                    "signal_date": trading_date,
                    "execution_date": next_date,
                    "coverage": coverage,
                    "eligible_count": scan.eligible_count,
                    "target_count": len(scan.targets),
                    "actionable_count": plan.actionable_count,
                    "frozen": bool("coverage" in plan.errors or "freshness" in plan.errors),
                }
            )

    curve = pd.DataFrame(curve_rows)
    curve["strategy_return"] = curve["equity"] / settings.initial_capital - 1.0
    curve["benchmark_return"] = (
        curve["benchmark_equity"] / settings.initial_capital - 1.0
    )
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    curve["benchmark_drawdown"] = (
        curve["benchmark_equity"] / curve["benchmark_equity"].cummax() - 1.0
    )
    trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(trade.date),
                "code": trade.code,
                "name": trade.name,
                "action": trade.action,
                "price": trade.price,
                "shares": trade.shares,
                "amount": trade.amount,
                "commission": trade.commission,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "reason": trade.reason,
            }
            for trade in portfolio.trades
        ]
    )
    metrics = _performance_metrics(
        curve,
        settings.initial_capital,
        trades,
        settings.risk_free_rate,
    )
    benchmark_curve = curve[["date", "benchmark_equity"]].rename(
        columns={"benchmark_equity": "equity"}
    )
    benchmark_curve["exposure"] = 1.0
    benchmark_metrics = _performance_metrics(
        benchmark_curve,
        settings.initial_capital,
        None,
        settings.risk_free_rate,
    )
    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        monthly_returns=_monthly_returns(curve, settings.initial_capital),
        signal_log=pd.DataFrame(signal_rows),
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        data_errors=errors,
        requested_count=requested,
        benchmark_code=benchmark_code,
    )


def run_parameter_sweep(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, object]],
    base_config: RotationConfig,
    settings: BacktestSettings,
    max_positions_values: Iterable[int] = (2, 3, 4, 5),
) -> pd.DataFrame:
    """One-dimensional robustness check that avoids combinatorial tuning."""

    rows = []
    for max_positions in dict.fromkeys(int(value) for value in max_positions_values):
        config = replace(base_config, max_positions=max_positions)
        result = run_rotation_backtest(histories, metadata, config, settings)
        rows.append(
            {
                "max_positions": max_positions,
                "total_return": result.metrics["total_return"],
                "annual_return": result.metrics["annual_return"],
                "max_drawdown": result.metrics["max_drawdown"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
                "trade_count": result.metrics["trade_count"],
                "annual_turnover": result.metrics["annual_turnover"],
            }
        )
    return pd.DataFrame(rows).sort_values("max_positions").reset_index(drop=True)
