"""Local paper-trading workflow driven only by project scan results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.engine.portfolio import LOT_SIZE, ExecutedTrade, PortfolioManager
from src.engine.rotation_scanner import RotationScanResult
from src.strategy.etf_rotation import (
    PositionState,
    RotationConfig,
    evaluate_exit,
)


PLAN_COLUMNS = [
    "code",
    "name",
    "action",
    "current_shares",
    "target_shares",
    "delta_shares",
    "reference_price",
    "target_weight",
    "estimated_amount",
    "reason",
]


@dataclass
class RebalancePlan:
    """A deterministic adjustment plan for one completed-daily-bar scan."""

    orders: pd.DataFrame
    errors: dict[str, str]
    as_of: date | None
    account_equity: float

    @property
    def actionable_count(self) -> int:
        if self.orders.empty:
            return 0
        return int(self.orders["action"].isin(["buy", "sell"]).sum())


@dataclass
class ExecutionResult:
    """Trades and explicit failures produced by a simulated execution."""

    trades: list[ExecutedTrade]
    errors: dict[str, str]


def _holding_days(
    history: pd.DataFrame | None,
    entry_date: str,
    as_of: date | None,
) -> int:
    """Count completed trading bars since entry, inclusive of the entry bar."""

    if history is None or history.empty or not entry_date or as_of is None:
        return 0
    if "date" not in history.columns:
        return 0
    dates = pd.to_datetime(history["date"], errors="coerce").dropna().dt.date
    entry = pd.to_datetime(entry_date, errors="coerce")
    if pd.isna(entry):
        return 0
    return int(((dates >= entry.date()) & (dates <= as_of)).sum())


def _safe_price(row: pd.Series | None) -> float:
    if row is None:
        return 0.0
    price = pd.to_numeric(row.get("close"), errors="coerce")
    return float(price) if pd.notna(price) and float(price) > 0 else 0.0


def update_portfolio_from_scan(
    portfolio: PortfolioManager,
    scan: RotationScanResult,
    config: RotationConfig | None = None,
) -> None:
    """Mark holdings and advance exit-confirmation counters once per data date."""

    config = config or RotationConfig()
    if scan.rankings.empty:
        return

    rows = {
        str(row["code"]): row
        for _, row in scan.rankings.iterrows()
    }
    prices = {
        code: price
        for code, row in rows.items()
        if (price := _safe_price(row)) > 0
    }
    portfolio.update_prices(prices)

    eligible_codes = [
        str(code)
        for code in scan.rankings.loc[scan.rankings["eligible"], "code"].tolist()
    ]
    ranks = {code: rank for rank, code in enumerate(eligible_codes, start=1)}
    signal_date = scan.as_of.isoformat() if scan.as_of else ""

    for code, holding in portfolio.holdings.items():
        row = rows.get(code)
        if row is None or not signal_date or holding.last_signal_date == signal_date:
            continue

        rank = ranks.get(code)
        if rank is None or rank > config.rank_exit:
            holding.rank_weak_days += 1
        else:
            holding.rank_weak_days = 0

        close = _safe_price(row)
        ma20 = pd.to_numeric(row.get("ma20"), errors="coerce")
        if close <= 0 or pd.isna(ma20) or close <= float(ma20):
            holding.trend_weak_days += 1
        else:
            holding.trend_weak_days = 0
        holding.last_signal_date = signal_date


def build_rebalance_plan(
    portfolio: PortfolioManager,
    scan: RotationScanResult,
    config: RotationConfig | None = None,
    trade_date: str | None = None,
    max_data_age_days: int = 7,
) -> RebalancePlan:
    """Compare target weights with the paper account and create lot-sized orders.

    A held symbol missing from the successful scan is frozen rather than sold.
    This prevents a network or data-source failure from becoming an exit signal.
    """

    config = config or RotationConfig()
    effective_trade_date = trade_date or date.today().isoformat()
    update_portfolio_from_scan(portfolio, scan, config)

    if scan.rankings.empty:
        return RebalancePlan(
            pd.DataFrame(columns=PLAN_COLUMNS),
            {"scan": "没有可用排名，禁止生成调仓指令"},
            scan.as_of,
            portfolio.total_equity,
        )

    ranking_rows = {
        str(row["code"]): row
        for _, row in scan.rankings.iterrows()
    }
    target_rows = {
        str(row["code"]): row
        for _, row in scan.targets.iterrows()
    }
    eligible_codes = [
        str(code)
        for code in scan.rankings.loc[scan.rankings["eligible"], "code"].tolist()
    ]
    ranks = {code: rank for rank, code in enumerate(eligible_codes, start=1)}
    account_equity = portfolio.total_equity
    errors: dict[str, str] = {}
    records: list[dict] = []

    all_codes = list(dict.fromkeys([*portfolio.holdings.keys(), *target_rows.keys()]))
    for code in all_codes:
        holding = portfolio.get_holding(code)
        ranking = ranking_rows.get(code)
        target = target_rows.get(code)
        if ranking is None:
            errors[code] = "持仓未包含在本次成功扫描结果中，已冻结自动交易"
            if holding is not None:
                records.append(
                    {
                        "code": code,
                        "name": holding.name,
                        "action": "hold",
                        "current_shares": holding.shares,
                        "target_shares": holding.shares,
                        "delta_shares": 0,
                        "reference_price": holding.current_price,
                        "target_weight": (
                            holding.market_value / account_equity
                            if account_equity > 0 else 0.0
                        ),
                        "estimated_amount": 0.0,
                        "reason": "数据缺失，禁止自动交易",
                    }
                )
            continue

        reference_price = _safe_price(ranking)
        if reference_price <= 0:
            errors[code] = "参考收盘价无效，已冻结自动交易"
            continue

        name = str(ranking.get("name") or (holding.name if holding else code))
        current_shares = holding.shares if holding else 0
        requested_weight = float(target.get("target_weight", 0.0)) if target is not None else 0.0
        target_shares = int(
            np.floor(account_equity * requested_weight / (reference_price * LOT_SIZE))
            * LOT_SIZE
        ) if account_equity > 0 else 0
        reason = "目标组合调仓"

        if holding is not None:
            holding_days = _holding_days(
                scan.histories.get(code),
                holding.entry_date,
                scan.as_of,
            )
            position = PositionState(
                code=code,
                entry_price=holding.avg_cost,
                highest_price=max(holding.highest_price, reference_price),
                holding_days=holding_days,
                rank_weak_days=holding.rank_weak_days,
                trend_weak_days=holding.trend_weak_days,
                closeable=portfolio.available_shares(code, effective_trade_date) > 0,
            )
            decision = evaluate_exit(
                position,
                ranking.to_dict(),
                reference_price,
                ranks.get(code),
                len(eligible_codes),
                config,
            )
            if decision.should_exit:
                target_shares = 0
                requested_weight = 0.0
                reason = decision.reason
            elif target is None:
                target_shares = current_shares
                requested_weight = (
                    current_shares * reference_price / account_equity
                    if account_equity > 0 else 0.0
                )
                reason = "未触发退出条件，继续持有"
            elif target_shares < current_shares and holding_days < config.min_hold_days:
                target_shares = current_shares
                reason = f"持有不足{config.min_hold_days}个交易日，暂不减仓"

        delta_shares = target_shares - current_shares
        if delta_shares < 0 and holding is not None:
            available = portfolio.available_shares(code, effective_trade_date)
            sell_shares = min(abs(delta_shares), available)
            delta_shares = -sell_shares
            target_shares = current_shares - sell_shares
            if sell_shares == 0:
                reason = "当日新买份额不可卖出"

        action = "buy" if delta_shares > 0 else "sell" if delta_shares < 0 else "hold"
        if action == "buy" and current_shares == 0:
            reason = "进入目标组合"
        elif action == "buy":
            reason = "目标权重上调"
        elif action == "sell" and reason == "目标组合调仓":
            reason = "目标权重下调"

        records.append(
            {
                "code": code,
                "name": name,
                "action": action,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "delta_shares": delta_shares,
                "reference_price": reference_price,
                "target_weight": requested_weight,
                "estimated_amount": abs(delta_shares) * reference_price,
                "reason": reason,
            }
        )

    orders = pd.DataFrame(records, columns=PLAN_COLUMNS)
    requested_count = scan.scanned_count + len(scan.errors)
    coverage = scan.scanned_count / requested_count if requested_count else 0.0
    data_date = pd.to_datetime(scan.as_of, errors="coerce")
    execution_date = pd.to_datetime(effective_trade_date, errors="coerce")
    data_age_days = (
        None
        if pd.isna(data_date) or pd.isna(execution_date)
        else (execution_date.date() - data_date.date()).days
    )
    stale = (
        pd.isna(data_date)
        or pd.isna(execution_date)
        or data_age_days < 0
        or data_age_days > max_data_age_days
    )
    if coverage < 0.80:
        errors["coverage"] = f"扫描成功率仅{coverage:.0%}，低于80%，已冻结自动交易"
    if stale:
        errors["freshness"] = "行情日期过旧或无效，已冻结自动交易"
    if (coverage < 0.80 or stale) and not orders.empty:
        actionable = orders["action"].isin(["buy", "sell"])
        orders.loc[actionable, "target_shares"] = orders.loc[actionable, "current_shares"]
        orders.loc[actionable, "delta_shares"] = 0
        orders.loc[actionable, "estimated_amount"] = 0.0
        orders.loc[actionable, "action"] = "hold"
        orders.loc[actionable, "reason"] = "数据质量门禁未通过，禁止自动交易"
    if not orders.empty:
        order_priority = {"sell": 0, "buy": 1, "hold": 2}
        orders["_priority"] = orders["action"].map(order_priority)
        orders = orders.sort_values(
            ["_priority", "estimated_amount"], ascending=[True, False]
        ).drop(columns="_priority").reset_index(drop=True)
    return RebalancePlan(orders, errors, scan.as_of, account_equity)


def execute_rebalance_plan(
    portfolio: PortfolioManager,
    plan: RebalancePlan,
    trade_date: str | None = None,
    slippage_pct: float = 0.001,
) -> ExecutionResult:
    """Execute all actionable paper orders, sells first and buys second."""

    effective_trade_date = trade_date or date.today().isoformat()
    trades: list[ExecutedTrade] = []
    errors: dict[str, str] = {}
    if plan.orders.empty:
        return ExecutionResult(trades, errors)

    actionable = plan.orders[plan.orders["action"].isin(["sell", "buy"])]
    for action in ("sell", "buy"):
        for _, order in actionable[actionable["action"] == action].iterrows():
            code = str(order["code"])
            reference_price = float(order["reference_price"])
            shares = abs(int(order["delta_shares"]))
            filled_price = reference_price * (
                1.0 - slippage_pct if action == "sell" else 1.0 + slippage_pct
            )
            if action == "sell":
                trade = portfolio.sell(
                    code,
                    filled_price,
                    shares,
                    name=str(order["name"]),
                    reason=str(order["reason"]),
                    trade_date=effective_trade_date,
                )
            else:
                trade = portfolio.buy(
                    code,
                    filled_price,
                    shares,
                    name=str(order["name"]),
                    reason=str(order["reason"]),
                    trade_date=effective_trade_date,
                )
            if trade is None:
                errors[code] = "可用现金、可卖份额或交易单位不足，未成交"
            else:
                trades.append(trade)
    return ExecutionResult(trades, errors)
