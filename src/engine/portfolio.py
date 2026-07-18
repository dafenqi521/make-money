"""Paper-trading portfolio manager for the ETF rotation strategy.

Tracks cash, holdings, and trade history.  Executes buy/sell based
on strategy signals and computes P&L.  Designed to be stored in
``st.session_state`` for persistence across Streamlit reruns.

A-share ETF rules:
  - Minimum trading unit: 100 shares (1 手 / 1 lot)
  - All orders rounded to whole lots

Usage::

    from src.engine.portfolio import PortfolioManager

    pm = PortfolioManager(initial_capital=100_000)
    trade = pm.buy("510300", price=4.829, shares=1000, reason="进入目标组合")
    trade = pm.sell("510300", price=5.100, shares=500, reason="趋势退出")
    print(pm.summary())
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOT_SIZE = 100  # A-share minimum trading unit (1手 = 100股)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Holding:
    """A single position in one ETF."""

    code: str
    name: str = ""
    shares: int = 0
    avg_cost: float = 0.0        # average cost per share
    total_cost: float = 0.0      # total amount spent (includes commission)
    current_price: float = 0.0   # latest market price
    entry_date: str = ""         # first entry date of the current position
    highest_price: float = 0.0   # high-water mark used by trailing exits
    last_buy_date: str = ""      # T+1 bookkeeping
    last_buy_shares: int = 0     # shares bought on ``last_buy_date``
    rank_weak_days: int = 0
    trend_weak_days: int = 0
    last_signal_date: str = ""

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.total_cost > 0:
            return self.unrealized_pnl / self.total_cost
        return 0.0

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name,
            "shares": self.shares, "avg_cost": round(self.avg_cost, 4),
            "total_cost": round(self.total_cost, 2),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct * 100, 2),
            "entry_date": self.entry_date,
            "highest_price": round(self.highest_price, 4),
        }


@dataclass
class ExecutedTrade:
    """A single completed buy or sell execution."""

    trade_id: str
    date: str                     # "YYYY-MM-DD"
    code: str
    name: str = ""
    action: str = ""              # "buy" | "sell"
    price: float = 0.0
    shares: int = 0
    amount: float = 0.0           # trade amount (price * shares)
    commission: float = 0.0
    net_amount: float = 0.0       # cash out (buy) or cash in (sell)
    pnl: Optional[float] = None   # only for sell
    pnl_pct: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id[:8],
            "日期": self.date,
            "代码": self.code,
            "名称": self.name,
            "操作": "买入" if self.action == "buy" else "卖出",
            "价格": round(self.price, 4),
            "股数": self.shares,
            "金额": round(abs(self.net_amount), 2),
            "手续费": round(self.commission, 2),
            "盈亏": (round(self.pnl, 2) if self.pnl is not None else "—"),
            "盈亏%": (f"{self.pnl_pct:+.2%}" if self.pnl_pct is not None else "—"),
            "原因": self.reason,
        }


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioManager:
    """Tracks a paper-trading portfolio with cash, holdings, and trade log.

    Designed to be stored in ``st.session_state["portfolio"]``.
    All mutations are explicit — no side effects.

    Commission model: 万三 (0.03%), min ¥5 per trade, no stamp duty on ETFs.
    Lot size: 100 shares (A-share minimum trading unit).
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission

        # code → Holding
        self._holdings: dict[str, Holding] = {}
        # All executed trades (buy + sell)
        self._trades: list[ExecutedTrade] = []
        # Total realized P&L from closed positions
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def round_lot(shares: int) -> int:
        """Round down to the nearest whole lot (100 shares)."""
        return (shares // LOT_SIZE) * LOT_SIZE

    @staticmethod
    def to_lots(shares: int) -> str:
        """Format shares as 'N手' for display."""
        lots = shares // LOT_SIZE
        return f"{lots}手" if lots > 0 else "0手"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def holdings(self) -> dict[str, Holding]:
        return self._holdings

    @property
    def trades(self) -> list[ExecutedTrade]:
        return list(self._trades)

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    def get_holding(self, code: str) -> Holding | None:
        return self._holdings.get(code)

    def available_shares(self, code: str, trade_date: str | None = None) -> int:
        """Return shares that can be sold under the unified T+1 rule."""

        holding = self._holdings.get(code)
        if holding is None:
            return 0
        effective_date = trade_date or date.today().isoformat()
        blocked = holding.last_buy_shares if holding.last_buy_date == effective_date else 0
        return self.round_lot(max(0, holding.shares - blocked))

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def buy(
        self,
        code: str,
        price: float,
        shares: int,
        name: str = "",
        reason: str = "",
        trade_date: str | None = None,
    ) -> ExecutedTrade | None:
        """Execute a buy order.

        Args:
            code: ETF ticker code (e.g. "510300").
            price: Execution price per share.
            shares: Number of shares to buy.
            name: ETF display name (optional).
            reason: Human-readable reason for the trade.
            trade_date: Override trade date (default: today).

        Returns:
            ExecutedTrade if successful, None if insufficient cash.
        """
        # Round to whole lots (A-share minimum = 100 shares)
        shares = self.round_lot(shares)
        if shares <= 0 or price <= 0:
            return None

        trade_amount = price * shares
        commission = max(self.min_commission, trade_amount * self.commission_rate)
        total_cost = trade_amount + commission

        if total_cost > self.cash:
            # Fill as many lots as affordable
            affordable_shares = int((self.cash - self.min_commission) / (price * (1 + self.commission_rate)))
            affordable_shares = self.round_lot(affordable_shares)
            if affordable_shares <= 0:
                return None
            shares = affordable_shares
            trade_amount = price * shares
            commission = max(self.min_commission, trade_amount * self.commission_rate)
            total_cost = trade_amount + commission

        if total_cost > self.cash or shares <= 0:
            return None

        effective_date = trade_date or date.today().isoformat()

        # Update cash
        self.cash -= total_cost

        # Update / create holding (average-cost method)
        if code in self._holdings:
            h = self._holdings[code]
            new_total_shares = h.shares + shares
            new_total_cost = h.total_cost + total_cost
            h.shares = new_total_shares
            h.total_cost = new_total_cost
            h.avg_cost = new_total_cost / new_total_shares if new_total_shares > 0 else 0.0
            h.current_price = price
            h.highest_price = max(h.highest_price, price)
            if h.last_buy_date == effective_date:
                h.last_buy_shares += shares
            else:
                h.last_buy_date = effective_date
                h.last_buy_shares = shares
            if name:
                h.name = name
        else:
            self._holdings[code] = Holding(
                code=code, name=name, shares=shares,
                avg_cost=total_cost / shares,
                total_cost=total_cost,
                current_price=price,
                entry_date=effective_date,
                highest_price=price,
                last_buy_date=effective_date,
                last_buy_shares=shares,
            )

        # Record trade
        trade = ExecutedTrade(
            trade_id=uuid.uuid4().hex,
            date=effective_date,
            code=code, name=name, action="buy",
            price=price, shares=shares,
            amount=trade_amount, commission=commission,
            net_amount=-total_cost,
            reason=reason,
        )
        self._trades.append(trade)
        return trade

    def sell(
        self,
        code: str,
        price: float,
        shares: int,
        name: str = "",
        reason: str = "",
        trade_date: str | None = None,
    ) -> ExecutedTrade | None:
        """Execute a sell order.

        Args:
            code: ETF ticker code.
            price: Execution price per share.
            shares: Number of shares to sell.
            name: ETF display name (optional).
            reason: Human-readable reason.
            trade_date: Override trade date (default: today).

        Returns:
            ExecutedTrade with P&L info, or None if insufficient shares.
        """
        # Round to whole lots
        shares = self.round_lot(shares)
        if shares <= 0 or price <= 0:
            return None

        holding = self._holdings.get(code)
        if holding is None or holding.shares <= 0:
            return None

        effective_date = trade_date or date.today().isoformat()
        shares = min(shares, self.available_shares(code, effective_date))
        if shares <= 0:
            return None

        # Compute cost of the sold portion (proportional)
        cost_portion = holding.total_cost * (shares / holding.shares)
        trade_amount = price * shares
        commission = max(self.min_commission, trade_amount * self.commission_rate)
        net_proceeds = trade_amount - commission

        # Update cash
        self.cash += net_proceeds

        # Update holding
        pnl = net_proceeds - cost_portion
        pnl_pct = pnl / cost_portion if cost_portion > 0 else 0.0
        self._realized_pnl += pnl

        holding.shares -= shares
        holding.total_cost -= cost_portion
        if holding.shares > 0:
            holding.avg_cost = holding.total_cost / holding.shares
            holding.current_price = price
        else:
            del self._holdings[code]
        if name:
            holding.name = name

        # Record trade
        trade = ExecutedTrade(
            trade_id=uuid.uuid4().hex,
            date=effective_date,
            code=code, name=name or (holding.name if holding else ""),
            action="sell", price=price, shares=shares,
            amount=trade_amount, commission=commission,
            net_amount=net_proceeds,
            pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 4),
            reason=reason,
        )
        self._trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current market prices for all holdings."""
        for code, price in prices.items():
            if code in self._holdings and price > 0:
                self._holdings[code].current_price = price
                self._holdings[code].highest_price = max(
                    self._holdings[code].highest_price,
                    price,
                )

    @property
    def total_market_value(self) -> float:
        return sum(h.market_value for h in self._holdings.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.total_market_value

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(h.unrealized_pnl for h in self._holdings.values())

    @property
    def total_pnl(self) -> float:
        """Total P&L = realized + unrealized."""
        return self._realized_pnl + self.total_unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital > 0:
            return self.total_pnl / self.initial_capital
        return 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict with all key portfolio metrics."""
        return {
            "initial_capital": self.initial_capital,
            "cash": round(self.cash, 2),
            "market_value": round(self.total_market_value, 2),
            "total_equity": round(self.total_equity, 2),
            "realized_pnl": round(self._realized_pnl, 2),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "positions": len(self._holdings),
            "total_trades": len(self._trades),
            "buy_trades": sum(1 for t in self._trades if t.action == "buy"),
            "sell_trades": sum(1 for t in self._trades if t.action == "sell"),
        }

    def get_trade_history(self, n: int = 50) -> list[dict]:
        """Return the most recent N trades as dicts (most recent first)."""
        trades = list(reversed(self._trades))[:n]
        return [t.to_dict() for t in trades]

    def get_holdings_table(self) -> list[dict]:
        """Return current holdings as a list of dicts."""
        return [h.to_dict() for h in self._holdings.values() if h.shares > 0]

    def to_dict(self) -> dict:
        """Serialize to dict for session_state persistence."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "_realized_pnl": self._realized_pnl,
            "_holdings": {
                code: {
                    "shares": h.shares,
                    "avg_cost": h.avg_cost,
                    "total_cost": h.total_cost,
                    "current_price": h.current_price,
                    "name": h.name,
                    "entry_date": h.entry_date,
                    "highest_price": h.highest_price,
                    "last_buy_date": h.last_buy_date,
                    "last_buy_shares": h.last_buy_shares,
                    "rank_weak_days": h.rank_weak_days,
                    "trend_weak_days": h.trend_weak_days,
                    "last_signal_date": h.last_signal_date,
                }
                for code, h in self._holdings.items()
            },
            "_trades": [
                {
                    "trade_id": t.trade_id, "date": t.date,
                    "code": t.code, "name": t.name,
                    "action": t.action, "price": t.price,
                    "shares": t.shares, "amount": t.amount,
                    "commission": t.commission, "net_amount": t.net_amount,
                    "pnl": t.pnl, "pnl_pct": t.pnl_pct, "reason": t.reason,
                }
                for t in self._trades
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioManager":
        """Restore from a serialized dict."""
        pm = cls(
            initial_capital=d["initial_capital"],
            commission_rate=d.get("commission_rate", 0.0003),
            min_commission=d.get("min_commission", 5.0),
        )
        pm.cash = d["cash"]
        pm._realized_pnl = d.get("_realized_pnl", 0.0)

        for code, hd in d.get("_holdings", {}).items():
            pm._holdings[code] = Holding(
                code=code, name=hd.get("name", ""),
                shares=hd["shares"], avg_cost=hd["avg_cost"],
                total_cost=hd["total_cost"],
                current_price=hd.get("current_price", 0.0),
                entry_date=hd.get("entry_date", ""),
                highest_price=hd.get("highest_price", hd.get("current_price", 0.0)),
                last_buy_date=hd.get("last_buy_date", ""),
                last_buy_shares=hd.get("last_buy_shares", 0),
                rank_weak_days=hd.get("rank_weak_days", 0),
                trend_weak_days=hd.get("trend_weak_days", 0),
                last_signal_date=hd.get("last_signal_date", ""),
            )

        for td in d.get("_trades", []):
            pm._trades.append(ExecutedTrade(
                trade_id=td["trade_id"], date=td["date"],
                code=td["code"], name=td.get("name", ""),
                action=td["action"], price=td["price"],
                shares=td["shares"], amount=td["amount"],
                commission=td["commission"], net_amount=td["net_amount"],
                pnl=td.get("pnl"), pnl_pct=td.get("pnl_pct"),
                reason=td.get("reason", ""),
            ))

        return pm
