"""Simulated broker — commission, slippage, and order execution.

Models Chinese ETF trading costs:
  - No stamp duty on ETFs (免征印花税)
  - Commission: 0.03% (万三), minimum 5 CNY per trade
  - Slippage: 0.1% adverse price movement
"""

from __future__ import annotations


class Broker:
    """Simulates order execution with realistic costs."""

    def __init__(
        self,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        slippage_pct: float = 0.001,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_pct = slippage_pct

    # ------------------------------------------------------------------
    # Buy
    # ------------------------------------------------------------------

    def buy(
        self, price: float, requested_shares: int, cash_available: float
    ) -> tuple[float, int, float]:
        """Execute a buy order.

        Args:
            price: Current market price (before slippage).
            requested_shares: Desired number of shares to buy.
            cash_available: Available cash in the account.

        Returns:
            (filled_price, filled_shares, total_cost)
            filled_price is the execution price WITH slippage.
            total_cost includes commission.

        If cash is insufficient, fills as many shares as possible.
        """
        if requested_shares <= 0 or cash_available <= 0 or price <= 0:
            return (price, 0, 0.0)

        # Apply slippage — buy fills at a slightly higher price
        filled_price = price * (1.0 + self.slippage_pct)

        # How many shares can we actually afford?
        raw_cost_per_share = filled_price
        max_affordable = int(cash_available / raw_cost_per_share)
        filled_shares = min(requested_shares, max_affordable)

        if filled_shares <= 0:
            return (filled_price, 0, 0.0)

        # Compute cost with commission
        trade_amount = filled_price * filled_shares
        commission = max(self.min_commission, trade_amount * self.commission_rate)
        total_cost = trade_amount + commission

        # Final check — if commission pushes us over, reduce by 1 share
        if total_cost > cash_available and filled_shares > 1:
            filled_shares -= 1
            trade_amount = filled_price * filled_shares
            commission = max(self.min_commission, trade_amount * self.commission_rate)
            total_cost = trade_amount + commission

        if total_cost > cash_available:
            return (filled_price, 0, 0.0)

        return (filled_price, filled_shares, total_cost)

    # ------------------------------------------------------------------
    # Sell
    # ------------------------------------------------------------------

    def sell(
        self, price: float, requested_shares: int, shares_held: int
    ) -> tuple[float, int, float]:
        """Execute a sell order.

        Args:
            price: Current market price (before slippage).
            requested_shares: Desired number of shares to sell.
            shares_held: Shares currently held.

        Returns:
            (filled_price, filled_shares, net_proceeds)
            net_proceeds is AFTER commission deduction.

        Cannot sell more than shares_held.
        """
        if requested_shares <= 0 or shares_held <= 0 or price <= 0:
            return (price, 0, 0.0)

        # Apply slippage — sell fills at a slightly lower price
        filled_price = price * (1.0 - self.slippage_pct)

        filled_shares = min(requested_shares, shares_held)

        trade_amount = filled_price * filled_shares
        commission = max(self.min_commission, trade_amount * self.commission_rate)
        net_proceeds = trade_amount - commission

        if net_proceeds < 0:
            return (filled_price, 0, 0.0)

        return (filled_price, filled_shares, net_proceeds)
