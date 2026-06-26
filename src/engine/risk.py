"""Risk management — position limits, PE filters, cash reserves.

All checks are pure logic — no side effects. The engine calls these
before executing any trade.
"""

from __future__ import annotations


class RiskManager:
    """Enforces risk control rules for ETF trading."""

    def __init__(
        self,
        max_position_pct: float = 0.20,
        cash_reserve_pct: float = 0.30,
        pe_warning_threshold: float = 30.0,
    ):
        self.max_position_pct = max_position_pct
        self.cash_reserve_pct = cash_reserve_pct
        self.pe_warning_threshold = pe_warning_threshold

    # ------------------------------------------------------------------
    # Buy-side checks
    # ------------------------------------------------------------------

    def check_buy(
        self,
        requested_amount: float,
        current_position_value: float,
        total_equity: float,
        pe_value: float | None = None,
    ) -> tuple[bool, float, str]:
        """Validate a buy signal against risk rules.

        Args:
            requested_amount: Cost of the proposed buy (before commission).
            current_position_value: Current market value of holdings.
            total_equity: cash + position_value.
            pe_value: Current PE(TTM) value, or None if unavailable.

        Returns:
            (allowed, adjusted_amount, reason)
        """
        if total_equity <= 0:
            return (False, 0.0, "总资产为0")

        # 1. PE filter — suppress buy when valuation is stretched
        if pe_value is not None and pe_value > self.pe_warning_threshold:
            return (
                False,
                0.0,
                f"PE({pe_value:.1f}) > 阈值({self.pe_warning_threshold})，抑制买入",
            )

        # 2. Position limit — single position ≤ max_position_pct of equity
        max_position_value = total_equity * self.max_position_pct
        new_position_value = current_position_value + requested_amount
        if new_position_value > max_position_value:
            allowed_amount = max(0.0, max_position_value - current_position_value)
            if allowed_amount <= 0:
                return (
                    False,
                    0.0,
                    f"仓位已达上限 {self.max_position_pct:.0%}",
                )
            return (
                True,
                allowed_amount,
                f"仓位限制: 调整至 {allowed_amount:.0f}",
            )

        # 3. Cash reserve — must keep cash_reserve_pct of capital in cash
        available_cash = total_equity - current_position_value
        min_cash = total_equity * self.cash_reserve_pct
        max_spendable = available_cash - min_cash
        if requested_amount > max_spendable:
            allowed_amount = max(0.0, max_spendable)
            if allowed_amount <= 0:
                return (
                    False,
                    0.0,
                    f"现金储备不足 {self.cash_reserve_pct:.0%}",
                )
            return (
                True,
                allowed_amount,
                f"现金储备限制: 调整至 {allowed_amount:.0f}",
            )

        return (True, requested_amount, "")

    # ------------------------------------------------------------------
    # Warnings (non-blocking)
    # ------------------------------------------------------------------

    def check_liquidation(
        self, current_price: float, grid_lower_bound: float
    ) -> tuple[bool, str]:
        """Warn if price has fallen significantly below the grid floor.

        Returns:
            (should_warn, message)
        """
        if grid_lower_bound <= 0:
            return (False, "")
        if current_price < grid_lower_bound * 0.9:
            return (
                True,
                f"⚠️ 当前价 {current_price:.3f} 跌破网格下限 {grid_lower_bound:.3f} 10%，建议清仓",
            )
        return (False, "")

    def check_step_size(
        self, grid_step: float, current_price: float
    ) -> tuple[bool, str]:
        """Warn if grid step size is too small (< 1% of price)."""
        if current_price <= 0:
            return (False, "")
        step_pct = grid_step / current_price
        if step_pct < 0.01:
            return (
                True,
                f"⚠️ 网格步长 {grid_step:.4f} ({step_pct:.2%}) 不足1%%，交易成本可能侵蚀利润",
            )
        return (False, "")
