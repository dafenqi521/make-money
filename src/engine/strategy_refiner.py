"""Strategy parameter refinement based on completed trade cycle analysis.

Analyzes closed buy→sell cycles to recommend (or auto-apply) parameter
adjustments. Designed to work with the AutoTrader but usable standalone.

Usage::

    from src.engine.strategy_refiner import CycleAnalyzer

    analyzer = CycleAnalyzer()
    stats = analyzer.compute_stats()
    print(f"Win rate: {stats.win_rate:.0%}, PF: {stats.profit_factor:.2f}")

    changes = analyzer.recommend_changes(current_params=strategy.get_default_params())
    for c in changes:
        print(f"  {c.parameter}: {c.current_value} → {c.proposed_value} [{c.confidence}]")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.portfolio_db import PortfolioDB

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TradeCycle:
    """One complete buy→sell round trip."""

    cycle_id: str
    code: str
    name: str
    entry_date: str
    exit_date: str | None = None
    entry_price: float = 0.0
    exit_price: float | None = None
    shares: int = 0
    entry_amount: float = 0.0
    pnl: float | None = None
    pnl_pct: float | None = None
    holding_days: int | None = None
    entry_reason: str = ""
    exit_reason: str = ""
    is_closed: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TradeCycle":
        return cls(
            cycle_id=d.get("cycle_id", ""),
            code=d.get("code", ""),
            name=d.get("name", ""),
            entry_date=d.get("entry_date", ""),
            exit_date=d.get("exit_date"),
            entry_price=float(d.get("entry_price", 0)),
            exit_price=float(d["exit_price"]) if d.get("exit_price") else None,
            shares=int(d.get("shares", 0)),
            entry_amount=float(d.get("entry_amount", 0)),
            pnl=float(d["pnl"]) if d.get("pnl") is not None else None,
            pnl_pct=float(d["pnl_pct"]) if d.get("pnl_pct") is not None else None,
            holding_days=int(d["holding_days"]) if d.get("holding_days") else None,
            entry_reason=d.get("entry_reason", ""),
            exit_reason=d.get("exit_reason", ""),
            is_closed=bool(d.get("is_closed", False)),
        )


@dataclass
class CycleStats:
    """Aggregated statistics across trade cycles."""

    total_cycles: int = 0
    closed_cycles: int = 0
    open_cycles: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_days: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    best_cycle: dict = field(default_factory=dict)
    worst_cycle: dict = field(default_factory=dict)
    exit_reasons: dict[str, int] = field(default_factory=dict)
    by_code: dict[str, dict] = field(default_factory=dict)
    by_hold_days: dict[int, dict] = field(default_factory=dict)


@dataclass
class ParameterChange:
    """A recommended parameter adjustment with justification."""

    parameter: str
    current_value: float
    proposed_value: float
    reason: str
    confidence: str = "medium"  # "high" | "medium" | "low"
    supporting_data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "stop_loss_pct": (0.01, 0.05),
    "take_profit_pct": (0.02, 0.10),
    "max_hold_days": (2, 7),
    "entry_score_threshold": (30, 70),
    "min_prev_day_change": (0.01, 0.04),
    "trailing_giveback_pct": (0.005, 0.03),
    "trailing_activate_pct": (0.01, 0.04),
    "partial_take_profit_pct": (0.02, 0.06),
    "position_pct": (0.10, 0.35),
    "max_gap_down_pct": (0.01, 0.05),
}

# Parameter step sizes (increment for each adjustment)
PARAM_STEPS: dict[str, float] = {
    "stop_loss_pct": 0.005,
    "take_profit_pct": 0.005,
    "max_hold_days": 1,
    "entry_score_threshold": 5,
    "min_prev_day_change": 0.005,
    "trailing_giveback_pct": 0.002,
    "trailing_activate_pct": 0.005,
    "partial_take_profit_pct": 0.005,
    "position_pct": 0.05,
    "max_gap_down_pct": 0.005,
}


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CycleAnalyzer:
    """Analyze completed trade cycles and recommend parameter changes."""

    def __init__(self, db: PortfolioDB | None = None):
        self._db = db or PortfolioDB()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_cycles(self) -> list[TradeCycle]:
        """Load all cycles (open + closed) from the database."""
        rows = []
        rows.extend(self._db.get_open_cycles())
        rows.extend(self._db.get_closed_cycles(limit=500))
        seen = set()
        result = []
        for r in rows:
            cid = r.get("cycle_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                result.append(TradeCycle.from_dict(r))
        return result

    def get_open_cycles(self) -> list[TradeCycle]:
        rows = self._db.get_open_cycles()
        return [TradeCycle.from_dict(r) for r in rows]

    def get_closed_cycles(self, limit: int = 100) -> list[TradeCycle]:
        rows = self._db.get_closed_cycles(limit=limit)
        return [TradeCycle.from_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_stats(
        self, cycles: list[TradeCycle] | None = None
    ) -> CycleStats:
        """Compute aggregate statistics from trade cycles."""
        if cycles is None:
            cycles = self.get_closed_cycles(limit=200)

        closed = [c for c in cycles if c.is_closed and c.pnl_pct is not None]
        open_ = [c for c in cycles if not c.is_closed]

        stats = CycleStats(
            total_cycles=len(cycles),
            closed_cycles=len(closed),
            open_cycles=len(open_),
        )

        if not closed:
            return stats

        wins = [c for c in closed if (c.pnl or 0) > 0]
        losses = [c for c in closed if (c.pnl or 0) <= 0]
        stats.win_count = len(wins)
        stats.loss_count = len(losses)
        stats.win_rate = len(wins) / len(closed) if closed else 0

        win_pcts = [c.pnl_pct for c in wins if c.pnl_pct is not None]
        loss_pcts = [c.pnl_pct for c in losses if c.pnl_pct is not None]
        stats.avg_win_pct = float(np.mean(win_pcts)) if win_pcts else 0.0
        stats.avg_loss_pct = float(np.mean(loss_pcts)) if loss_pcts else 0.0

        hold_days = [c.holding_days for c in closed if c.holding_days is not None]
        stats.avg_holding_days = float(np.mean(hold_days)) if hold_days else 0.0

        total_wins = sum(c.pnl for c in wins if c.pnl) if wins else 0
        total_losses = abs(sum(c.pnl for c in losses if c.pnl)) if losses else 0
        stats.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        stats.total_pnl = sum(c.pnl for c in closed if c.pnl) or 0
        pnl_pcts = [c.pnl_pct for c in closed if c.pnl_pct is not None]
        # Total return: cumulative product of (1 + pnl_pct)
        if pnl_pcts:
            cumulative = 1.0
            for p in pnl_pcts:
                cumulative *= (1.0 + p)
            stats.total_pnl_pct = cumulative - 1.0

        stats.expectancy = (
            stats.win_rate * stats.avg_win_pct
            + (1 - stats.win_rate) * stats.avg_loss_pct
        )

        # Best / worst
        sorted_by_pnl = sorted(
            [c for c in closed if c.pnl_pct is not None],
            key=lambda c: c.pnl_pct or 0,
        )
        if sorted_by_pnl:
            best = sorted_by_pnl[-1]
            worst = sorted_by_pnl[0]
            stats.best_cycle = {
                "code": best.code, "name": best.name,
                "pnl": best.pnl, "pnl_pct": best.pnl_pct,
                "holding_days": best.holding_days,
                "exit_reason": best.exit_reason,
            }
            stats.worst_cycle = {
                "code": worst.code, "name": worst.name,
                "pnl": worst.pnl, "pnl_pct": worst.pnl_pct,
                "holding_days": worst.holding_days,
                "exit_reason": worst.exit_reason,
            }

        # Exit reason distribution
        reasons: dict[str, int] = {}
        for c in closed:
            # Simplify reason to category
            reason = c.exit_reason or "未知"
            if "止损" in reason:
                cat = "硬止损"
            elif "时间" in reason or "持有" in reason:
                cat = "时间止盈"
            elif "目标止盈" in reason or "止盈" in reason:
                cat = "目标止盈"
            elif "阶梯" in reason:
                cat = "阶梯止盈"
            elif "移动" in reason:
                cat = "移动止盈"
            elif "RSI" in reason or "超买" in reason:
                cat = "RSI超买"
            elif "MA5" in reason or "跌破" in reason:
                cat = "跌破MA5"
            elif "跳空" in reason:
                cat = "跳空保护"
            else:
                cat = reason[:10] if reason else "未知"
            reasons[cat] = reasons.get(cat, 0) + 1
        stats.exit_reasons = dict(
            sorted(reasons.items(), key=lambda x: -x[1])
        )

        # Per-code stats
        code_groups: dict[str, list[TradeCycle]] = {}
        for c in closed:
            code_groups.setdefault(c.code, []).append(c)
        for code, group in code_groups.items():
            w = [c for c in group if (c.pnl or 0) > 0]
            stats.by_code[code] = {
                "name": group[0].name or code,
                "cycles": len(group),
                "wins": len(w),
                "losses": len(group) - len(w),
                "win_rate": len(w) / len(group) if group else 0,
                "total_pnl": sum(c.pnl for c in group if c.pnl) or 0,
                "avg_pnl_pct": float(np.mean([
                    c.pnl_pct for c in group if c.pnl_pct is not None
                ])) if group else 0,
            }

        # Per-holding-days stats
        day_groups: dict[int, list[TradeCycle]] = {}
        for c in closed:
            if c.holding_days is not None:
                day_groups.setdefault(c.holding_days, []).append(c)
        for days, group in day_groups.items():
            w = [c for c in group if (c.pnl or 0) > 0]
            stats.by_hold_days[days] = {
                "cycles": len(group),
                "wins": len(w),
                "win_rate": len(w) / len(group) if group else 0,
                "avg_pnl_pct": float(np.mean([
                    c.pnl_pct for c in group if c.pnl_pct is not None
                ])) if group else 0,
            }

        return stats

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _clamp(self, param: str, value: float) -> float:
        """Clamp value to parameter bounds and round to step size."""
        lo, hi = PARAM_BOUNDS.get(param, (0, float("inf")))
        step = PARAM_STEPS.get(param, 0.01)
        clamped = max(lo, min(hi, value))
        return round(clamped / step) * step

    def recommend_changes(
        self,
        cycles: list[TradeCycle] | None = None,
        current_params: dict | None = None,
    ) -> list[ParameterChange]:
        """Generate parameter change recommendations.

        Uses the most recent closed cycles to analyze strategy performance
        and suggest parameter tweaks.

        Args:
            cycles: Closed cycles to analyze. If None, loads from DB.
            current_params: Current strategy params. If None, uses defaults.

        Returns:
            List of ParameterChange recommendations, most important first.
        """
        if cycles is None:
            cycles = self.get_closed_cycles(limit=200)

        if current_params is None:
            from src.strategy.short_term_momentum import ShortTermMomentumStrategy
            current_params = ShortTermMomentumStrategy().get_default_params()

        closed = [c for c in cycles if c.is_closed and c.pnl_pct is not None]
        changes: list[ParameterChange] = []

        if len(closed) < 5:
            return changes  # Not enough data

        stats = self.compute_stats(closed)

        # Take most recent cycles for recency-weighted decisions
        recent = closed[: min(20, len(closed))]
        recent_wins = [c for c in recent if (c.pnl or 0) > 0]
        recent_wr = len(recent_wins) / len(recent) if recent else 0

        # ── Rule 1: Low recent win rate → defensive tightening ──
        if len(recent) >= 10 and recent_wr < 0.40:
            old_sl = float(current_params.get("stop_loss_pct", 0.02))
            new_sl = self._clamp("stop_loss_pct", old_sl - 0.005)
            if new_sl != old_sl:
                changes.append(ParameterChange(
                    parameter="stop_loss_pct",
                    current_value=old_sl,
                    proposed_value=new_sl,
                    reason=f"近{len(recent)}笔胜率仅{recent_wr:.0%}，收紧止损减少单笔亏损",
                    confidence="high",
                    supporting_data={"recent_win_rate": recent_wr, "sample_size": len(recent)},
                ))

            old_days = int(current_params.get("max_hold_days", 3))
            new_days = self._clamp("max_hold_days", old_days - 1)
            if new_days != old_days and new_days >= 2:
                changes.append(ParameterChange(
                    parameter="max_hold_days",
                    current_value=float(old_days),
                    proposed_value=float(new_days),
                    reason=f"胜率偏低，缩短持仓时间减少暴露",
                    confidence="medium",
                    supporting_data={"recent_win_rate": recent_wr},
                ))

        # ── Rule 2: Stop loss hit rate too high → stop may be too tight ──
        stop_hits = sum(
            1 for c in recent if "止损" in (c.exit_reason or "")
        )
        if len(recent) >= 10 and stop_hits / len(recent) > 0.50:
            old_sl = float(current_params.get("stop_loss_pct", 0.02))
            new_sl = self._clamp("stop_loss_pct", old_sl + 0.005)
            if new_sl != old_sl:
                changes.append(ParameterChange(
                    parameter="stop_loss_pct",
                    current_value=old_sl,
                    proposed_value=new_sl,
                    reason=f"近{len(recent)}笔中{stop_hits}笔触发止损({stop_hits/len(recent):.0%})，止损可能太紧",
                    confidence="medium",
                    supporting_data={"stop_hit_rate": stop_hits / len(recent)},
                ))

        # ── Rule 3: Take profit rarely hit → target too far ──
        tp_hits = sum(
            1 for c in closed if "目标止盈" in (c.exit_reason or "")
        )
        tp_rate = tp_hits / len(closed) if closed else 0
        if len(closed) >= 10 and tp_rate < 0.15:
            old_tp = float(current_params.get("take_profit_pct", 0.06))
            new_tp = self._clamp("take_profit_pct", old_tp - 0.005)
            if new_tp != old_tp:
                changes.append(ParameterChange(
                    parameter="take_profit_pct",
                    current_value=old_tp,
                    proposed_value=new_tp,
                    reason=f"止盈触发率仅{tp_rate:.0%}（{tp_hits}/{len(closed)}），止盈线可能太高",
                    confidence="medium",
                    supporting_data={"tp_hit_rate": tp_rate},
                ))

        # ── Rule 4: Strategy working well → consider loosening ──
        if (
            len(closed) >= 15
            and stats.win_rate > 0.55
            and stats.profit_factor > 2.0
            and stats.expectancy > 0.01
        ):
            old_score = float(current_params.get("entry_score_threshold", 35))
            new_score = self._clamp("entry_score_threshold", old_score - 5)
            if new_score != old_score:
                changes.append(ParameterChange(
                    parameter="entry_score_threshold",
                    current_value=old_score,
                    proposed_value=new_score,
                    reason=f"策略表现优秀(PF={stats.profit_factor:.1f},WR={stats.win_rate:.0%})，放宽入场门槛捕捉更多机会",
                    confidence="low",
                    supporting_data={
                        "profit_factor": stats.profit_factor,
                        "win_rate": stats.win_rate,
                    },
                ))

        # ── Rule 5: Time exits dominate → may need more time ──
        time_exits = sum(
            1 for c in closed if "时间" in (c.exit_reason or "")
        )
        if len(closed) >= 10 and time_exits / len(closed) > 0.40:
            old_days = int(current_params.get("max_hold_days", 3))
            new_days = self._clamp("max_hold_days", old_days + 1)
            if new_days != old_days:
                changes.append(ParameterChange(
                    parameter="max_hold_days",
                    current_value=float(old_days),
                    proposed_value=float(new_days),
                    reason=f"时间止盈占{time_exits/len(closed):.0%}，可能太早退出",
                    confidence="low",
                    supporting_data={"time_exit_rate": time_exits / len(closed)},
                ))

        return changes

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply_changes(
        self,
        changes: list[ParameterChange],
        current_params: dict,
    ) -> dict:
        """Apply validated parameter changes, returning updated params dict."""
        params = dict(current_params)
        for change in changes:
            old = params.get(change.parameter)
            if old is not None:
                params[change.parameter] = change.proposed_value
        return params
