"""Autonomous paper-trading bot for ETF momentum strategy.

Runs a continuous polling loop during A-share market hours (9:30-15:00,
Mon-Fri), scanning 34 ETF candidates for momentum entry/exit signals
and executing trades automatically via the PortfolioManager.

Usage::

    from src.engine.auto_trader import AutoTrader

    trader = AutoTrader(initial_capital=4000)
    trader.run_once()                    # one scan cycle
    trader.run_loop(interval_minutes=5)  # continuous loop
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import date, datetime, time as dtime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_etf_hist, fetch_multi_etf_info
from src.data.portfolio_db import PortfolioDB
from src.engine.portfolio import PortfolioManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_PARAMS_FILE = Path(__file__).resolve().parent / "auto_trader_params.json"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger("auto_trader")
_logger.setLevel(logging.INFO)
_logger.propagate = False

if not _logger.handlers:
    _LOG_DIR.mkdir(exist_ok=True)
    fh = RotatingFileHandler(
        _LOG_DIR / "auto_trader.log",
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(sh)


# ---------------------------------------------------------------------------
# AutoTrader
# ---------------------------------------------------------------------------


class AutoTrader:
    """Autonomous momentum-strategy paper-trading bot.

    Polls real-time prices, checks entry/exit signals via
    ShortTermMomentumStrategy, and executes trades on a PortfolioManager
    backed by SQLite persistence.
    """

    def __init__(
        self,
        initial_capital: float = 4000.0,
        auto_refine: bool = False,
        refine_interval: int = 10,
    ):
        self._auto_refine = auto_refine
        self._refine_interval = refine_interval
        self._log = _logger
        self._hist_cache: dict[str, pd.DataFrame | None] = {}
        self._hist_cache_date: str = ""
        self._new_today: set[str] = set()  # codes entered today
        self._sold_today: set[str] = set()  # codes exited today
        self._today: str = date.today().isoformat()
        self._last_refine_count = 0

        # ── Two-tier scanning ──
        self._watchlist: list[str] = []        # Top 5 candidate codes from full scan
        self._full_scan_am_done: bool = False
        self._full_scan_midday_done: bool = False

        # ── Dashboard signal cache (reads by Streamlit UI) ──
        self._position_signals: dict = {}   # {code: {action, reason, price, ...}}
        self._watchlist_signals: dict = {}  # {code: {action, reason, price, score, ...}}

        # ── Load or create portfolio ──
        self._db = PortfolioDB()
        pm = self._db.load()
        if pm is None or self._db.get_trade_count() == 0:
            pm = PortfolioManager(initial_capital=initial_capital)
            self._log.info(
                f"创建新模拟账户，初始资金 ¥{initial_capital:,.0f}"
            )
            self._db.save(pm)
        self._pm = pm

        # ── Strategy & Analyzer (lazy; must exist before _load_params) ──
        self._strategy = None
        self._analyzer = None

        # ── Load or create params ──
        self._params = self._load_params()

        self._log.info(
            f"AutoTrader 就绪 | 资金 ¥{pm.cash:,.0f} | "
            f"持仓 {len(pm.holdings)} | 交易历史 {self._db.get_trade_count()} 笔"
        )

    # ------------------------------------------------------------------
    # Strategy (lazy)
    # ------------------------------------------------------------------

    @property
    def strategy(self):
        if self._strategy is None:
            from src.strategy.short_term_momentum import ShortTermMomentumStrategy
            self._strategy = ShortTermMomentumStrategy()
        return self._strategy

    @property
    def analyzer(self):
        if self._analyzer is None:
            from src.engine.strategy_refiner import CycleAnalyzer
            self._analyzer = CycleAnalyzer(self._db)
        return self._analyzer

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    @staticmethod
    def is_market_open() -> bool:
        """Check if China A-share market is currently open.

        Hours: 9:30-11:30, 13:00-15:00, Monday-Friday.
        Does NOT check holidays — handled by fetch failure.
        """
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return (dtime(9, 30) <= t <= dtime(11, 30)) or (
            dtime(13, 0) <= t <= dtime(15, 0)
        )

    @staticmethod
    def seconds_until_next_open() -> float:
        """Seconds until the next market open window."""
        now = datetime.now()
        if now.weekday() >= 5:
            # Weekend: next Monday 9:30
            days_to_mon = 7 - now.weekday()
            next_open = now.replace(
                hour=9, minute=30, second=0, microsecond=0
            ) + pd.Timedelta(days=days_to_mon)
        else:
            t = now.time()
            if t < dtime(9, 30):
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            elif dtime(11, 30) < t < dtime(13, 0):
                next_open = now.replace(hour=13, minute=0, second=0, microsecond=0)
            elif t > dtime(15, 0):
                # Next weekday 9:30
                if now.weekday() == 4:  # Friday
                    next_open = now.replace(
                        hour=9, minute=30, second=0, microsecond=0
                    ) + pd.Timedelta(days=3)
                else:
                    next_open = now.replace(
                        hour=9, minute=30, second=0, microsecond=0
                    ) + pd.Timedelta(days=1)
            else:
                return 0  # Already open
        return (next_open - now).total_seconds()

    # ------------------------------------------------------------------
    # Params persistence
    # ------------------------------------------------------------------

    def _load_params(self) -> dict:
        """Load strategy params from JSON, or copy from strategy defaults."""
        if _PARAMS_FILE.exists():
            try:
                with open(_PARAMS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._log.info("已加载策略参数文件")
                return data.get("strategy_params", {})
            except Exception:
                self._log.warning("参数文件损坏，使用默认参数")
        # Create from strategy defaults
        params = self.strategy.get_default_params()
        # ¥4,000 适配
        params["position_pct"] = 0.25
        params["max_concurrent"] = 2
        self._save_params(params)
        self._log.info("已创建默认策略参数文件")
        return params

    def _save_params(self, params: dict | None = None):
        """Save current params to JSON file."""
        p = params or self._params
        data = {
            "strategy_params": dict(p),
            "refine_interval": self._refine_interval,
            "auto_refine": self._auto_refine,
            "last_refine_count": self._last_refine_count,
        }
        with open(_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_hist(self, code: str) -> pd.DataFrame | None:
        """Fetch historical K-line data with daily caching."""
        today = date.today().isoformat()
        if today != self._hist_cache_date:
            self._hist_cache.clear()
            self._hist_cache_date = today
        if code not in self._hist_cache:
            try:
                self._hist_cache[code] = fetch_etf_hist(code)
            except Exception:
                self._hist_cache[code] = None
        return self._hist_cache.get(code)

    def _get_prices_and_info(
        self, codes: list[str]
    ) -> dict[str, dict]:
        """Batch-fetch real-time quotes for multiple codes."""
        try:
            return fetch_multi_etf_info(codes)
        except Exception as e:
            self._log.warning(f"获取实时行情失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self):
        """Reset daily counters if it's a new day."""
        today = date.today().isoformat()
        if today != self._today:
            self._today = today
            self._new_today.clear()
            self._sold_today.clear()
            self._full_scan_am_done = False
            self._full_scan_midday_done = False
            self._watchlist.clear()

    # ------------------------------------------------------------------
    # Scan scheduling
    # ------------------------------------------------------------------

    def _needs_full_scan(self) -> bool:
        """Decide whether a full candidate-pool scan is due.

        Full scan triggers:
          - First scan of the morning session (>=9:30)
          - First scan during the midday break (11:30–13:00)
          - Watchlist is empty (first run ever, or cleared on new day)
        """
        if not self._watchlist:
            return True
        now = datetime.now().time()
        if not self._full_scan_am_done and now >= dtime(9, 30):
            return True
        if not self._full_scan_midday_done and dtime(11, 30) <= now <= dtime(13, 0):
            return True
        return False

    def _mark_full_scan_done(self):
        """Record that a full scan completed in the current time window."""
        now = datetime.now().time()
        if now < dtime(11, 30):
            self._full_scan_am_done = True
        else:
            self._full_scan_midday_done = True

    # ------------------------------------------------------------------
    # Core: one scan cycle
    # ------------------------------------------------------------------

    def run_once(self, dry_run: bool = False) -> dict:
        """Execute one scan-and-trade cycle (dispatches to full or quick scan).

        Args:
            dry_run: If True, only scan signals, do NOT execute trades.

        Returns:
            Summary dict with scanned, signals_found, exits_triggered,
            entries_executed, equity, cash, positions.
        """
        self._maybe_reset_daily()

        if self._needs_full_scan():
            self._log.info(
                f"{'上午' if not self._full_scan_am_done else '午休'}"
                f"完整扫描 — 全池34只ETF评估"
            )
            result = self._do_full_scan(dry_run)
            self._mark_full_scan_done()
        else:
            result = self._do_quick_scan(dry_run)

        return result

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------

    def _do_full_scan(self, dry_run: bool = False) -> dict:
        """Full scan: all 34 ETFs, K-line + scoring, rebuild watchlist."""
        # ── Get candidate pool ──
        pool = self.strategy.get_candidate_pool("etf")
        all_codes = [e["code"] for e in pool]
        # Also include currently held codes
        for code in self._pm.holdings:
            if code not in all_codes:
                all_codes.append(code)

        # ── Batch fetch real-time prices ──
        info_map = self._get_prices_and_info(all_codes)

        # Update portfolio mark-to-market
        price_map = {
            code: info.get("current_price", 0)
            for code, info in info_map.items()
            if info.get("current_price")
        }
        self._pm.update_prices(price_map)

        result = {
            "scanned": len(info_map),
            "signals_found": 0,
            "exits_triggered": 0,
            "entries_executed": 0,
            "equity": self._pm.summary().get("total_equity", 0),
            "cash": self._pm.cash,
            "positions": len(self._pm.holdings),
            "timestamp": datetime.now().isoformat(),
        }

        # ═════════════════════════════════════════════════════════════
        # Phase 1: Check exits for current positions
        # ═════════════════════════════════════════════════════════════
        for code, holding in list(self._pm.holdings.items()):
            if holding.shares <= 0:
                continue

            info = info_map.get(code)
            if not info or not info.get("current_price"):
                continue

            df = self._get_hist(code)
            if df is None or df.empty:
                continue

            current_price = info["current_price"]

            # Build portfolio context
            pf_ctx = {
                "has_position": True,
                "holding_avg_cost": holding.avg_cost,
                "holding_shares": holding.shares,
                "available_cash": self._pm.cash,
                "code": code,
                "last_buy_date": self._find_last_buy_date(code),
            }

            try:
                signal = self.strategy.get_live_signal(
                    df, info, portfolio_context=pf_ctx, **self._params,
                )
            except Exception as e:
                self._log.error(f"get_live_signal({code}) 异常: {e}")
                continue

            # ── Cache signal for dashboard ──
            self._position_signals[code] = {
                "action": signal.action,
                "reason": signal.trigger_description or signal.reason,
                "current_price": current_price,
                "suggested_price_low": getattr(signal, "suggested_price_low", None),
                "suggested_price_high": getattr(signal, "suggested_price_high", None),
            }

            if signal.action == "sell":
                result["signals_found"] += 1
                if dry_run:
                    self._log.info(
                        f"[DRY RUN] {code} 卖出信号: {signal.trigger_description}"
                    )
                else:
                    trade = self._pm.sell(
                        code=code,
                        price=current_price,
                        shares=min(100, holding.shares),
                        name=info.get("name", code),
                        reason=signal.reason,
                    )
                    if trade:
                        result["exits_triggered"] += 1
                        self._sold_today.add(code)
                        self._db.save(self._pm)
                        # Close the corresponding trade cycle
                        self._close_cycle_for_code(code, trade)
                        self._log.info(
                            f"🔴 卖出 {code} {trade.shares}股 @ ¥{current_price:.3f} "
                            f"盈亏 {trade.pnl:+.2f} ({trade.pnl_pct:+.2%}) | "
                            f"{signal.trigger_description}"
                        )
                    else:
                        self._log.warning(f"卖出 {code} 失败")

        # ═════════════════════════════════════════════════════════════
        # Phase 2: Scan for new entries
        # ═════════════════════════════════════════════════════════════
        max_concurrent = int(self._params.get("max_concurrent", 2))
        max_new_per_day = int(self._params.get("max_new_per_day", 1))
        current_positions = len(self._pm.holdings)

        if current_positions < max_concurrent and len(self._new_today) < max_new_per_day:
            # Use scan_candidates for efficient multi-ETF screening
            try:
                candidates = self.strategy.scan_candidates(
                    candidates=pool,
                    asset_mode="etf",
                    top_n=5,
                    **self._params,
                )
            except Exception as e:
                self._log.error(f"scan_candidates 异常: {e}")
                candidates = []

            # ── Build watchlist from top candidates (for quick scans) ──
            watchlist_size = 5
            self._watchlist = [
                c["code"] for c in candidates
                if c.get("passed") and c["code"] not in self._pm.holdings
            ][:watchlist_size]

            # ── Cache candidate signals for dashboard ──
            for c in candidates:
                code = c["code"]
                if code in self._watchlist or code in self._pm.holdings:
                    self._watchlist_signals[code] = {
                        "action": "buy" if c.get("passed") else "wait",
                        "reason": c.get("action_detail", "") if c.get("passed") else "未通过筛选",
                        "current_price": c.get("current_price"),
                        "suggested_price_low": c.get("current_price", 0) * 0.99 if c.get("current_price") else None,
                        "suggested_price_high": c.get("current_price", 0) * 1.01 if c.get("current_price") else None,
                        "score": c.get("score"),
                    }

            if self._watchlist:
                self._log.info(
                    f"候选池: {', '.join(self._watchlist)}"
                )

            for cand in candidates:
                if len(self._new_today) >= max_new_per_day:
                    break
                if current_positions >= max_concurrent:
                    break

                code = cand["code"]
                if code in self._pm.holdings:
                    continue
                if code in self._sold_today:
                    continue  # Don't re-buy same day
                if not cand.get("passed"):
                    continue

                cp = cand.get("current_price", 0)
                if cp <= 0:
                    continue

                # Position sizing — fixed 100 shares
                shares = 100

                result["signals_found"] += 1

                if dry_run:
                    self._log.info(
                        f"[DRY RUN] {code} {cand['name']} 买入信号: "
                        f"评分{cand['score']:.0f} @ ¥{cp:.3f} × {shares}股"
                    )
                else:
                    name = cand.get("name_from_api") or cand.get("name", code)
                    reason = (
                        f"🚀 自动买入 | 评分{cand['score']:.0f} | "
                        f"前日涨{cand.get('daily_return', 0):+.1%}"
                    )
                    trade = self._pm.buy(
                        code=code,
                        price=cp,
                        shares=shares,
                        name=name,
                        reason=reason,
                    )
                    if trade:
                        result["entries_executed"] += 1
                        self._new_today.add(code)
                        current_positions += 1
                        self._db.save(self._pm)
                        # Create trade cycle
                        self._db.create_cycle(
                            cycle_id=str(uuid.uuid4())[:12],
                            code=code,
                            name=name,
                            entry_date=trade.date,
                            entry_price=cp,
                            shares=trade.shares,
                            entry_amount=trade.amount,
                            entry_reason=reason,
                        )
                        self._log.info(
                            f"🟢 买入 {code} {name} {trade.shares}股 @ ¥{cp:.3f} "
                            f"≈ ¥{trade.amount:,.0f} | 评分{cand['score']:.0f}"
                        )
                    else:
                        self._log.warning(f"买入 {code} 失败（资金不足或其他原因）")

        # Final save
        if not dry_run:
            self._db.save(self._pm)
            result["equity"] = self._pm.summary().get("total_equity", 0)
            result["cash"] = self._pm.cash
            result["positions"] = len(self._pm.holdings)

        return result

    # ------------------------------------------------------------------
    # Quick scan (held + watchlist only)
    # ------------------------------------------------------------------

    def _do_quick_scan(self, dry_run: bool = False) -> dict:
        """Lightweight scan: only held positions + watchlist candidates.

        Still batch-fetches all 34 prices (one HTTP call regardless of count),
        but only pulls K-line history for the monitored subset.
        """
        # ── Monitor set = held codes ∪ watchlist codes ──
        held_codes = list(self._pm.holdings.keys())
        monitor_codes = list(set(held_codes) | set(self._watchlist))

        # If nothing to monitor, short-circuit
        if not monitor_codes:
            return {
                "scanned": 0, "signals_found": 0,
                "exits_triggered": 0, "entries_executed": 0,
                "equity": self._pm.summary().get("total_equity", 0),
                "cash": self._pm.cash, "positions": 0,
                "timestamp": datetime.now().isoformat(),
            }

        # ── Batch fetch all 34 real-time prices ──
        pool = self.strategy.get_candidate_pool("etf")
        all_codes = [e["code"] for e in pool]
        info_map = self._get_prices_and_info(all_codes)

        # Update portfolio mark-to-market
        price_map = {
            code: info.get("current_price", 0)
            for code, info in info_map.items()
            if info.get("current_price")
        }
        self._pm.update_prices(price_map)

        # ── Pre-fetch K-line for monitor codes only ──
        for code in monitor_codes:
            self._get_hist(code)  # daily cache — cold-fetches, rest are no-ops

        result = {
            "scanned": len(info_map),  # prices fetched for all 34
            "signals_found": 0,
            "exits_triggered": 0,
            "entries_executed": 0,
            "equity": self._pm.summary().get("total_equity", 0),
            "cash": self._pm.cash,
            "positions": len(self._pm.holdings),
            "timestamp": datetime.now().isoformat(),
        }

        # ═════════════════════════════════════════════════════════════
        # Phase 1: Check exits for held positions (same as full scan)
        # ═════════════════════════════════════════════════════════════
        for code, holding in list(self._pm.holdings.items()):
            if holding.shares <= 0:
                continue

            info = info_map.get(code)
            if not info or not info.get("current_price"):
                continue

            df = self._get_hist(code)
            if df is None or df.empty:
                continue

            current_price = info["current_price"]

            pf_ctx = {
                "has_position": True,
                "holding_avg_cost": holding.avg_cost,
                "holding_shares": holding.shares,
                "available_cash": self._pm.cash,
                "code": code,
                "last_buy_date": self._find_last_buy_date(code),
            }

            try:
                signal = self.strategy.get_live_signal(
                    df, info, portfolio_context=pf_ctx, **self._params,
                )
            except Exception as e:
                self._log.error(f"get_live_signal({code}) 异常: {e}")
                continue

            # ── Cache signal for dashboard ──
            self._position_signals[code] = {
                "action": signal.action,
                "reason": signal.trigger_description or signal.reason,
                "current_price": current_price,
                "suggested_price_low": getattr(signal, "suggested_price_low", None),
                "suggested_price_high": getattr(signal, "suggested_price_high", None),
            }

            if signal.action == "sell":
                result["signals_found"] += 1
                if dry_run:
                    self._log.info(
                        f"[DRY RUN] {code} 卖出信号: {signal.trigger_description}"
                    )
                else:
                    trade = self._pm.sell(
                        code=code,
                        price=current_price,
                        shares=min(100, holding.shares),
                        name=info.get("name", code),
                        reason=signal.reason,
                    )
                    if trade:
                        result["exits_triggered"] += 1
                        self._sold_today.add(code)
                        self._db.save(self._pm)
                        self._close_cycle_for_code(code, trade)
                        self._log.info(
                            f"🔴 卖出 {code} {trade.shares}股 @ ¥{current_price:.3f} "
                            f"盈亏 {trade.pnl:+.2f} ({trade.pnl_pct:+.2%}) | "
                            f"{signal.trigger_description}"
                        )
                    else:
                        self._log.warning(f"卖出 {code} 失败")

        # ═════════════════════════════════════════════════════════════
        # Phase 2: Check watchlist for entries (one by one via get_live_signal)
        # ═════════════════════════════════════════════════════════════
        max_concurrent = int(self._params.get("max_concurrent", 2))
        max_new_per_day = int(self._params.get("max_new_per_day", 1))
        current_positions = len(self._pm.holdings)

        if current_positions < max_concurrent and len(self._new_today) < max_new_per_day:
            for code in list(self._watchlist):  # iterate a copy — may mutate
                if len(self._new_today) >= max_new_per_day:
                    break
                if current_positions >= max_concurrent:
                    break
                if code in self._pm.holdings:
                    continue
                if code in self._sold_today:
                    continue

                info = info_map.get(code)
                if not info or not info.get("current_price"):
                    continue

                df = self._get_hist(code)
                if df is None or df.empty:
                    continue

                cp = info["current_price"]
                if cp <= 0:
                    continue

                # Check entry signal via get_live_signal (no position)
                pf_ctx = {
                    "has_position": False,
                    "available_cash": self._pm.cash,
                    "code": code,
                }

                try:
                    signal = self.strategy.get_live_signal(
                        df, info, portfolio_context=pf_ctx, **self._params,
                    )
                except Exception as e:
                    self._log.error(f"get_live_signal({code}) 快速扫描异常: {e}")
                    continue

                # ── Cache signal for dashboard ──
                self._watchlist_signals[code] = {
                    "action": signal.action,
                    "reason": signal.trigger_description or signal.reason,
                    "current_price": cp,
                    "suggested_price_low": getattr(signal, "suggested_price_low", None),
                    "suggested_price_high": getattr(signal, "suggested_price_high", None),
                    "score": None,
                }

                if signal.action != "buy":
                    # Not ready yet — stays on watchlist
                    continue

                # ── Position sizing — fixed 100 shares ──
                shares = 100

                result["signals_found"] += 1

                if dry_run:
                    self._log.info(
                        f"[DRY RUN] {code} {info.get('name', code)} 买入信号: "
                        f"¥{cp:.3f} × {shares}股 — {signal.trigger_description}"
                    )
                else:
                    name = info.get("name", code)
                    reason = (
                        f"🚀 自动买入(快速) | {signal.trigger_description or '动量信号触发'}"
                    )
                    trade = self._pm.buy(
                        code=code, price=cp, shares=shares,
                        name=name, reason=reason,
                    )
                    if trade:
                        result["entries_executed"] += 1
                        self._new_today.add(code)
                        current_positions += 1
                        self._watchlist.remove(code)
                        self._db.save(self._pm)
                        self._db.create_cycle(
                            cycle_id=str(uuid.uuid4())[:12],
                            code=code, name=name,
                            entry_date=trade.date, entry_price=cp,
                            shares=trade.shares, entry_amount=trade.amount,
                            entry_reason=reason,
                        )
                        self._log.info(
                            f"🟢 买入 {code} {name} {trade.shares}股 @ ¥{cp:.3f} "
                            f"≈ ¥{trade.amount:,.0f} | {signal.trigger_description}"
                        )
                    else:
                        self._log.warning(f"买入 {code} 失败（资金不足或其他原因）")

        # Final save
        if not dry_run:
            self._db.save(self._pm)
            result["equity"] = self._pm.summary().get("total_equity", 0)
            result["cash"] = self._pm.cash
            result["positions"] = len(self._pm.holdings)

        return result

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------

    def get_dashboard_signals(self, info_map: dict | None = None) -> dict:
        """Return cached signal data for positions + watchlist.

        Called by the Streamlit UI between scan cycles. Returns a dict with:
          - positions: {code: {action, reason, price, pnl_pct, ...}}
          - watchlist: {code: {action, reason, price, ...}}
        """
        result: dict = {"positions": {}, "watchlist": {}}

        # ── Position signals ──
        for code, holding in self._pm.holdings.items():
            if holding.shares <= 0:
                continue

            # Compute unrealized P&L
            last_price = None
            for t in reversed(self._pm.trades):
                if t.code == code:
                    last_price = t.price
                    break

            pnl_pct = None
            if last_price and holding.avg_cost:
                pnl_pct = (last_price / holding.avg_cost) - 1

            # Try cached signal, fallback to basic hold status
            cached = self._position_signals.get(code, {})
            result["positions"][code] = {
                "name": holding.name or code,
                "shares": holding.shares,
                "avg_cost": holding.avg_cost,
                "current_price": last_price,
                "pnl_pct": pnl_pct,
                "action": cached.get("action", "hold"),
                "reason": cached.get("reason", "继续持有"),
                "suggested_price_low": cached.get("suggested_price_low"),
                "suggested_price_high": cached.get("suggested_price_high"),
            }

        # ── Watchlist signals ──
        pool = self.strategy.get_candidate_pool("etf")
        name_map = {e["code"]: e.get("name", "") for e in pool}

        for code in self._watchlist:
            if code in self._pm.holdings:
                continue
            cached = self._watchlist_signals.get(code, {})
            result["watchlist"][code] = {
                "name": name_map.get(code, code),
                "current_price": cached.get("current_price"),
                "action": cached.get("action", "wait"),
                "reason": cached.get("reason", "等待信号..."),
                "score": cached.get("score"),
                "suggested_price_low": cached.get("suggested_price_low"),
                "suggested_price_high": cached.get("suggested_price_high"),
            }

        return result

    # ------------------------------------------------------------------
    # Trade cycle tracking
    # ------------------------------------------------------------------

    def _find_last_buy_date(self, code: str) -> str | None:
        """Find the last buy date for a code from trade history."""
        for t in reversed(self._pm.trades):
            if t.code == code and t.action == "buy":
                return t.date
        return None

    def _close_cycle_for_code(self, code: str, sell_trade) -> None:
        """Close the oldest open cycle for this code (FIFO)."""
        open_cycles = self._db.get_open_cycles_for_code(code)
        if not open_cycles:
            return
        oldest = open_cycles[0]
        cycle_id = oldest["cycle_id"]
        pnl = sell_trade.pnl or 0
        pnl_pct = sell_trade.pnl_pct or 0
        entry_date = oldest["entry_date"]
        try:
            ed = datetime.strptime(entry_date, "%Y-%m-%d")
            sd = datetime.strptime(sell_trade.date, "%Y-%m-%d")
            holding_days = (sd - ed).days
        except Exception:
            holding_days = 0

        self._db.close_cycle(
            cycle_id=cycle_id,
            exit_date=sell_trade.date,
            exit_price=sell_trade.price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason=sell_trade.reason or "",
        )

    # ------------------------------------------------------------------
    # Strategy refinement
    # ------------------------------------------------------------------

    def _maybe_refine(self):
        """Check if refinement is due and apply if configured."""
        closed_count = self._db.get_closed_cycle_count()
        cycles_since = closed_count - self._last_refine_count
        if cycles_since < self._refine_interval:
            return

        closed_cycles = self.analyzer.get_closed_cycles(limit=200)
        recommendations = self.analyzer.recommend_changes(
            cycles=closed_cycles,
            current_params=self._params,
        )

        if recommendations:
            self._log.info(f"=== 策略优化建议 ({len(recommendations)} 条) ===")
            for rec in recommendations:
                self._log.info(
                    f"  [{rec.confidence}] {rec.parameter}: "
                    f"{rec.current_value} → {rec.proposed_value} | {rec.reason}"
                )

            if self._auto_refine:
                self._params = self.analyzer.apply_changes(
                    recommendations, self._params
                )
                self._save_params()
                self._log.info("✓ 参数已自动更新")

        self._last_refine_count = closed_count
        self._save_params()

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    def run_loop(self, interval_minutes: int = 5):
        """Run continuous polling loop during market hours.

        Args:
            interval_minutes: Minutes between scans (default 5).
        """
        self._log.info(
            f"自动交易循环启动 | 扫描间隔 {interval_minutes} 分钟 | "
            f"自动优化={'开' if self._auto_refine else '关'}"
        )

        while True:
            try:
                if not self.is_market_open():
                    wait = self.seconds_until_next_open()
                    if wait > 0:
                        wait_min = wait / 60
                        if wait_min > 60:
                            self._log.info(
                                f"盘后/周末，休眠 {wait_min/60:.1f} 小时..."
                            )
                        time.sleep(min(wait, 3600))
                    continue

                result = self.run_once()
                trades = (
                    f"卖出{result['exits_triggered']} | 买入{result['entries_executed']}"
                )
                self._log.info(
                    f"扫描 {result['scanned']}个 | {trades} | "
                    f"权益 ¥{result['equity']:,.2f} | "
                    f"现金 ¥{result['cash']:,.2f} | "
                    f"持仓 {result['positions']}个"
                )

                self._maybe_refine()
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                self._log.info("用户中断，自动交易停止")
                break
            except Exception as e:
                self._log.error(f"循环异常: {e}", exc_info=True)
                time.sleep(60)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, detailed: bool = False):
        """Print a formatted trade cycle report."""
        from src.engine.strategy_refiner import CycleAnalyzer

        analyzer = CycleAnalyzer(self._db)
        closed = analyzer.get_closed_cycles(limit=500)
        stats = analyzer.compute_stats(closed)

        print()
        print("=" * 60)
        print(f"  交易周期报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)
        print(f"  总周期: {stats.total_cycles} | 已关闭: {stats.closed_cycles} | 进行中: {stats.open_cycles}")
        print(f"  胜率: {stats.win_rate:.0%} ({stats.win_count}W/{stats.loss_count}L)")
        print(f"  均盈: {stats.avg_win_pct:+.2%} | 均亏: {stats.avg_loss_pct:+.2%}")
        print(f"  平均持有: {stats.avg_holding_days:.1f} 天")
        print(f"  盈亏比: {stats.profit_factor:.2f} | 期望值: {stats.expectancy:+.2%}")
        if stats.total_pnl_pct:
            print(f"  累计收益: {stats.total_pnl_pct:+.2%}")
        print(f"  总盈亏: ¥{stats.total_pnl:+,.2f}")

        if stats.best_cycle:
            b = stats.best_cycle
            print(f"\n  🏆 最佳: {b.get('name','')}({b['code']}) {b.get('pnl_pct',0):+.2%} "
                  f"持有{b['holding_days']}天 {b.get('exit_reason','')}")
        if stats.worst_cycle:
            w = stats.worst_cycle
            print(f"  💀 最差: {w.get('name','')}({w['code']}) {w.get('pnl_pct',0):+.2%} "
                  f"持有{w['holding_days']}天 {w.get('exit_reason','')}")

        print(f"\n  退出原因分布:")
        for reason, count in stats.exit_reasons.items():
            bar = "█" * min(count, 30)
            print(f"    {reason}: {count} {bar}")

        if stats.by_code:
            print(f"\n  按ETF统计:")
            for code, info in sorted(
                stats.by_code.items(),
                key=lambda x: x[1].get("total_pnl", 0) or 0,
                reverse=True,
            ):
                print(
                    f"    {code} {info['name']}: "
                    f"{info['cycles']}笔 WR={info['win_rate']:.0%} "
                    f"总盈亏 ¥{(info['total_pnl'] or 0):+,.2f}"
                )

        if detailed and closed:
            print(f"\n  详细记录（最近20笔）:")
            for c in closed[:20]:
                print(
                    f"    {c.code} {c.name} | "
                    f"{c.entry_date} → {c.exit_date or '持仓中'} | "
                    f"¥{c.entry_price:.3f} → ¥{c.exit_price or 0:.3f} | "
                    f"{(c.pnl_pct or 0):+.2%} | {c.holding_days or '?'}天 | "
                    f"{c.exit_reason or '—'}"
                )

        # Current params
        print(f"\n  当前策略参数:")
        key_params = [
            "min_prev_day_change", "entry_score_threshold",
            "stop_loss_pct", "take_profit_pct", "max_hold_days",
            "position_pct", "max_concurrent",
        ]
        for k in key_params:
            v = self._params.get(k, "—")
            print(f"    {k}: {v}")

        # Recommendations
        recs = analyzer.recommend_changes(
            cycles=closed, current_params=self._params
        )
        if recs:
            print(f"\n  💡 优化建议:")
            for r in recs:
                print(
                    f"    [{r.confidence}] {r.parameter}: "
                    f"{r.current_value} → {r.proposed_value}"
                )
                print(f"      {r.reason}")
        else:
            print(f"\n  ✅ 参数无需调整（数据不足或表现良好）")
        print()
