"""SQLite persistence for the ETF-rotation paper account.

Stores portfolio state (cash, P&L), holdings, full trade history,
and trade cycle tracking so that positions survive Streamlit server restarts.

Schema (5 tables, auto-created):
  - portfolio_state: singleton row with cash, P&L, config
  - holdings: one row per ETF position
  - trades: full trade log, append-only
  - trade_cycles: buy→sell round-trip tracking for strategy refinement
  - equity_snapshots: daily account-equity history

Usage::

    from src.data.portfolio_db import PortfolioDB

    db = PortfolioDB()
    db.save(pm)                           # persist after trade
    pm = db.load()                        # restore on startup, or None
    db.reset()                            # wipe everything
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.engine.portfolio import LOT_SIZE, ExecutedTrade, Holding, PortfolioManager

# ---------------------------------------------------------------------------
# DB location
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).resolve().parent / "portfolio_db"
_DB_PATH = _DB_DIR / "portfolio.sqlite3"


def _get_db_path() -> Path:
    """Ensure the DB directory exists and return the path."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    initial_capital REAL NOT NULL DEFAULT 100000,
    cash REAL NOT NULL DEFAULT 100000,
    realized_pnl REAL NOT NULL DEFAULT 0,
    commission_rate REAL DEFAULT 0.0003,
    min_commission REAL DEFAULT 5.0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
    code TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    shares INTEGER NOT NULL DEFAULT 0,
    avg_cost REAL NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0,
    current_price REAL NOT NULL DEFAULT 0,
    entry_date TEXT DEFAULT '',
    highest_price REAL NOT NULL DEFAULT 0,
    last_buy_date TEXT DEFAULT '',
    last_buy_shares INTEGER NOT NULL DEFAULT 0,
    rank_weak_days INTEGER NOT NULL DEFAULT 0,
    trend_weak_days INTEGER NOT NULL DEFAULT 0,
    last_signal_date TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT DEFAULT '',
    action TEXT NOT NULL,
    price REAL NOT NULL,
    shares INTEGER NOT NULL,
    amount REAL NOT NULL,
    commission REAL DEFAULT 0,
    net_amount REAL NOT NULL,
    pnl REAL,
    pnl_pct REAL,
    reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS trade_cycles (
    cycle_id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    name TEXT DEFAULT '',
    entry_date TEXT NOT NULL,
    exit_date TEXT,
    entry_price REAL NOT NULL,
    exit_price REAL,
    shares INTEGER NOT NULL,
    entry_amount REAL NOT NULL,
    pnl REAL,
    pnl_pct REAL,
    holding_days INTEGER,
    entry_reason TEXT DEFAULT '',
    exit_reason TEXT DEFAULT '',
    is_closed INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    snapshot_date TEXT PRIMARY KEY,
    data_as_of TEXT,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    market_value REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    total_return REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

UPSERT_STATE = """
INSERT OR REPLACE INTO portfolio_state
    (id, initial_capital, cash, realized_pnl, commission_rate, min_commission, updated_at)
VALUES (1, ?, ?, ?, ?, ?, datetime('now', 'localtime'));
"""

INSERT_TRADE = """
INSERT OR IGNORE INTO trades
    (trade_id, date, code, name, action, price, shares, amount,
     commission, net_amount, pnl, pnl_pct, reason)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

UPSERT_HOLDING = """
INSERT OR REPLACE INTO holdings
    (code, name, shares, avg_cost, total_cost, current_price, entry_date,
     highest_price, last_buy_date, last_buy_shares, rank_weak_days,
     trend_weak_days, last_signal_date)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

CLEAR_HOLDINGS = "DELETE FROM holdings;"


# ── Trade cycle SQL ──

INSERT_CYCLE = """
INSERT OR IGNORE INTO trade_cycles
    (cycle_id, code, name, entry_date, entry_price, shares, entry_amount, entry_reason)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

CLOSE_CYCLE = """
UPDATE trade_cycles SET
    exit_date = ?, exit_price = ?, pnl = ?, pnl_pct = ?,
    holding_days = ?, exit_reason = ?, is_closed = 1
WHERE cycle_id = ?;
"""

SELECT_OPEN_CYCLES = """
SELECT * FROM trade_cycles WHERE is_closed = 0 ORDER BY entry_date ASC;
"""

SELECT_CLOSED_CYCLES = """
SELECT * FROM trade_cycles WHERE is_closed = 1 ORDER BY entry_date DESC LIMIT ?;
"""

SELECT_OPEN_CYCLES_FOR_CODE = """
SELECT * FROM trade_cycles WHERE code = ? AND is_closed = 0 ORDER BY entry_date ASC;
"""

COUNT_CYCLES = "SELECT COUNT(*) FROM trade_cycles;"
COUNT_CLOSED_CYCLES = "SELECT COUNT(*) FROM trade_cycles WHERE is_closed = 1;"


def _cycle_row_to_dict(row: tuple) -> dict:
    """Convert a trade_cycles row to a dict for external use."""
    return {
        "cycle_id": row[0], "code": row[1], "name": row[2],
        "entry_date": row[3], "exit_date": row[4],
        "entry_price": row[5], "exit_price": row[6],
        "shares": row[7], "entry_amount": row[8],
        "pnl": row[9], "pnl_pct": row[10],
        "holding_days": row[11],
        "entry_reason": row[12] or "", "exit_reason": row[13] or "",
        "is_closed": bool(row[14]),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PortfolioDB:
    """SQLite-backed persistence for a single paper-trading portfolio."""

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path) if db_path else _get_db_path()
        self._init_tables()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        with self._conn() as conn:
            conn.executescript(CREATE_TABLES)
            self._migrate_holdings(conn)
            conn.commit()

    @staticmethod
    def _migrate_holdings(conn: sqlite3.Connection) -> None:
        """Add paper-trading state columns to databases from older releases."""

        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings)").fetchall()
        }
        additions = {
            "current_price": "REAL NOT NULL DEFAULT 0",
            "entry_date": "TEXT DEFAULT ''",
            "highest_price": "REAL NOT NULL DEFAULT 0",
            "last_buy_date": "TEXT DEFAULT ''",
            "last_buy_shares": "INTEGER NOT NULL DEFAULT 0",
            "rank_weak_days": "INTEGER NOT NULL DEFAULT 0",
            "trend_weak_days": "INTEGER NOT NULL DEFAULT 0",
            "last_signal_date": "TEXT DEFAULT ''",
        }
        for column, definition in additions.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE holdings ADD COLUMN {column} {definition}")

    def _conn(self) -> sqlite3.Connection:
        """Return a new connection (each call). Caller must close."""
        conn = sqlite3.connect(str(self._path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, pm: PortfolioManager) -> bool:
        """Persist the full portfolio state (state + holdings + new trades).

        Returns True on success.
        """
        try:
            with self._conn() as conn:
                # -- State --
                conn.execute(UPSERT_STATE, (
                    pm.initial_capital, pm.cash, pm.realized_pnl,
                    pm.commission_rate, pm.min_commission,
                ))

                # -- Holdings (full replace) --
                conn.execute(CLEAR_HOLDINGS)
                for code, h in pm.holdings.items():
                    if h.shares > 0:
                        conn.execute(UPSERT_HOLDING, (
                            code, h.name, h.shares, h.avg_cost, h.total_cost,
                            h.current_price, h.entry_date, h.highest_price,
                            h.last_buy_date, h.last_buy_shares,
                            h.rank_weak_days, h.trend_weak_days,
                            h.last_signal_date,
                        ))

                # -- Trades (append only, skip duplicates by trade_id) --
                for t in pm.trades:
                    conn.execute(INSERT_TRADE, (
                        t.trade_id, t.date, t.code, t.name, t.action,
                        t.price, t.shares, t.amount, t.commission,
                        t.net_amount, t.pnl, t.pnl_pct, t.reason,
                    ))

                conn.commit()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> PortfolioManager | None:
        """Restore portfolio from DB. Returns None if no saved state exists."""
        try:
            with self._conn() as conn:
                # State
                row = conn.execute(
                    "SELECT initial_capital, cash, realized_pnl, commission_rate, "
                    "min_commission FROM portfolio_state WHERE id=1"
                ).fetchone()

                if row is None:
                    return None  # First run — no state yet

                pm = PortfolioManager(
                    initial_capital=row[0],
                    commission_rate=row[3],
                    min_commission=row[4],
                )
                pm.cash = row[1]
                pm._realized_pnl = row[2]

                # Holdings
                for hr in conn.execute(
                    "SELECT code, name, shares, avg_cost, total_cost, current_price, "
                    "entry_date, highest_price, last_buy_date, last_buy_shares, "
                    "rank_weak_days, trend_weak_days, last_signal_date FROM holdings"
                ):
                    if hr[2] > 0:
                        pm._holdings[hr[0]] = Holding(
                            code=hr[0], name=hr[1], shares=hr[2],
                            avg_cost=hr[3], total_cost=hr[4], current_price=hr[5],
                            entry_date=hr[6] or "", highest_price=hr[7],
                            last_buy_date=hr[8] or "", last_buy_shares=hr[9],
                            rank_weak_days=hr[10], trend_weak_days=hr[11],
                            last_signal_date=hr[12] or "",
                        )

                # Trades (replay in chronological order to rebuild state)
                for tr in conn.execute(
                    "SELECT trade_id, date, code, name, action, price, shares, "
                    "amount, commission, net_amount, pnl, pnl_pct, reason "
                    "FROM trades ORDER BY id ASC"
                ):
                    pnl_val = tr[10] if tr[10] is not None else None
                    pnl_pct_val = tr[11] if tr[11] is not None else None
                    pm._trades.append(ExecutedTrade(
                        trade_id=tr[0], date=tr[1], code=tr[2], name=tr[3],
                        action=tr[4], price=tr[5], shares=tr[6],
                        amount=tr[7], commission=tr[8], net_amount=tr[9],
                        pnl=pnl_val, pnl_pct=pnl_pct_val, reason=tr[12] or "",
                    ))

                # Recalculate realized P&L from sell trades
                pm._realized_pnl = sum(
                    t.pnl for t in pm._trades if t.action == "sell" and t.pnl
                )

                return pm
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_trade_count(self) -> int:
        """Total number of trades stored."""
        try:
            with self._conn() as conn:
                row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def get_all_trades(self, limit: int = 100) -> list[dict]:
        """Return recent trades as dicts (most recent first)."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT trade_id, date, code, name, action, price, shares, "
                    "amount, net_amount, pnl, pnl_pct, reason "
                    "FROM trades ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()

            return [
                {
                    "trade_id": r[0][:8], "日期": r[1], "代码": r[2], "名称": r[3],
                    "操作": "买入" if r[4] == "buy" else "卖出",
                    "价格": round(r[5], 4), "股数": r[6],
                    "金额": round(abs(r[8]), 2) if r[8] is not None else round(r[7], 2),
                    "盈亏": (round(r[9], 2) if r[9] is not None else "—"),
                    "盈亏%": (f"{r[10]:+.2%}" if r[10] is not None else "—"),
                    "原因": r[11] or "",
                }
                for r in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Daily equity snapshots
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        pm: PortfolioManager,
        snapshot_date: str,
        data_as_of: str | None = None,
    ) -> bool:
        """Upsert one end-of-day account valuation."""

        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO equity_snapshots
                        (snapshot_date, data_as_of, equity, cash, market_value,
                         realized_pnl, unrealized_pnl, total_return, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
                    """,
                    (
                        snapshot_date,
                        data_as_of,
                        pm.total_equity,
                        pm.cash,
                        pm.total_market_value,
                        pm.realized_pnl,
                        pm.total_unrealized_pnl,
                        pm.total_return_pct,
                    ),
                )
                conn.commit()
            return True
        except Exception:
            return False

    def get_equity_curve(self, limit: int = 1000) -> pd.DataFrame:
        """Return persisted daily account valuations in ascending date order."""

        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT snapshot_date, data_as_of, equity, cash, market_value,
                           realized_pnl, unrealized_pnl, total_return
                    FROM equity_snapshots
                    ORDER BY snapshot_date DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            columns = [
                "date", "data_as_of", "equity", "cash", "market_value",
                "realized_pnl", "unrealized_pnl", "total_return",
            ]
            result = pd.DataFrame(rows, columns=columns)
            if not result.empty:
                result["date"] = pd.to_datetime(result["date"])
                result = result.sort_values("date").reset_index(drop=True)
            return result
        except Exception:
            return pd.DataFrame(
                columns=[
                    "date", "data_as_of", "equity", "cash", "market_value",
                    "realized_pnl", "unrealized_pnl", "total_return",
                ]
            )

    # ------------------------------------------------------------------
    # Portable backup for ephemeral Streamlit hosting
    # ------------------------------------------------------------------

    def export_backup(self, pm: PortfolioManager) -> dict:
        """Return a JSON-serialisable account and equity-history backup."""

        curve = self.get_equity_curve()
        snapshots = []
        for row in curve.to_dict("records"):
            row["date"] = pd.Timestamp(row["date"]).date().isoformat()
            snapshots.append(row)
        return {
            "schema_version": 1,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "portfolio": pm.to_dict(),
            "equity_snapshots": snapshots,
        }

    def restore_backup(self, payload: dict) -> PortfolioManager:
        """Validate and replace the local account from an exported backup."""

        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("不支持的备份格式")
        portfolio_data = payload.get("portfolio")
        if not isinstance(portfolio_data, dict):
            raise ValueError("备份中缺少账户数据")
        try:
            pm = PortfolioManager.from_dict(portfolio_data)
            account_numbers = (
                float(pm.initial_capital),
                float(pm.cash),
                float(pm.commission_rate),
                float(pm.min_commission),
                float(pm.realized_pnl),
            )
            if not all(math.isfinite(value) for value in account_numbers):
                raise ValueError
            if pm.initial_capital <= 0 or pm.cash < 0:
                raise ValueError
            if pm.commission_rate < 0 or pm.min_commission < 0:
                raise ValueError
            for holding in pm.holdings.values():
                if holding.shares <= 0 or holding.shares % LOT_SIZE:
                    raise ValueError
                holding_numbers = (
                    float(holding.avg_cost),
                    float(holding.total_cost),
                    float(holding.current_price),
                    float(holding.highest_price),
                )
                if not all(math.isfinite(value) and value >= 0 for value in holding_numbers):
                    raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("备份中的账户数据无效") from error

        snapshots = payload.get("equity_snapshots", [])
        if not isinstance(snapshots, list):
            raise ValueError("备份中的净值记录无效")
        snapshot_rows = []
        try:
            for snapshot in snapshots:
                if not isinstance(snapshot, dict):
                    raise ValueError
                snapshot_date = datetime.fromisoformat(str(snapshot["date"])).date().isoformat()
                data_as_of = snapshot.get("data_as_of")
                if data_as_of:
                    data_as_of = datetime.fromisoformat(str(data_as_of)).date().isoformat()
                values = tuple(
                    float(snapshot[key])
                    for key in (
                        "equity",
                        "cash",
                        "market_value",
                        "realized_pnl",
                        "unrealized_pnl",
                        "total_return",
                    )
                )
                if not all(math.isfinite(value) for value in values):
                    raise ValueError
                if values[0] < 0 or values[1] < 0 or values[2] < 0:
                    raise ValueError
                snapshot_rows.append((snapshot_date, data_as_of, *values))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("备份中的净值记录无效") from error

        if not self.reset() or not self.save(pm):
            raise RuntimeError("恢复账户失败")
        try:
            with self._conn() as conn:
                for snapshot_row in snapshot_rows:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO equity_snapshots
                            (snapshot_date, data_as_of, equity, cash, market_value,
                             realized_pnl, unrealized_pnl, total_return)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        snapshot_row,
                    )
                conn.commit()
        except sqlite3.Error as error:
            raise RuntimeError("恢复净值记录失败") from error
        return pm

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> bool:
        """Drop all tables and re-create them. Returns True on success."""
        try:
            with self._conn() as conn:
                conn.execute("DROP TABLE IF EXISTS trade_cycles")
                conn.execute("DROP TABLE IF EXISTS trades")
                conn.execute("DROP TABLE IF EXISTS holdings")
                conn.execute("DROP TABLE IF EXISTS portfolio_state")
                conn.execute("DROP TABLE IF EXISTS equity_snapshots")
                conn.commit()
            self._init_tables()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Trade cycles
    # ------------------------------------------------------------------

    def create_cycle(self, cycle_id: str, code: str, name: str,
                     entry_date: str, entry_price: float, shares: int,
                     entry_amount: float, entry_reason: str = "") -> bool:
        """Create a new open trade cycle from a buy trade."""
        try:
            with self._conn() as conn:
                conn.execute(INSERT_CYCLE, (
                    cycle_id, code, name, entry_date, entry_price,
                    shares, entry_amount, entry_reason,
                ))
                conn.commit()
            return True
        except Exception:
            return False

    def close_cycle(self, cycle_id: str, exit_date: str, exit_price: float,
                    pnl: float, pnl_pct: float, holding_days: int,
                    exit_reason: str = "") -> bool:
        """Close a trade cycle with sell information."""
        try:
            with self._conn() as conn:
                conn.execute(CLOSE_CYCLE, (
                    exit_date, exit_price, pnl, pnl_pct,
                    holding_days, exit_reason, cycle_id,
                ))
                conn.commit()
            return True
        except Exception:
            return False

    def get_open_cycles(self) -> list[dict]:
        """Return all currently open (unclosed) trade cycles."""
        try:
            with self._conn() as conn:
                rows = conn.execute(SELECT_OPEN_CYCLES).fetchall()
            return [_cycle_row_to_dict(r) for r in rows]
        except Exception:
            return []

    def get_open_cycles_for_code(self, code: str) -> list[dict]:
        """Return open cycles for a specific code (FIFO-ordered)."""
        try:
            with self._conn() as conn:
                rows = conn.execute(SELECT_OPEN_CYCLES_FOR_CODE, (code,)).fetchall()
            return [_cycle_row_to_dict(r) for r in rows]
        except Exception:
            return []

    def get_closed_cycles(self, limit: int = 100) -> list[dict]:
        """Return most recent closed cycles."""
        try:
            with self._conn() as conn:
                rows = conn.execute(SELECT_CLOSED_CYCLES, (limit,)).fetchall()
            return [_cycle_row_to_dict(r) for r in rows]
        except Exception:
            return []

    def get_cycle_count(self) -> int:
        """Total number of trade cycles."""
        try:
            with self._conn() as conn:
                row = conn.execute(COUNT_CYCLES).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def get_closed_cycle_count(self) -> int:
        """Number of completed trade cycles."""
        try:
            with self._conn() as conn:
                row = conn.execute(COUNT_CLOSED_CYCLES).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    @property
    def db_path(self) -> str:
        return str(self._path)
