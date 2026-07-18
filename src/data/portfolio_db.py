"""Persistent storage for the single ETF-rotation paper account.

The store uses SQLite by default and switches to PostgreSQL when
``DATABASE_URL`` is configured.  The schema keeps an internal fixed account
key only to make singleton upserts portable across both database engines.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    delete,
    inspect,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from src.engine.portfolio import LOT_SIZE, ExecutedTrade, Holding, PortfolioManager


_DB_DIR = Path(__file__).resolve().parent / "portfolio_db"
_DB_PATH = _DB_DIR / "portfolio.sqlite3"
_ACCOUNT_ID = "default"


def _default_db_path() -> Path:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _sqlite_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.resolve().as_posix()}"


def _normalise_database_url(url: str) -> str:
    value = str(url or "").strip()
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value[len("postgres://") :]
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


METADATA = MetaData()

ACCOUNTS = Table(
    "paper_accounts",
    METADATA,
    Column("account_id", String(64), primary_key=True),
    Column("created_at", String(32), nullable=False),
    Column("updated_at", String(32), nullable=False),
)

PORTFOLIO_STATE = Table(
    "paper_portfolio_state",
    METADATA,
    Column("account_id", String(64), primary_key=True),
    Column("initial_capital", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("realized_pnl", Float, nullable=False),
    Column("commission_rate", Float, nullable=False),
    Column("min_commission", Float, nullable=False),
    Column("updated_at", String(32), nullable=False),
)

HOLDINGS = Table(
    "paper_holdings",
    METADATA,
    Column("account_id", String(64), primary_key=True),
    Column("code", String(12), primary_key=True),
    Column("name", String(120), nullable=False, default=""),
    Column("shares", Integer, nullable=False),
    Column("avg_cost", Float, nullable=False),
    Column("total_cost", Float, nullable=False),
    Column("current_price", Float, nullable=False),
    Column("entry_date", String(16), nullable=False, default=""),
    Column("highest_price", Float, nullable=False),
    Column("last_buy_date", String(16), nullable=False, default=""),
    Column("last_buy_shares", Integer, nullable=False),
    Column("rank_weak_days", Integer, nullable=False),
    Column("trend_weak_days", Integer, nullable=False),
    Column("last_signal_date", String(16), nullable=False, default=""),
)

TRADES = Table(
    "paper_trades",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("account_id", String(64), nullable=False, index=True),
    Column("trade_id", String(64), nullable=False),
    Column("date", String(16), nullable=False),
    Column("code", String(12), nullable=False),
    Column("name", String(120), nullable=False, default=""),
    Column("action", String(8), nullable=False),
    Column("price", Float, nullable=False),
    Column("shares", Integer, nullable=False),
    Column("amount", Float, nullable=False),
    Column("commission", Float, nullable=False),
    Column("net_amount", Float, nullable=False),
    Column("pnl", Float),
    Column("pnl_pct", Float),
    Column("reason", String(500), nullable=False, default=""),
    UniqueConstraint("account_id", "trade_id", name="uq_paper_trade_account_id"),
)

EQUITY_SNAPSHOTS = Table(
    "paper_equity_snapshots",
    METADATA,
    Column("account_id", String(64), primary_key=True),
    Column("snapshot_date", String(16), primary_key=True),
    Column("data_as_of", String(16)),
    Column("equity", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("market_value", Float, nullable=False),
    Column("realized_pnl", Float, nullable=False),
    Column("unrealized_pnl", Float, nullable=False),
    Column("total_return", Float, nullable=False),
    Column("created_at", String(32), nullable=False),
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class PortfolioDB:
    """SQLAlchemy-backed persistence for one paper portfolio."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        database_url: str | None = None,
    ):
        self._account_id = _ACCOUNT_ID
        self._path: Path | None = None
        self._migration_warning: str | None = None

        configured_url = database_url or os.getenv("DATABASE_URL")
        if db_path is not None:
            self._path = Path(db_path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            url = _sqlite_url(self._path)
        elif configured_url:
            url = _normalise_database_url(configured_url)
        else:
            self._path = _default_db_path()
            url = _sqlite_url(self._path)

        engine_options: dict = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                engine_options["poolclass"] = StaticPool
        self._engine: Engine = create_engine(url, **engine_options)
        METADATA.create_all(self._engine)
        self._ensure_account()
        self._migrate_legacy_sqlite()

    def _ensure_account(self) -> None:
        now = _now()
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(ACCOUNTS.c.account_id).where(
                    ACCOUNTS.c.account_id == self._account_id
                )
            ).first()
            if exists is None:
                conn.execute(
                    insert(ACCOUNTS).values(
                        account_id=self._account_id,
                        created_at=now,
                        updated_at=now,
                    )
                )

    def _migrate_legacy_sqlite(self) -> None:
        """Import the previous single-account SQLite schema once."""

        if self.backend != "sqlite" or self._account_id != "default":
            return
        if self.load() is not None:
            return
        tables = set(inspect(self._engine).get_table_names())
        if "portfolio_state" not in tables:
            return
        try:
            with self._engine.connect() as conn:
                state = conn.execute(
                    text(
                        "SELECT initial_capital, cash, realized_pnl, "
                        "commission_rate, min_commission "
                        "FROM portfolio_state WHERE id=1"
                    )
                ).first()
                if state is None:
                    return
                pm = PortfolioManager(
                    initial_capital=state[0],
                    commission_rate=state[3],
                    min_commission=state[4],
                )
                pm.cash = state[1]
                pm._realized_pnl = state[2]
                if "holdings" in tables:
                    rows = conn.execute(
                        text(
                            "SELECT code, name, shares, avg_cost, total_cost, "
                            "current_price, entry_date, highest_price, "
                            "last_buy_date, last_buy_shares, rank_weak_days, "
                            "trend_weak_days, last_signal_date FROM holdings"
                        )
                    ).fetchall()
                    for row in rows:
                        if row[2] > 0:
                            pm._holdings[row[0]] = Holding(
                                code=row[0],
                                name=row[1] or "",
                                shares=row[2],
                                avg_cost=row[3],
                                total_cost=row[4],
                                current_price=row[5],
                                entry_date=row[6] or "",
                                highest_price=row[7],
                                last_buy_date=row[8] or "",
                                last_buy_shares=row[9],
                                rank_weak_days=row[10],
                                trend_weak_days=row[11],
                                last_signal_date=row[12] or "",
                            )
                if "trades" in tables:
                    rows = conn.execute(
                        text(
                            "SELECT trade_id, date, code, name, action, price, "
                            "shares, amount, commission, net_amount, pnl, pnl_pct, "
                            "reason FROM trades ORDER BY id ASC"
                        )
                    ).fetchall()
                    for row in rows:
                        pm._trades.append(
                            ExecutedTrade(
                                trade_id=row[0],
                                date=row[1],
                                code=row[2],
                                name=row[3] or "",
                                action=row[4],
                                price=row[5],
                                shares=row[6],
                                amount=row[7],
                                commission=row[8],
                                net_amount=row[9],
                                pnl=row[10],
                                pnl_pct=row[11],
                                reason=row[12] or "",
                            )
                        )
            if not self.save(pm):
                self._migration_warning = "旧版SQLite账户自动迁移失败，请使用JSON备份恢复"
        except (SQLAlchemyError, TypeError, ValueError) as error:
            self._migration_warning = f"旧版SQLite账户未迁移：{error}"

    def save(self, pm: PortfolioManager) -> bool:
        """Persist the complete account in one transaction."""

        now = _now()
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    delete(PORTFOLIO_STATE).where(
                        PORTFOLIO_STATE.c.account_id == self._account_id
                    )
                )
                conn.execute(
                    insert(PORTFOLIO_STATE).values(
                        account_id=self._account_id,
                        initial_capital=pm.initial_capital,
                        cash=pm.cash,
                        realized_pnl=pm.realized_pnl,
                        commission_rate=pm.commission_rate,
                        min_commission=pm.min_commission,
                        updated_at=now,
                    )
                )
                conn.execute(
                    delete(HOLDINGS).where(HOLDINGS.c.account_id == self._account_id)
                )
                holding_rows = [
                    {
                        "account_id": self._account_id,
                        "code": code,
                        "name": holding.name,
                        "shares": holding.shares,
                        "avg_cost": holding.avg_cost,
                        "total_cost": holding.total_cost,
                        "current_price": holding.current_price,
                        "entry_date": holding.entry_date,
                        "highest_price": holding.highest_price,
                        "last_buy_date": holding.last_buy_date,
                        "last_buy_shares": holding.last_buy_shares,
                        "rank_weak_days": holding.rank_weak_days,
                        "trend_weak_days": holding.trend_weak_days,
                        "last_signal_date": holding.last_signal_date,
                    }
                    for code, holding in pm.holdings.items()
                    if holding.shares > 0
                ]
                if holding_rows:
                    conn.execute(insert(HOLDINGS), holding_rows)

                existing_ids = set(
                    conn.execute(
                        select(TRADES.c.trade_id).where(
                            TRADES.c.account_id == self._account_id
                        )
                    ).scalars()
                )
                trade_rows = [
                    {
                        "account_id": self._account_id,
                        "trade_id": trade.trade_id,
                        "date": trade.date,
                        "code": trade.code,
                        "name": trade.name,
                        "action": trade.action,
                        "price": trade.price,
                        "shares": trade.shares,
                        "amount": trade.amount,
                        "commission": trade.commission,
                        "net_amount": trade.net_amount,
                        "pnl": trade.pnl,
                        "pnl_pct": trade.pnl_pct,
                        "reason": trade.reason,
                    }
                    for trade in pm.trades
                    if trade.trade_id not in existing_ids
                ]
                if trade_rows:
                    conn.execute(insert(TRADES), trade_rows)
                conn.execute(
                    ACCOUNTS.update()
                    .where(ACCOUNTS.c.account_id == self._account_id)
                    .values(updated_at=now)
                )
            return True
        except SQLAlchemyError:
            return False

    def load(self) -> PortfolioManager | None:
        """Load the current account, or ``None`` before first creation."""

        try:
            with self._engine.connect() as conn:
                state = conn.execute(
                    select(PORTFOLIO_STATE).where(
                        PORTFOLIO_STATE.c.account_id == self._account_id
                    )
                ).mappings().first()
                if state is None:
                    return None
                pm = PortfolioManager(
                    initial_capital=state["initial_capital"],
                    commission_rate=state["commission_rate"],
                    min_commission=state["min_commission"],
                )
                pm.cash = state["cash"]
                pm._realized_pnl = state["realized_pnl"]
                holding_rows = conn.execute(
                    select(HOLDINGS)
                    .where(HOLDINGS.c.account_id == self._account_id)
                    .order_by(HOLDINGS.c.code)
                ).mappings()
                for row in holding_rows:
                    pm._holdings[row["code"]] = Holding(
                        code=row["code"],
                        name=row["name"] or "",
                        shares=row["shares"],
                        avg_cost=row["avg_cost"],
                        total_cost=row["total_cost"],
                        current_price=row["current_price"],
                        entry_date=row["entry_date"] or "",
                        highest_price=row["highest_price"],
                        last_buy_date=row["last_buy_date"] or "",
                        last_buy_shares=row["last_buy_shares"],
                        rank_weak_days=row["rank_weak_days"],
                        trend_weak_days=row["trend_weak_days"],
                        last_signal_date=row["last_signal_date"] or "",
                    )
                trade_rows = conn.execute(
                    select(TRADES)
                    .where(TRADES.c.account_id == self._account_id)
                    .order_by(TRADES.c.id)
                ).mappings()
                for row in trade_rows:
                    pm._trades.append(
                        ExecutedTrade(
                            trade_id=row["trade_id"],
                            date=row["date"],
                            code=row["code"],
                            name=row["name"] or "",
                            action=row["action"],
                            price=row["price"],
                            shares=row["shares"],
                            amount=row["amount"],
                            commission=row["commission"],
                            net_amount=row["net_amount"],
                            pnl=row["pnl"],
                            pnl_pct=row["pnl_pct"],
                            reason=row["reason"] or "",
                        )
                    )
                return pm
        except SQLAlchemyError:
            return None

    def get_trade_count(self) -> int:
        try:
            with self._engine.connect() as conn:
                return len(
                    conn.execute(
                        select(TRADES.c.trade_id).where(
                            TRADES.c.account_id == self._account_id
                        )
                    ).all()
                )
        except SQLAlchemyError:
            return 0

    def get_all_trades(self, limit: int = 100) -> list[dict]:
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(TRADES)
                    .where(TRADES.c.account_id == self._account_id)
                    .order_by(TRADES.c.id.desc())
                    .limit(max(1, int(limit)))
                ).mappings().all()
            return [
                {
                    "trade_id": row["trade_id"][:8],
                    "日期": row["date"],
                    "代码": row["code"],
                    "名称": row["name"],
                    "操作": "买入" if row["action"] == "buy" else "卖出",
                    "价格": round(row["price"], 4),
                    "股数": row["shares"],
                    "金额": round(abs(row["net_amount"]), 2),
                    "盈亏": round(row["pnl"], 2) if row["pnl"] is not None else "—",
                    "盈亏%": f"{row['pnl_pct']:+.2%}" if row["pnl_pct"] is not None else "—",
                    "原因": row["reason"] or "",
                }
                for row in rows
            ]
        except SQLAlchemyError:
            return []

    def record_snapshot(
        self,
        pm: PortfolioManager,
        snapshot_date: str,
        data_as_of: str | None = None,
    ) -> bool:
        row = {
            "account_id": self._account_id,
            "snapshot_date": snapshot_date,
            "data_as_of": data_as_of,
            "equity": pm.total_equity,
            "cash": pm.cash,
            "market_value": pm.total_market_value,
            "realized_pnl": pm.realized_pnl,
            "unrealized_pnl": pm.total_unrealized_pnl,
            "total_return": pm.total_return_pct,
            "created_at": _now(),
        }
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    delete(EQUITY_SNAPSHOTS).where(
                        (EQUITY_SNAPSHOTS.c.account_id == self._account_id)
                        & (EQUITY_SNAPSHOTS.c.snapshot_date == snapshot_date)
                    )
                )
                conn.execute(insert(EQUITY_SNAPSHOTS).values(**row))
            return True
        except SQLAlchemyError:
            return False

    def get_equity_curve(self, limit: int = 1000) -> pd.DataFrame:
        columns = [
            "date",
            "data_as_of",
            "equity",
            "cash",
            "market_value",
            "realized_pnl",
            "unrealized_pnl",
            "total_return",
        ]
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(
                        EQUITY_SNAPSHOTS.c.snapshot_date,
                        EQUITY_SNAPSHOTS.c.data_as_of,
                        EQUITY_SNAPSHOTS.c.equity,
                        EQUITY_SNAPSHOTS.c.cash,
                        EQUITY_SNAPSHOTS.c.market_value,
                        EQUITY_SNAPSHOTS.c.realized_pnl,
                        EQUITY_SNAPSHOTS.c.unrealized_pnl,
                        EQUITY_SNAPSHOTS.c.total_return,
                    )
                    .where(EQUITY_SNAPSHOTS.c.account_id == self._account_id)
                    .order_by(EQUITY_SNAPSHOTS.c.snapshot_date.desc())
                    .limit(max(1, int(limit)))
                ).all()
            result = pd.DataFrame(rows, columns=columns)
            if not result.empty:
                result["date"] = pd.to_datetime(result["date"])
                result = result.sort_values("date").reset_index(drop=True)
            return result
        except SQLAlchemyError:
            return pd.DataFrame(columns=columns)

    def export_backup(self, pm: PortfolioManager) -> dict:
        curve = self.get_equity_curve()
        snapshots = []
        for row in curve.to_dict("records"):
            row["date"] = pd.Timestamp(row["date"]).date().isoformat()
            snapshots.append(row)
        return {
            "schema_version": 2,
            "exported_at": _now(),
            "portfolio": pm.to_dict(),
            "equity_snapshots": snapshots,
        }

    @staticmethod
    def _validated_backup(payload: dict) -> tuple[PortfolioManager, list[dict]]:
        if not isinstance(payload, dict) or payload.get("schema_version") not in (1, 2):
            raise ValueError("不支持的备份格式")
        portfolio_data = payload.get("portfolio")
        if not isinstance(portfolio_data, dict):
            raise ValueError("备份中缺少账户数据")
        try:
            pm = PortfolioManager.from_dict(portfolio_data)
            account_values = (
                float(pm.initial_capital),
                float(pm.cash),
                float(pm.commission_rate),
                float(pm.min_commission),
                float(pm.realized_pnl),
            )
            if not all(math.isfinite(value) for value in account_values):
                raise ValueError
            if pm.initial_capital <= 0 or pm.cash < 0:
                raise ValueError
            if pm.commission_rate < 0 or pm.min_commission < 0:
                raise ValueError
            for holding in pm.holdings.values():
                if holding.shares <= 0 or holding.shares % LOT_SIZE:
                    raise ValueError
                numbers = (
                    float(holding.avg_cost),
                    float(holding.total_cost),
                    float(holding.current_price),
                    float(holding.highest_price),
                )
                if not all(math.isfinite(value) and value >= 0 for value in numbers):
                    raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("备份中的账户数据无效") from error

        snapshots = payload.get("equity_snapshots", [])
        if not isinstance(snapshots, list):
            raise ValueError("备份中的净值记录无效")
        validated = []
        try:
            for snapshot in snapshots:
                snapshot_date = datetime.fromisoformat(str(snapshot["date"])).date().isoformat()
                data_as_of = snapshot.get("data_as_of")
                if data_as_of:
                    data_as_of = datetime.fromisoformat(str(data_as_of)).date().isoformat()
                keys = (
                    "equity",
                    "cash",
                    "market_value",
                    "realized_pnl",
                    "unrealized_pnl",
                    "total_return",
                )
                values = {key: float(snapshot[key]) for key in keys}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError
                if values["equity"] < 0 or values["cash"] < 0 or values["market_value"] < 0:
                    raise ValueError
                validated.append(
                    {
                        "snapshot_date": snapshot_date,
                        "data_as_of": data_as_of,
                        **values,
                    }
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("备份中的净值记录无效") from error
        return pm, validated

    def restore_backup(self, payload: dict) -> PortfolioManager:
        pm, snapshots = self._validated_backup(payload)
        if not self.reset() or not self.save(pm):
            raise RuntimeError("恢复账户失败")
        try:
            with self._engine.begin() as conn:
                for snapshot in snapshots:
                    conn.execute(
                        insert(EQUITY_SNAPSHOTS).values(
                            account_id=self._account_id,
                            created_at=_now(),
                            **snapshot,
                        )
                    )
        except SQLAlchemyError as error:
            raise RuntimeError("恢复净值记录失败") from error
        return pm

    def reset(self) -> bool:
        """Delete only the active account; never affect other users."""

        try:
            with self._engine.begin() as conn:
                for table in (EQUITY_SNAPSHOTS, TRADES, HOLDINGS, PORTFOLIO_STATE):
                    conn.execute(
                        delete(table).where(table.c.account_id == self._account_id)
                    )
                conn.execute(
                    ACCOUNTS.update()
                    .where(ACCOUNTS.c.account_id == self._account_id)
                    .values(updated_at=_now())
                )
            return True
        except SQLAlchemyError:
            return False

    def healthcheck(self) -> tuple[bool, str]:
        try:
            with self._engine.connect() as conn:
                conn.execute(select(ACCOUNTS.c.account_id).limit(1)).first()
            return True, f"{self.backend}连接正常"
        except SQLAlchemyError as error:
            return False, str(error)

    @property
    def backend(self) -> str:
        return self._engine.dialect.name

    @property
    def is_cloud_persistent(self) -> bool:
        return self.backend == "postgresql"

    @property
    def migration_warning(self) -> str | None:
        return self._migration_warning

    @property
    def db_path(self) -> str:
        if self._path is not None:
            return str(self._path)
        return self._engine.url.render_as_string(hide_password=True)
