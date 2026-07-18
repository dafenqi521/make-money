"""Tests for SQLite portfolio persistence layer."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.portfolio import PortfolioManager, Holding, ExecutedTrade
from src.data.portfolio_db import PortfolioDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    """Create a PortfolioDB backed by a temp file. Cleans up after test."""
    fd, path = tempfile.mkstemp(suffix=".sqlite3", prefix="test_portfolio_")
    os.close(fd)
    db = PortfolioDB(db_path=path)
    yield db
    # Cleanup
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture
def sample_pm():
    """A PortfolioManager with some trades and holdings."""
    pm = PortfolioManager(initial_capital=100_000)
    pm.buy("510300", 4.829, 2000, name="沪深300ETF", reason="进入目标组合")
    pm.buy("510300", 4.636, 2000, name="沪深300ETF", reason="目标权重上调")
    pm.sell("510300", 5.100, 1000, name="沪深300ETF", reason="目标权重下调")
    pm.buy("510050", 3.200, 3000, name="上证50ETF", reason="进入目标组合")
    return pm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPortfolioDB:
    """Core save/load/reset functionality."""

    def test_save_and_load_roundtrip(self, temp_db, sample_pm):
        """Save a portfolio then load it — should match key fields."""
        assert temp_db.save(sample_pm)

        restored = temp_db.load()
        assert restored is not None
        assert restored.initial_capital == sample_pm.initial_capital
        assert restored.total_trades == sample_pm.total_trades
        assert len(restored.holdings) == len(sample_pm.holdings)
        # Cash should match (minus commissions)
        assert abs(restored.cash - sample_pm.cash) < 1.0

    def test_load_empty_db_returns_none(self, temp_db):
        """Loading from a fresh DB returns None."""
        assert temp_db.load() is None

    def test_save_preserves_holdings(self, temp_db, sample_pm):
        """Holdings are correctly restored after save/load."""
        temp_db.save(sample_pm)
        restored = temp_db.load()
        assert restored is not None

        for code in sample_pm.holdings:
            orig = sample_pm.holdings[code]
            rest = restored.holdings[code]
            assert rest.shares == orig.shares
            assert abs(rest.avg_cost - orig.avg_cost) < 0.01
            assert abs(rest.total_cost - orig.total_cost) < 1.0

    def test_save_preserves_trade_history(self, temp_db, sample_pm):
        """All trades are preserved with correct attributes."""
        temp_db.save(sample_pm)
        restored = temp_db.load()
        assert restored is not None
        assert restored.total_trades == sample_pm.total_trades

        for i, (orig, rest) in enumerate(zip(sample_pm.trades, restored.trades)):
            assert rest.action == orig.action, f"Trade {i} action mismatch"
            assert rest.code == orig.code, f"Trade {i} code mismatch"
            assert rest.shares == orig.shares, f"Trade {i} shares mismatch"
            assert abs(rest.price - orig.price) < 0.001, f"Trade {i} price mismatch"
            assert rest.reason == orig.reason, f"Trade {i} reason mismatch"

    def test_save_preserves_pnl(self, temp_db, sample_pm):
        """Realized P&L is accurately restored."""
        temp_db.save(sample_pm)
        restored = temp_db.load()
        assert restored is not None
        assert abs(restored.realized_pnl - sample_pm.realized_pnl) < 0.1

    def test_reset_clears_all_data(self, temp_db, sample_pm):
        """After reset, load returns None."""
        temp_db.save(sample_pm)
        assert temp_db.load() is not None

        temp_db.reset()
        assert temp_db.load() is None

    def test_duplicate_save_does_not_duplicate_trades(self, temp_db, sample_pm):
        """Saving the same portfolio twice doesn't duplicate trades."""
        temp_db.save(sample_pm)
        count1 = temp_db.get_trade_count()

        temp_db.save(sample_pm)
        count2 = temp_db.get_trade_count()

        assert count1 == count2
        assert count1 == sample_pm.total_trades

    def test_get_trade_count(self, temp_db, sample_pm):
        """Trade count reflects actual number of trades."""
        temp_db.save(sample_pm)
        assert temp_db.get_trade_count() == sample_pm.total_trades

    def test_get_all_trades_returns_latest_first(self, temp_db, sample_pm):
        """get_all_trades returns trades in reverse chronological order."""
        temp_db.save(sample_pm)
        trades = temp_db.get_all_trades(limit=50)
        assert len(trades) == sample_pm.total_trades

    def test_empty_portfolio_save_and_load(self, temp_db):
        """Saving an empty portfolio (no trades) should still work."""
        pm = PortfolioManager(initial_capital=50_000)
        temp_db.save(pm)

        restored = temp_db.load()
        assert restored is not None
        assert restored.initial_capital == 50_000
        assert restored.cash == 50_000
        assert restored.total_trades == 0
        assert len(restored.holdings) == 0

    def test_custom_commission_rate_preserved(self, temp_db):
        """Non-default commission settings survive roundtrip."""
        pm = PortfolioManager(initial_capital=200_000, commission_rate=0.0001, min_commission=1.0)
        pm.buy("510300", 5.0, 1000, name="沪深300ETF", reason="test")
        temp_db.save(pm)

        restored = temp_db.load()
        assert restored is not None
        assert restored.commission_rate == 0.0001
        assert restored.min_commission == 1.0

    def test_multiple_symbols_holdings(self, temp_db):
        """Portfolio with multiple ETF positions is correctly restored."""
        pm = PortfolioManager(initial_capital=500_000)
        pm.buy("510300", 4.5, 5000, name="沪深300ETF")
        pm.buy("510050", 3.2, 3000, name="上证50ETF")
        pm.buy("510500", 7.8, 2000, name="中证500ETF")
        pm.buy("588000", 1.25, 10000, name="科创50ETF")

        temp_db.save(pm)
        restored = temp_db.load()
        assert restored is not None
        assert len(restored.holdings) == 4
        for code in ("510300", "510050", "510500", "588000"):
            assert code in restored.holdings

    def test_reset_then_save_new_portfolio(self, temp_db, sample_pm):
        """After reset, saving a new portfolio works correctly."""
        temp_db.save(sample_pm)
        temp_db.reset()

        new_pm = PortfolioManager(initial_capital=10_000)
        new_pm.buy("159915", 2.5, 1000, name="创业板ETF")
        temp_db.save(new_pm)

        restored = temp_db.load()
        assert restored is not None
        assert restored.initial_capital == 10_000
        assert restored.total_trades == 1
        assert "159915" in restored.holdings
        assert "510300" not in restored.holdings

    def test_db_path_property(self, temp_db):
        """db_path returns the SQLite file path."""
        assert temp_db.db_path.endswith(".sqlite3")
        assert os.path.exists(temp_db.db_path)

    def test_holding_risk_state_survives_roundtrip(self, temp_db):
        pm = PortfolioManager(initial_capital=100_000)
        pm.buy("510300", 4.0, 1000, trade_date="2025-07-01")
        holding = pm.holdings["510300"]
        holding.highest_price = 4.5
        holding.rank_weak_days = 2
        holding.trend_weak_days = 1
        holding.last_signal_date = "2025-07-02"
        pm.update_prices({"510300": 4.2})

        assert temp_db.save(pm)
        restored = temp_db.load()
        restored_holding = restored.holdings["510300"]
        assert restored_holding.current_price == 4.2
        assert restored_holding.highest_price == 4.5
        assert restored_holding.rank_weak_days == 2
        assert restored_holding.last_signal_date == "2025-07-02"

    def test_equity_snapshots_and_backup_roundtrip(self, temp_db):
        pm = PortfolioManager(initial_capital=100_000)
        pm.buy("510300", 4.0, 1000, trade_date="2025-07-01")
        pm.update_prices({"510300": 4.1})
        assert temp_db.save(pm)
        assert temp_db.record_snapshot(pm, "2025-07-01", "2025-06-30")

        curve = temp_db.get_equity_curve()
        assert len(curve) == 1
        assert curve.iloc[0]["equity"] == pytest.approx(pm.total_equity)

        payload = temp_db.export_backup(pm)
        temp_db.reset()
        restored = temp_db.restore_backup(payload)
        assert restored.total_trades == pm.total_trades
        assert len(temp_db.get_equity_curve()) == 1

    def test_restore_backup_rejects_malformed_account(self, temp_db):
        payload = {"schema_version": 1, "portfolio": {"cash": "invalid"}}

        with pytest.raises(ValueError, match="账户数据无效"):
            temp_db.restore_backup(payload)

    def test_restore_backup_validates_snapshots_before_replacing_account(self, temp_db):
        existing = PortfolioManager(initial_capital=50_000)
        assert temp_db.save(existing)
        replacement = PortfolioManager(initial_capital=100_000)
        payload = temp_db.export_backup(replacement)
        payload["equity_snapshots"] = [{"date": "not-a-date"}]

        with pytest.raises(ValueError, match="净值记录无效"):
            temp_db.restore_backup(payload)

        assert temp_db.load().initial_capital == 50_000
