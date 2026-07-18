"""Tests for completed-bar and confirmation-window safety rules."""

from datetime import date, datetime

import pandas as pd

from src.engine.trading_schedule import (
    confirmation_window,
    drop_incomplete_daily_bar,
    is_trading_day,
    next_trading_day,
    quote_matches_trade_date,
    realtime_price_for_action,
    validate_realtime_quote,
)


def test_current_day_bar_is_removed_before_settlement_time():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-17", "2026-07-20"]),
            "close": [1.0, 1.1],
        }
    )

    filtered = drop_incomplete_daily_bar(
        history, datetime(2026, 7, 20, 14, 59)
    )

    assert filtered["date"].dt.date.tolist() == [date(2026, 7, 17)]


def test_current_day_bar_is_kept_after_settlement_time():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-17", "2026-07-20"]),
            "close": [1.0, 1.1],
        }
    )

    filtered = drop_incomplete_daily_bar(
        history, datetime(2026, 7, 20, 15, 6)
    )

    assert len(filtered) == 2


def test_confirmation_window_recommends_next_session_morning():
    status = confirmation_window(
        date(2026, 7, 17), datetime(2026, 7, 20, 9, 40)
    )

    assert status.can_confirm
    assert status.recommended
    assert status.state == "recommended"


def test_confirmation_window_blocks_same_day_weekend_and_after_hours():
    same_day = confirmation_window(
        date(2026, 7, 20), datetime(2026, 7, 20, 15, 10)
    )
    weekend = confirmation_window(
        date(2026, 7, 17), datetime(2026, 7, 19, 9, 40)
    )
    after_hours = confirmation_window(
        date(2026, 7, 17), datetime(2026, 7, 20, 15, 1)
    )

    assert not same_day.can_confirm
    assert not weekend.can_confirm
    assert not after_hours.can_confirm


def test_confirmation_window_allows_fallback_continuous_auction_times():
    morning = confirmation_window(
        date(2026, 7, 17), datetime(2026, 7, 20, 10, 30)
    )
    afternoon = confirmation_window(
        date(2026, 7, 17), datetime(2026, 7, 20, 14, 30)
    )

    assert morning.can_confirm and not morning.recommended
    assert afternoon.can_confirm and not afternoon.recommended


def test_live_price_requires_today_quote_and_uses_correct_book_side():
    quote = {
        "date": "2026-07-20",
        "current_price": 4.0,
        "bid1_price": 3.99,
        "ask1_price": 4.01,
    }

    assert quote_matches_trade_date(quote, date(2026, 7, 20))
    assert realtime_price_for_action(quote, "buy", date(2026, 7, 20)) == 4.01
    assert realtime_price_for_action(quote, "sell", date(2026, 7, 20)) == 3.99
    assert realtime_price_for_action(quote, "buy", date(2026, 7, 21)) is None


def test_exchange_calendar_blocks_official_holiday_and_finds_next_session():
    assert not is_trading_day(date(2026, 10, 1))
    assert next_trading_day(date(2026, 9, 30)) == date(2026, 10, 8)


def test_realtime_quote_gate_checks_age_depth_limits_and_deviation():
    base = {
        "date": "2026-07-20",
        "time": "09:39:50",
        "bid1_price": 3.99,
        "bid1_volume": 100,
        "ask1_price": 4.01,
        "ask1_volume": 100,
        "limit_up": 4.4,
        "limit_down": 3.6,
    }
    now = datetime(2026, 7, 20, 9, 40, 0)

    valid = validate_realtime_quote(
        base, "buy", date(2026, 7, 20), 5000, 4.0, now=now
    )
    stale = validate_realtime_quote(
        {**base, "time": "09:39:00"},
        "buy",
        date(2026, 7, 20),
        5000,
        4.0,
        now=now,
    )
    shallow = validate_realtime_quote(
        {**base, "ask1_volume": 10},
        "buy",
        date(2026, 7, 20),
        5000,
        4.0,
        now=now,
    )
    deviated = validate_realtime_quote(
        {**base, "ask1_price": 4.2},
        "buy",
        date(2026, 7, 20),
        5000,
        4.0,
        now=now,
    )

    assert valid.valid and valid.price == 4.01
    assert valid.available_shares == 10_000
    assert not stale.valid and "过期" in stale.reason
    assert not shallow.valid and "挂单量" in shallow.reason
    assert not deviated.valid and "偏离" in deviated.reason
