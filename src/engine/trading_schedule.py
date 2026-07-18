"""Shanghai-time data freshness and paper-trading window rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
DAILY_BAR_READY_TIME = time(15, 5)
RECOMMENDED_START = time(9, 35)
RECOMMENDED_END = time(10, 0)
MORNING_END = time(11, 25)
AFTERNOON_START = time(13, 5)
AFTERNOON_END = time(14, 50)


@dataclass(frozen=True)
class ConfirmationWindow:
    """Whether a signal may be confirmed at the current Shanghai time."""

    state: str
    message: str
    can_confirm: bool
    recommended: bool


def shanghai_now(now: datetime | None = None) -> datetime:
    """Return an aware Asia/Shanghai timestamp, also normalising test inputs."""

    if now is None:
        return datetime.now(SHANGHAI_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI_TZ)
    return now.astimezone(SHANGHAI_TZ)


def drop_incomplete_daily_bar(
    history: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Remove today's bar until the cash-market close has settled.

    Some public quote sources expose a changing current-day row as a daily bar.
    A close-based strategy must not rank on that partial row.
    """

    if history is None or history.empty or "date" not in history.columns:
        return history.copy() if history is not None else pd.DataFrame()
    frame = history.copy()
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    current = shanghai_now(now)
    if current.time() < DAILY_BAR_READY_TIME:
        frame = frame.loc[parsed_dates.dt.date != current.date()].copy()
    return frame.reset_index(drop=True)


def confirmation_window(
    signal_date: date | None,
    now: datetime | None = None,
) -> ConfirmationWindow:
    """Apply the recommended and fallback confirmation windows.

    Exchange holidays are ultimately verified by requiring a quote stamped with
    today's date.  Weekends and obviously premature same-day signals are blocked
    here before any quote request is made.
    """

    current = shanghai_now(now)
    if signal_date is None:
        return ConfirmationWindow(
            "unavailable", "没有完整日线信号，暂不能确认交易。", False, False
        )
    if current.date() <= signal_date:
        return ConfirmationWindow(
            "waiting",
            "信号需等到下一个开市日 09:35–10:00（北京时间）再确认。",
            False,
            False,
        )
    if current.weekday() >= 5:
        return ConfirmationWindow(
            "closed", "今天是周末，请在下一个开市日 09:35–10:00 确认。", False, False
        )

    current_time = current.time().replace(tzinfo=None)
    if RECOMMENDED_START <= current_time <= RECOMMENDED_END:
        return ConfirmationWindow(
            "recommended",
            "当前处于首选确认窗口 09:35–10:00，可用实时盘口复核后确认。",
            True,
            True,
        )
    if (
        RECOMMENDED_END < current_time <= MORNING_END
        or AFTERNOON_START <= current_time <= AFTERNOON_END
    ):
        return ConfirmationWindow(
            "allowed",
            "当前仍可确认，但已不在首选窗口；系统会先用实时盘口重新计算份额。",
            True,
            False,
        )
    return ConfirmationWindow(
        "closed",
        "当前不在确认时段。请优先在开市日 09:35–10:00 确认；备用时段为 10:00–11:25、13:05–14:50。",
        False,
        False,
    )


def quote_matches_trade_date(quote: dict | None, trade_date: date) -> bool:
    """Return True only for a quote explicitly stamped with the trade date."""

    if not quote or not quote.get("date"):
        return False
    parsed = pd.to_datetime(quote.get("date"), errors="coerce")
    return bool(pd.notna(parsed) and parsed.date() == trade_date)


def realtime_price_for_action(
    quote: dict | None,
    action: str,
    trade_date: date,
) -> float | None:
    """Select ask-one for buys and bid-one for sells, with last-price fallback."""

    if not quote_matches_trade_date(quote, trade_date):
        return None
    field = "ask1_price" if action == "buy" else "bid1_price"
    for key in (field, "current_price"):
        price = pd.to_numeric((quote or {}).get(key), errors="coerce")
        if pd.notna(price) and float(price) > 0:
            return float(price)
    return None
