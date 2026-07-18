"""Shanghai-time data freshness and paper-trading window rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
DAILY_BAR_READY_TIME = time(15, 5)
RECOMMENDED_START = time(9, 35)
RECOMMENDED_END = time(10, 0)
MORNING_END = time(11, 25)
AFTERNOON_START = time(13, 5)
AFTERNOON_END = time(14, 50)
MAX_QUOTE_AGE_SECONDS = 30
MAX_PRICE_DEVIATION = 0.03
_XSHG = xcals.get_calendar("XSHG")


@dataclass(frozen=True)
class ConfirmationWindow:
    """Whether a signal may be confirmed at the current Shanghai time."""

    state: str
    message: str
    can_confirm: bool
    recommended: bool


@dataclass(frozen=True)
class QuoteValidation:
    """Executable price and the result of all real-time quote gates."""

    valid: bool
    price: float | None
    reason: str
    quote_age_seconds: float | None = None
    available_shares: int | None = None


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


def is_trading_day(value: date) -> bool:
    """Return whether Shanghai Stock Exchange has a regular session."""

    try:
        return bool(_XSHG.is_session(pd.Timestamp(value)))
    except (ValueError, TypeError):
        return False


def next_trading_day(value: date) -> date:
    """Return the first exchange session strictly after ``value``."""

    timestamp = pd.Timestamp(value)
    try:
        if _XSHG.is_session(timestamp):
            return _XSHG.next_session(timestamp).date()
        return _XSHG.date_to_session(timestamp, direction="next").date()
    except (ValueError, TypeError, IndexError):
        candidate = value
        while True:
            candidate += timedelta(days=1)
            if candidate.weekday() < 5:
                return candidate


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
    expected_trade_date = next_trading_day(signal_date)
    if current.date() < expected_trade_date:
        return ConfirmationWindow(
            "waiting",
            f"信号需等到 {expected_trade_date.isoformat()} 09:35–10:00（北京时间）再确认。",
            False,
            False,
        )
    if current.date() > expected_trade_date:
        return ConfirmationWindow(
            "expired",
            "已错过该信号的下一开市日确认窗口，请等待最新完整日线重新生成信号。",
            False,
            False,
        )
    if not is_trading_day(current.date()):
        following = next_trading_day(current.date())
        return ConfirmationWindow(
            "closed",
            f"今天休市，请在 {following.isoformat()} 09:35–10:00 确认。",
            False,
            False,
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


def _quote_datetime(quote: dict | None) -> datetime | None:
    if not quote or not quote.get("date") or not quote.get("time"):
        return None
    parsed = pd.to_datetime(
        f"{quote.get('date')} {quote.get('time')}", errors="coerce"
    )
    if pd.isna(parsed):
        return None
    return shanghai_now(parsed.to_pydatetime())


def validate_realtime_quote(
    quote: dict | None,
    action: str,
    trade_date: date,
    requested_shares: int,
    reference_price: float,
    now: datetime | None = None,
    max_age_seconds: int = MAX_QUOTE_AGE_SECONDS,
    max_price_deviation: float = MAX_PRICE_DEVIATION,
) -> QuoteValidation:
    """Validate quote date, age, book depth, limits, and price deviation."""

    current = shanghai_now(now)
    if not quote_matches_trade_date(quote, trade_date):
        return QuoteValidation(False, None, "盘口不是确认日行情")
    quote_time = _quote_datetime(quote)
    if quote_time is None:
        return QuoteValidation(False, None, "盘口缺少可验证的时间戳")
    quote_age = (current - quote_time).total_seconds()
    if quote_age < -5 or quote_age > max_age_seconds:
        return QuoteValidation(
            False,
            None,
            f"盘口已过期（{quote_age:.0f}秒）",
            quote_age_seconds=quote_age,
        )

    field = "ask1_price" if action == "buy" else "bid1_price"
    volume_field = "ask1_volume" if action == "buy" else "bid1_volume"
    price = pd.to_numeric((quote or {}).get(field), errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return QuoteValidation(False, None, "买一/卖一盘口价格无效", quote_age)
    price = float(price)

    book_lots = pd.to_numeric((quote or {}).get(volume_field), errors="coerce")
    available_shares = (
        int(float(book_lots) * 100)
        if pd.notna(book_lots) and float(book_lots) >= 0
        else None
    )
    if available_shares is None or available_shares < max(0, requested_shares):
        return QuoteValidation(
            False,
            None,
            "买一/卖一挂单量不足以覆盖模拟订单",
            quote_age,
            available_shares,
        )

    limit_up = pd.to_numeric((quote or {}).get("limit_up"), errors="coerce")
    limit_down = pd.to_numeric((quote or {}).get("limit_down"), errors="coerce")
    if (
        action == "buy"
        and pd.notna(limit_up)
        and float(limit_up) > 0
        and price >= float(limit_up) * 0.9999
    ):
        return QuoteValidation(False, None, "接近涨停价，禁止自动模拟买入", quote_age, available_shares)
    if (
        action == "sell"
        and pd.notna(limit_down)
        and float(limit_down) > 0
        and price <= float(limit_down) * 1.0001
    ):
        return QuoteValidation(False, None, "接近跌停价，禁止自动模拟卖出", quote_age, available_shares)

    reference = pd.to_numeric(reference_price, errors="coerce")
    if pd.notna(reference) and float(reference) > 0:
        deviation = abs(price / float(reference) - 1.0)
        if deviation > max_price_deviation:
            return QuoteValidation(
                False,
                None,
                f"盘口偏离信号价{deviation:.2%}，超过{max_price_deviation:.0%}",
                quote_age,
                available_shares,
            )
    return QuoteValidation(True, price, "实时盘口门禁通过", quote_age, available_shares)


def realtime_price_for_action(
    quote: dict | None,
    action: str,
    trade_date: date,
) -> float | None:
    """Select ask-one for buys and bid-one for sells after date validation.

    This compatibility helper intentionally performs fewer checks than
    :func:`validate_realtime_quote`; execution paths must use the latter.
    """

    if not quote_matches_trade_date(quote, trade_date):
        return None
    field = "ask1_price" if action == "buy" else "bid1_price"
    for key in (field, "current_price"):
        price = pd.to_numeric((quote or {}).get(key), errors="coerce")
        if pd.notna(price) and float(price) > 0:
            return float(price)
    return None
