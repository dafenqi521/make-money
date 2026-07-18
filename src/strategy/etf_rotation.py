"""Exchange-traded ETF trend-rotation strategy primitives.

This module contains platform-independent selection and risk logic.  It is
deliberately kept separate from JoinQuant APIs so the ranking rules can be
unit-tested locally and reused by the Streamlit application, a backtester, or
an execution adapter.

Scope
-----
* Shanghai/Shenzhen secondary-market ETFs only.
* No creation/redemption, LOF arbitrage, leverage, or intraday round trips.
* Signals are calculated from completed daily bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RotationConfig:
    """Configuration for ETF universe filtering, ranking, and risk control."""

    min_history_days: int = 121
    min_listed_days: int = 120
    liquidity_lookback: int = 20
    min_avg_amount: float = 30_000_000.0
    min_daily_amount: float = 5_000_000.0
    max_positions: int = 4
    cash_reserve: float = 0.10
    max_position_weight: float = 0.30
    correlation_lookback: int = 60
    correlation_threshold: float = 0.90
    min_hold_days: int = 5
    rank_exit: int = 8
    rank_exit_confirm_days: int = 2
    trend_exit_confirm_days: int = 2
    time_stop_days: int = 10
    time_stop_peak_return: float = 0.015
    trailing_activate_return: float = 0.02
    category_position_limits: Mapping[str, int] = field(
        default_factory=lambda: {
            "domestic_broad": 2,
            "domestic_sector": 1,
            "overseas_equity": 1,
            "commodity": 1,
            "bond": 1,
            "other": 1,
        }
    )
    category_weight_caps: Mapping[str, float] = field(
        default_factory=lambda: {
            "domestic_broad": 0.55,
            "domestic_sector": 0.25,
            "overseas_equity": 0.30,
            "commodity": 0.25,
            "bond": 0.40,
            "other": 0.25,
        }
    )
    excluded_name_keywords: tuple[str, ...] = (
        "货币",
        "现金",
        "理财",
        "分级",
        "杠杆",
        "反向",
        "REIT",
    )

    def __post_init__(self) -> None:
        if self.min_history_days < 121:
            raise ValueError("min_history_days must be at least 121")
        if not 0 <= self.cash_reserve < 1:
            raise ValueError("cash_reserve must be in [0, 1)")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")


@dataclass(frozen=True)
class PositionState:
    """State required to evaluate a held ETF's exit rules."""

    code: str
    entry_price: float
    highest_price: float
    holding_days: int
    rank_weak_days: int = 0
    trend_weak_days: int = 0
    closeable: bool = True


@dataclass(frozen=True)
class ExitDecision:
    """A deterministic exit decision with a machine-readable reason code."""

    should_exit: bool
    code: str
    reason_code: str
    reason: str


def classify_etf(name: str) -> str:
    """Classify an ETF into a coarse portfolio risk bucket by its name."""

    name = str(name or "").upper()
    if any(k in name for k in ("国债", "债券", "政金债", "信用债", "可转债", "公司债")):
        return "bond"
    if any(k in name for k in ("黄金", "白银", "原油", "豆粕", "商品", "有色", "能源化工")):
        return "commodity"
    if any(
        k in name
        for k in (
            "纳指",
            "纳斯达克",
            "标普",
            "恒生",
            "港股",
            "H股",
            "中概",
            "日经",
            "德国",
            "法国",
            "印度",
            "沙特",
            "东南亚",
        )
    ):
        return "overseas_equity"
    if any(
        k in name
        for k in (
            "沪深300",
            "上证50",
            "上证180",
            "中证500",
            "中证800",
            "中证1000",
            "中证2000",
            "A50",
            "A500",
            "深证100",
            "创业板",
            "科创50",
            "科创100",
            "红利",
        )
    ):
        return "domestic_broad"
    if "ETF" in name:
        return "domestic_sector"
    return "other"


def _normalise_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")

    df = history.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    else:
        df = df.sort_index()

    for col in ("close", "high", "low", "money", "amount", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)


def _return_over(closes: pd.Series, days: int) -> float:
    if len(closes) <= days or closes.iloc[-days - 1] <= 0:
        return np.nan
    return float(closes.iloc[-1] / closes.iloc[-days - 1] - 1.0)


def _max_drawdown(closes: pd.Series, days: int) -> float:
    window = closes.tail(days)
    if window.empty:
        return np.nan
    running_max = window.cummax()
    drawdown = window / running_max - 1.0
    return float(abs(drawdown.min()))


def _atr_pct(df: pd.DataFrame, days: int = 20) -> float:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    close = float(df["close"].iloc[-1])
    if close <= 0:
        return np.nan
    return float(true_range.tail(days).mean() / close)


def _amount_series(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Return daily turnover amount and whether it had to be estimated."""

    if "money" in df.columns and df["money"].notna().any():
        return df["money"], False
    if "amount" in df.columns and df["amount"].notna().any():
        return df["amount"], False
    if "volume" in df.columns and df["volume"].notna().any():
        return df["close"] * df["volume"], True
    return pd.Series(np.nan, index=df.index, dtype=float), True


def compute_etf_features(
    history: pd.DataFrame,
    metadata: Mapping[str, object] | None = None,
    config: RotationConfig | None = None,
) -> dict:
    """Compute one ETF's eligibility, trend, momentum, and risk features."""

    config = config or RotationConfig()
    metadata = metadata or {}
    df = _normalise_history(history)
    name = str(metadata.get("name", ""))
    category = str(metadata.get("category") or classify_etf(name))
    reasons: list[str] = []

    if len(df) < config.min_history_days:
        reasons.append(f"历史数据不足{config.min_history_days}日")

    closes = df["close"]
    last_close = float(closes.iloc[-1]) if len(closes) else np.nan
    ma5 = float(closes.tail(5).mean()) if len(closes) >= 5 else np.nan
    ma10 = float(closes.tail(10).mean()) if len(closes) >= 10 else np.nan
    ma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else np.nan
    ma60 = float(closes.tail(60).mean()) if len(closes) >= 60 else np.nan
    ret20 = _return_over(closes, 20)
    ret60 = _return_over(closes, 60)
    ret120 = _return_over(closes, 120)
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan)
    volatility20 = float(returns.tail(20).std(ddof=0) * np.sqrt(252))
    max_drawdown60 = _max_drawdown(closes, 60)
    atr20_pct = _atr_pct(df, 20)

    amount, liquidity_estimated = _amount_series(df)
    recent_amount = amount.tail(config.liquidity_lookback)
    avg_amount20 = float(recent_amount.mean()) if recent_amount.notna().any() else np.nan
    min_amount20 = float(recent_amount.min()) if recent_amount.notna().any() else np.nan

    listed_days = int(metadata.get("listed_days", 999999) or 0)
    if listed_days < config.min_listed_days:
        reasons.append(f"上市不足{config.min_listed_days}日")
    if bool(metadata.get("paused", False)):
        reasons.append("当前停牌")
    if any(keyword.upper() in name.upper() for keyword in config.excluded_name_keywords):
        reasons.append("名称命中排除规则")
    if not np.isfinite(avg_amount20) or avg_amount20 < config.min_avg_amount:
        reasons.append("20日平均成交额不足")
    if not np.isfinite(min_amount20) or min_amount20 < config.min_daily_amount:
        reasons.append("20日最低成交额不足")
    if not all(np.isfinite(v) for v in (last_close, ma5, ma10, ma20, ma60, ret20)):
        reasons.append("趋势指标数据不足")
    else:
        if not last_close > ma20 > ma60:
            reasons.append("未满足收盘价>MA20>MA60")
        if not ma5 > ma10:
            reasons.append("未满足MA5>MA10")
        if ret20 <= 0:
            reasons.append("20日绝对动量不为正")

    return {
        "name": name,
        "category": category,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "history_days": len(df),
        "listed_days": listed_days,
        "close": last_close,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "return20": ret20,
        "return60": ret60,
        "return120": ret120,
        "volatility20": volatility20,
        "max_drawdown60": max_drawdown60,
        "atr20_pct": atr20_pct,
        "avg_amount20": avg_amount20,
        "min_amount20": min_amount20,
        "liquidity_estimated": liquidity_estimated,
    }


def rank_etfs(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, object]] | None = None,
    config: RotationConfig | None = None,
) -> pd.DataFrame:
    """Filter and cross-sectionally rank an ETF universe.

    Scores are percentile based, which keeps unlike asset classes comparable
    without rewarding high nominal prices or raw volatility.
    """

    config = config or RotationConfig()
    metadata = metadata or {}
    records = []
    for code, history in histories.items():
        feature = compute_etf_features(history, metadata.get(code, {}), config)
        feature["code"] = code
        records.append(feature)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("code", drop=False)
    result["score"] = np.nan
    eligible_mask = result["eligible"].astype(bool)
    eligible = result.loc[eligible_mask]
    if not eligible.empty:
        ranks = pd.DataFrame(index=eligible.index)
        ranks["momentum20"] = eligible["return20"].rank(pct=True)
        ranks["momentum60"] = eligible["return60"].rank(pct=True)
        ranks["momentum120"] = eligible["return120"].rank(pct=True)
        ranks["low_volatility"] = eligible["volatility20"].rank(
            ascending=False, pct=True
        )
        ranks["low_drawdown"] = eligible["max_drawdown60"].rank(
            ascending=False, pct=True
        )
        result.loc[eligible.index, "score"] = 100.0 * (
            0.35 * ranks["momentum20"]
            + 0.30 * ranks["momentum60"]
            + 0.15 * ranks["momentum120"]
            + 0.10 * ranks["low_volatility"]
            + 0.10 * ranks["low_drawdown"]
        )

    result["rejection_reason"] = result["rejection_reasons"].apply("；".join)
    return result.sort_values(
        ["eligible", "score", "avg_amount20"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _return_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    lookback: int,
) -> float:
    left_df = _normalise_history(left)
    right_df = _normalise_history(right)
    if "date" in left_df.columns and "date" in right_df.columns:
        left_ret = left_df.set_index("date")["close"].pct_change().rename("left")
        right_ret = right_df.set_index("date")["close"].pct_change().rename("right")
        aligned = pd.concat([left_ret, right_ret], axis=1).dropna().tail(lookback)
    else:
        count = min(len(left_df), len(right_df), lookback + 1)
        aligned = pd.DataFrame(
            {
                "left": left_df["close"].tail(count).pct_change().to_numpy(),
                "right": right_df["close"].tail(count).pct_change().to_numpy(),
            }
        ).dropna()
    if len(aligned) < max(20, lookback // 2):
        return np.nan
    return float(aligned["left"].corr(aligned["right"]))


def select_targets(
    rankings: pd.DataFrame,
    histories: Mapping[str, pd.DataFrame],
    config: RotationConfig | None = None,
) -> pd.DataFrame:
    """Select diversified target ETFs and assign capped inverse-vol weights."""

    config = config or RotationConfig()
    if rankings.empty:
        return rankings.copy()

    selected_rows: list[pd.Series] = []
    category_counts: dict[str, int] = {}
    for _, row in rankings[rankings["eligible"]].iterrows():
        code = str(row["code"])
        category = str(row["category"])
        limit = int(config.category_position_limits.get(category, 1))
        if category_counts.get(category, 0) >= limit:
            continue

        too_correlated = False
        for selected in selected_rows:
            corr = _return_correlation(
                histories[code],
                histories[str(selected["code"])],
                config.correlation_lookback,
            )
            if np.isfinite(corr) and corr >= config.correlation_threshold:
                too_correlated = True
                break
        if too_correlated:
            continue

        selected_rows.append(row)
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(selected_rows) >= config.max_positions:
            break

    if not selected_rows:
        return pd.DataFrame(columns=list(rankings.columns) + ["target_weight"])

    targets = pd.DataFrame(selected_rows).copy().reset_index(drop=True)
    safe_vol = targets["volatility20"].clip(lower=0.01)
    inverse_vol = 1.0 / safe_vol
    investable = 1.0 - config.cash_reserve
    targets["target_weight"] = (
        investable * inverse_vol / inverse_vol.sum()
    ).clip(upper=config.max_position_weight)

    for category, cap in config.category_weight_caps.items():
        mask = targets["category"] == category
        current = float(targets.loc[mask, "target_weight"].sum())
        if current > cap and current > 0:
            targets.loc[mask, "target_weight"] *= cap / current

    targets["target_weight"] = targets["target_weight"].round(6)
    return targets


def evaluate_exit(
    position: PositionState,
    feature: Mapping[str, object],
    current_price: float,
    current_rank: int | None,
    eligible_count: int,
    config: RotationConfig | None = None,
) -> ExitDecision:
    """Evaluate exit rules in strict priority order for one position."""

    config = config or RotationConfig()
    if not position.closeable:
        return ExitDecision(False, position.code, "not_closeable", "当日份额不可卖出")
    if position.entry_price <= 0 or current_price <= 0:
        return ExitDecision(False, position.code, "invalid_price", "价格数据无效")

    atr_pct = float(feature.get("atr20_pct", np.nan))
    if not np.isfinite(atr_pct):
        atr_pct = 0.02
    hard_stop = float(np.clip(2.5 * atr_pct, 0.03, 0.08))
    pnl = current_price / position.entry_price - 1.0
    if pnl <= -hard_stop:
        return ExitDecision(
            True,
            position.code,
            "hard_stop",
            f"亏损{pnl:.1%}触发动态止损{-hard_stop:.1%}",
        )

    highest = max(position.highest_price, current_price, position.entry_price)
    peak_return = highest / position.entry_price - 1.0
    giveback = current_price / highest - 1.0
    trailing_stop = float(np.clip(2.0 * atr_pct, 0.025, 0.10))
    if (
        peak_return >= config.trailing_activate_return
        and giveback <= -trailing_stop
    ):
        return ExitDecision(
            True,
            position.code,
            "trailing_stop",
            f"最高盈利{peak_return:.1%}后回撤{giveback:.1%}",
        )

    ma5 = float(feature.get("ma5", np.nan))
    ma10 = float(feature.get("ma10", np.nan))
    if position.trend_weak_days >= config.trend_exit_confirm_days or (
        np.isfinite(ma5) and np.isfinite(ma10) and ma5 < ma10
    ):
        return ExitDecision(True, position.code, "trend_exit", "趋势转弱，执行退出")

    if (
        position.holding_days >= config.min_hold_days
        and position.rank_weak_days >= config.rank_exit_confirm_days
    ):
        return ExitDecision(
            True,
            position.code,
            "rank_exit",
            f"连续跌出综合排名前{config.rank_exit}",
        )

    median_rank = max(1, int(np.ceil(eligible_count / 2)))
    if (
        position.holding_days >= config.time_stop_days
        and peak_return < config.time_stop_peak_return
        and (current_rank is None or current_rank > median_rank)
    ):
        return ExitDecision(
            True,
            position.code,
            "time_stop",
            f"持有{position.holding_days}日仍未形成有效趋势",
        )

    return ExitDecision(False, position.code, "hold", "未触发退出条件")

