"""Short-term Band Trading strategy — 5-day cycle, quick in-and-out.

Designed for small accounts (~2000 yuan) doing band trading.
Entry: yesterday's close was down + price near MA10.
Exit: +5% profit / 5-day hold / -3% stop loss.
Includes auto ETF selection from a curated high-liquidity pool.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.signals import DashboardCard, LiveSignal


# ---------------------------------------------------------------------------
# Candidate ETF pool — high liquidity, high volatility, affordable
# ---------------------------------------------------------------------------

CANDIDATE_ETFS: list[dict] = [
    {"code": "510300", "name": "沪深300ETF", "approx_price": 3.9},
    {"code": "510050", "name": "上证50ETF", "approx_price": 2.7},
    {"code": "510500", "name": "中证500ETF", "approx_price": 5.8},
    {"code": "159915", "name": "创业板ETF", "approx_price": 2.2},
    {"code": "588000", "name": "科创50ETF", "approx_price": 0.9},
    {"code": "512880", "name": "证券ETF", "approx_price": 0.9},
    {"code": "512100", "name": "中证1000ETF", "approx_price": 2.3},
    {"code": "159949", "name": "创业板50", "approx_price": 0.9},
    {"code": "512690", "name": "酒ETF", "approx_price": 1.5},
    {"code": "159845", "name": "中证1000", "approx_price": 2.2},
]


class ShortTermBandStrategy(BaseStrategy):
    """Short-term band trading — 5-day cycle, quick in-and-out.

    Entry: yesterday was a down day AND price is near MA10 (within ±2%).
    Exit: +5% take-profit / 5-day time stop / -3% stop loss.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "短线波段"

    @property
    def description(self) -> str:
        return (
            "5日周期快进快出：昨天下跌+回踩均线时买入，"
            "赚5%就跑或最多拿5天，跌3%止损。"
            "自动从10只高流动性ETF中选最优标的，"
            "适合2000元级别小资金做波段。"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return {
            "take_profit_pct": 0.05,
            "stop_loss_pct": 0.03,
            "max_hold_days": 5,
            "ma_period": 10,
            "ma_proximity_pct": 0.02,
            "require_down_day": True,
            "position_pct": 1.0,
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "take_profit_pct": {
                "label": "止盈线",
                "type": "slider",
                "min": 0.02, "max": 0.15, "step": 0.01,
                "help": "盈利达到这个比例就卖（默认5%）",
            },
            "stop_loss_pct": {
                "label": "止损线",
                "type": "slider",
                "min": 0.01, "max": 0.10, "step": 0.01,
                "help": "亏损达到这个比例立即止损（默认3%）",
            },
            "max_hold_days": {
                "label": "最长持有天数",
                "type": "number",
                "min": 3, "max": 10, "step": 1,
                "help": "无论盈亏，持有超过这个天数就卖出（默认5天）",
            },
            "ma_period": {
                "label": "均线周期",
                "type": "number",
                "min": 5, "max": 30, "step": 1,
                "help": "参考均线的天数，价格接近这条线时买入（默认10日）",
            },
            "ma_proximity_pct": {
                "label": "均线接近度",
                "type": "slider",
                "min": 0.01, "max": 0.05, "step": 0.005,
                "help": "价格离均线多远算「接近」（默认±2%）",
            },
            "require_down_day": {
                "label": "要求昨日下跌",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=昨天收阴线才考虑买入 | 关闭=只看均线位置",
            },
            "position_pct": {
                "label": "仓位比例",
                "type": "slider",
                "min": 0.5, "max": 1.0, "step": 0.1,
                "help": "每次买入占总资金的比例（2000元建议100%=满仓）",
            },
        }

    # ------------------------------------------------------------------
    # Backtest signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        params = {**self.get_default_params(), **kwargs}
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        ma_period = int(params["ma_period"])
        ma_prox = float(params["ma_proximity_pct"])
        require_down = str(params.get("require_down_day", "True")).lower() in ("true", "1", "yes")
        position_pct = float(params["position_pct"])

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # --- Compute MA ---
        if "close" not in df.columns:
            df["signal"] = "hold"
            df["signal_price"] = 0.0
            df["signal_shares"] = 0
            df["signal_reason"] = ""
            return df

        df["_ma"] = df["close"].rolling(window=ma_period, min_periods=ma_period).mean()
        df["_prev_close"] = df["close"].shift(1)
        df["_prev_open"] = df["open"].shift(1)

        # --- State tracking across bars ---
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        in_position = False
        entry_price = 0.0
        entry_idx = -1

        for i in range(len(df)):
            close = float(df.at[i, "close"])
            ma_val = df.at[i, "_ma"]
            prev_close = df.at[i, "_prev_close"]
            prev_open = df.at[i, "_prev_open"]

            if in_position:
                # --- Check exit conditions ---
                pnl_pct = (close - entry_price) / entry_price if entry_price > 0 else 0
                hold_days = i - entry_idx

                if pnl_pct >= take_profit:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = 0  # sell all
                    df.at[i, "signal_reason"] = (
                        f"止盈卖出：盈利 {pnl_pct:.1%}（≥{take_profit:.0%}），"
                        f"入场价 {entry_price:.3f} → 出场价 {close:.3f}"
                    )
                    in_position = False

                elif pnl_pct <= -stop_loss:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = 0
                    df.at[i, "signal_reason"] = (
                        f"止损卖出：亏损 {pnl_pct:.1%}（≤-{stop_loss:.0%}），"
                        f"入场价 {entry_price:.3f} → 出场价 {close:.3f}"
                    )
                    in_position = False

                elif hold_days >= max_hold:
                    df.at[i, "signal"] = "sell"
                    df.at[i, "signal_price"] = close
                    df.at[i, "signal_shares"] = 0
                    df.at[i, "signal_reason"] = (
                        f"时间止盈：持有 {hold_days} 天（≥{max_hold}天），"
                        f"盈亏 {pnl_pct:+.1%}，入场价 {entry_price:.3f} → 出场价 {close:.3f}"
                    )
                    in_position = False

            else:
                # --- Check entry conditions ---
                if pd.isna(ma_val) or pd.isna(prev_close) or pd.isna(prev_open):
                    continue

                # Condition 1: yesterday was a down day
                is_down_day = prev_close < prev_open
                if require_down and not is_down_day:
                    continue

                # Condition 2: price near MA10
                dist_from_ma = abs(close - ma_val) / ma_val if ma_val > 0 else 999
                if dist_from_ma > ma_prox:
                    continue

                # Both conditions met → BUY
                df.at[i, "signal"] = "buy"
                df.at[i, "signal_price"] = close
                budget = position_pct * 100_000
                df.at[i, "signal_shares"] = max(1, int(budget / close))
                df.at[i, "signal_reason"] = (
                    f"买入信号：昨跌{prev_close:.3f}（开{prev_open:.3f}收{prev_close:.3f}），"
                    f"现价{close:.3f}距MA{ma_period}({ma_val:.3f})仅{dist_from_ma:.1%}"
                )
                in_position = True
                entry_price = close
                entry_idx = i

        # Clean up
        return df.drop(columns=["_ma", "_prev_close", "_prev_open"], errors="ignore")

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    def get_live_signal(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> LiveSignal:
        """Today's recommendation for the selected ETF.

        Checks if we're in a position (via portfolio context) and whether
        today meets entry criteria (yesterday down + near MA).
        """
        if df is None or df.empty:
            return LiveSignal(
                action="hold",
                reason="无历史数据",
                urgency_level="low",
            )

        params = {**self.get_default_params(), **kwargs}
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])
        max_hold = int(params["max_hold_days"])
        ma_period = int(params["ma_period"])
        ma_prox = float(params["ma_proximity_pct"])
        require_down = str(params.get("require_down_day", "True")).lower() in ("true", "1", "yes")

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

        current_price = info.get("current_price") if info else None
        if current_price is None:
            current_price = float(df_sorted.iloc[-1]["close"])

        # --- Compute MA ---
        closes = df_sorted["close"].values
        if len(closes) < ma_period + 1:
            return LiveSignal(
                action="hold",
                current_price=round(current_price, 4),
                reason=f"数据不足（需要至少{ma_period+1}个交易日）",
                urgency_level="low",
            )

        ma_val = float(np.mean(closes[-(ma_period + 1):-1]))  # MA of last N bars (excl today)
        prev_close = float(df_sorted.iloc[-2]["close"])
        prev_open = float(df_sorted.iloc[-2]["open"])
        is_down_day = prev_close < prev_open
        dist_from_ma = abs(current_price - ma_val) / ma_val if ma_val > 0 else 0

        # --- Check portfolio context for existing position ---
        portfolio_context = kwargs.get("portfolio_context")
        has_position = False
        entry_price_from_pf = None
        hold_days = 0
        if portfolio_context and portfolio_context.get("has_position"):
            has_position = True
            entry_price_from_pf = portfolio_context.get("holding_avg_cost")
            # Estimate hold days from last buy date
            last_buy_date = portfolio_context.get("last_buy_date")
            if last_buy_date:
                try:
                    if isinstance(last_buy_date, str):
                        last_buy_date = datetime.strptime(last_buy_date, "%Y-%m-%d").date()
                    hold_days = (datetime.now().date() - last_buy_date).days
                except (ValueError, TypeError):
                    pass

        # --- In position: check exit ---
        if has_position and entry_price_from_pf and entry_price_from_pf > 0:
            pnl_pct = (current_price - entry_price_from_pf) / entry_price_from_pf

            if pnl_pct >= take_profit:
                return LiveSignal(
                    action="sell",
                    current_price=round(current_price, 4),
                    suggested_shares=0,
                    suggested_amount=0,
                    trigger_description=f"盈利 {pnl_pct:.1%}，触发止盈线 {take_profit:.0%}",
                    next_trigger_price=None,
                    reason=(
                        f"🎯 止盈！入场价 ¥{entry_price_from_pf:.3f} → "
                        f"现价 ¥{current_price:.3f}（+{pnl_pct:.1%}），"
                        f"建议全部卖出。"
                    ),
                    urgency_level="high",
                    current_zone="止盈区",
                )

            if pnl_pct <= -stop_loss:
                return LiveSignal(
                    action="sell",
                    current_price=round(current_price, 4),
                    suggested_shares=0,
                    suggested_amount=0,
                    trigger_description=f"亏损 {pnl_pct:.1%}，触发止损线 -{stop_loss:.0%}",
                    next_trigger_price=None,
                    reason=(
                        f"🛑 止损！入场价 ¥{entry_price_from_pf:.3f} → "
                        f"现价 ¥{current_price:.3f}（{pnl_pct:.1%}），"
                        f"建议全部卖出。"
                    ),
                    urgency_level="high",
                    current_zone="止损区",
                )

            if hold_days >= max_hold:
                return LiveSignal(
                    action="sell",
                    current_price=round(current_price, 4),
                    suggested_shares=0,
                    suggested_amount=0,
                    trigger_description=f"持有 {hold_days} 天，触发时间止盈 {max_hold}天",
                    next_trigger_price=None,
                    reason=(
                        f"⏰ 时间到！持有 {hold_days} 天（≥{max_hold}天），"
                        f"盈亏 {pnl_pct:+.1%}，建议卖出换标的。"
                    ),
                    urgency_level="medium",
                    current_zone="时间止盈区",
                )

            # Still in position, waiting
            return LiveSignal(
                action="hold",
                current_price=round(current_price, 4),
                trigger_description=f"持仓中，盈亏 {pnl_pct:+.1%}（第{hold_days}天）",
                next_trigger_price=round(entry_price_from_pf * (1 + take_profit), 4),
                reason=(
                    f"📌 持仓中：入场 ¥{entry_price_from_pf:.3f}，"
                    f"现价 ¥{current_price:.3f}（{pnl_pct:+.1%}），"
                    f"止盈线 ¥{entry_price_from_pf * (1 + take_profit):.3f}，"
                    f"止损线 ¥{entry_price_from_pf * (1 - stop_loss):.3f}"
                ),
                urgency_level="low",
                portions_used=hold_days,
                portions_total=max_hold,
                current_zone=f"持仓中 第{hold_days}天",
            )

        # --- Not in position: check entry ---
        if require_down and not is_down_day:
            return LiveSignal(
                action="wait_for_drop",
                current_price=round(current_price, 4),
                trigger_description=f"昨日收阳（{prev_close:.3f}），等待回调",
                next_trigger_price=round(ma_val, 4),
                reason=(
                    f"⏳ 等待回调：昨日收阳（开{prev_open:.3f}收{prev_close:.3f}），"
                    f"等一个阴线回踩MA{ma_period}（¥{ma_val:.3f}）再入场。"
                ),
                urgency_level="low",
                current_zone="等待回调",
            )

        if dist_from_ma > ma_prox:
            return LiveSignal(
                action="wait_for_drop",
                current_price=round(current_price, 4),
                trigger_description=f"距MA{ma_period}太远（{dist_from_ma:.1%} > {ma_prox:.0%}）",
                next_trigger_price=round(ma_val, 4),
                reason=(
                    f"⏳ 现价 ¥{current_price:.3f} 离 MA{ma_period}（¥{ma_val:.3f}）"
                    f"还有 {dist_from_ma:.1%}，等回踩到 ¥{ma_val * (1 + ma_prox):.3f} 以下再买。"
                ),
                urgency_level="low",
                current_zone="等待回踩",
            )

        # --- Entry signal! ---
        # Calculate shares for 2000 yuan (or available cash from portfolio)
        available_cash = 2000.0
        if portfolio_context and portfolio_context.get("available_cash", 0) > 0:
            available_cash = float(portfolio_context["available_cash"])

        budget = available_cash * float(params["position_pct"])
        raw_shares = int(budget / current_price) if current_price > 0 else 0
        shares = (raw_shares // 100) * 100  # lot-aligned

        return LiveSignal(
            action="buy",
            current_price=round(current_price, 4),
            suggested_shares=shares,
            suggested_amount=round(shares * current_price, 2),
            trigger_description=(
                f"昨跌+回踩MA{ma_period}！"
                f"昨收{prev_close:.3f}，现价距均线仅{dist_from_ma:.1%}"
            ),
            next_trigger_price=round(ma_val, 4),
            reason=(
                f"🎯 买入信号！昨跌（开{prev_open:.3f}收{prev_close:.3f}），"
                f"现价 ¥{current_price:.3f} 在 MA{ma_period}（¥{ma_val:.3f}）附近"
                f"（偏离{dist_from_ma:.1%}）。"
                f"建议买入 {shares} 股（约¥{shares * current_price:.0f}），"
                f"止盈 +{take_profit:.0%}（¥{current_price * (1 + take_profit):.3f}），"
                f"止损 -{stop_loss:.0%}（¥{current_price * (1 - stop_loss):.3f}）。"
            ),
            urgency_level="high",
            current_zone="买入区",
        )

    # ------------------------------------------------------------------
    # Dashboard cards
    # ------------------------------------------------------------------

    def get_dashboard_cards(
        self, df: pd.DataFrame, info: dict, **kwargs
    ) -> list[DashboardCard]:
        """Show entry/exit rules and current market status."""
        if df is None or df.empty:
            return []

        params = {**self.get_default_params(), **kwargs}
        ma_period = int(params["ma_period"])
        take_profit = float(params["take_profit_pct"])
        stop_loss = float(params["stop_loss_pct"])

        df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
        closes = df_sorted["close"].values

        current_price = info.get("current_price") if info else None
        if current_price is None:
            current_price = float(df_sorted.iloc[-1]["close"])

        # MA value
        if len(closes) >= ma_period + 1:
            ma_val = float(np.mean(closes[-(ma_period + 1):-1]))
            dist_pct = (current_price - ma_val) / ma_val if ma_val > 0 else 0
        else:
            ma_val = None
            dist_pct = None

        # Recent stats
        last_5 = closes[-5:] if len(closes) >= 5 else closes
        volatility = float(np.std(last_5) / np.mean(last_5)) if len(last_5) > 1 else 0

        cards: list[DashboardCard] = []

        # Card 1: Entry rules status
        prev_close = float(df_sorted.iloc[-2]["close"]) if len(df_sorted) >= 2 else None
        prev_open = float(df_sorted.iloc[-2]["open"]) if len(df_sorted) >= 2 else None
        is_down = prev_close < prev_open if prev_close and prev_open else False
        near_ma = abs(dist_pct) <= float(params["ma_proximity_pct"]) if dist_pct is not None else False

        cards.append(DashboardCard(
            card_id="entry_status",
            title="今日入场条件",
            card_type="trigger",
            content={
                "rules": [
                    {"label": "昨日收阴", "met": is_down, "detail": f"昨收{prev_close}" if prev_close else "N/A"},
                    {"label": f"接近MA{ma_period}", "met": near_ma,
                     "detail": f"偏离{dist_pct:+.1%}" if dist_pct is not None else "N/A"},
                ],
                "ready": is_down and near_ma,
            },
            priority=1,
        ))

        # Card 2: Exit rules
        cards.append(DashboardCard(
            card_id="exit_rules",
            title="出场规则",
            card_type="info",
            content={
                "rules": [
                    {"label": "止盈", "value": f"+{take_profit:.0%}"},
                    {"label": "止损", "value": f"-{stop_loss:.0%}"},
                    {"label": "时间止盈", "value": f"持有{params['max_hold_days']}天"},
                ],
            },
            priority=1,
        ))

        # Card 3: Market context
        cards.append(DashboardCard(
            card_id="market_context",
            title="市场环境",
            card_type="info",
            content={
                "current_price": round(current_price, 4),
                "ma_value": round(ma_val, 4) if ma_val else None,
                "dist_from_ma": round(dist_pct, 4) if dist_pct is not None else None,
                "volatility_5d": f"{volatility:.2%}" if volatility else "N/A",
            },
            priority=2,
        ))

        return cards

    # ------------------------------------------------------------------
    # ETF selection (static — updated weekly or on close)
    # ------------------------------------------------------------------

    @staticmethod
    def get_candidate_pool() -> list[dict]:
        """Return the curated candidate ETF list."""
        return list(CANDIDATE_ETFS)

    @staticmethod
    def select_best_etf(
        candidates: list[dict] | None = None,
        max_price: float = 20.0,
    ) -> dict | None:
        """Select the best ETF for short-term band trading.

        Ranks candidates by recent volatility (amplitude) × volume,
        filtered by affordability (100 shares ≤ available cash).

        This is designed to be called from a headless script (weekly cron)
        or from the Streamlit UI when the user clicks "选最优标的".

        Args:
            candidates: Optional override list. Uses CANDIDATE_ETFS if None.
            max_price: Max share price that fits in a 2000-yuan account
                       (100 shares × price ≤ 2000 → price ≤ 20).

        Returns:
            The best candidate dict, or None if no suitable ETF found.
        """
        from src.data.fetcher import fetch_etf_info

        pool = candidates if candidates is not None else list(CANDIDATE_ETFS)
        scored: list[dict] = []

        for etf in pool:
            code = etf["code"]
            try:
                info = fetch_etf_info(code)
            except Exception:
                continue

            current_price = info.get("current_price")
            if current_price is None or current_price <= 0:
                continue

            # Affordability: 100 shares must fit in ~2000 yuan
            if current_price > max_price:
                continue

            # Score: prefer high volume + high volatility
            volume = info.get("volume", 0) or 0
            amplitude = info.get("amplitude", 0) or 0  # 振幅
            turnover = info.get("turnover_rate", 0) or 0  # 换手率

            # Normalize and score (higher = better for band trading)
            score = (amplitude * 40 + turnover * 30 + min(volume / 100_000_000, 1) * 30)

            scored.append({
                **etf,
                "current_price": current_price,
                "amplitude": amplitude,
                "volume": volume,
                "turnover_rate": turnover,
                "score": round(score, 2),
                "name_from_api": info.get("name", etf["name"]),
            })

        if not scored:
            return None

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[0]
