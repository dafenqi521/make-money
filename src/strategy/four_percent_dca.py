"""雷牛牛 4%定投法（强化版）— "1+1+4" 框架。

核心逻辑（雷牛牛原创）:
  1. 选优质指数基金 — 宽基优先，业绩稳定，穿越牛熊
  1. 合理估值内买入 — PE分位 < 30% 才出手，低估区积极
  4. 每跌 4% 定投一份 — 共 10 份，越跌越买

卖出规则:
  - PE分位 > 70%（高估区）→ 分批卖出
  - 或纯价格模式：每涨 4% 卖出一份

策略特点:
  - 天然规避追涨杀跌：上涨不触发买点，下跌正是收集筹码时
  - 涨也开心（持仓盈利），跌也开心（便宜筹码）
  - 中间呆坐不动，节约交易费用

✅ PE 历史分位已接入: 使用 legulegu.com 指数 PE 历史数据（2005-今），
  支持 PE 分位判断（"PE处于历史15%分位"），替代静态 PE 阈值。
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.signals import LiveSignal, DashboardCard

if TYPE_CHECKING:
    from src.data.pe_history import PEPercentile
    from src.data.macro_pulse import MacroPulse


class FourPercentDCAStrategy(BaseStrategy):
    """雷牛牛 4%定投法（强化版）。

    在合理估值前提下，以上次买入价为基准，每下跌 4% 定投一份，
    最多 10 份。高估时分批卖出。支持 PE 分位过滤和纯价格两种模式。

    PE 过滤支持两种模式（自动降级）:
      - **PE 分位模式**（推荐）: 基于历史 PE 分位判断买卖区间
      - **PE 阈值模式**（兜底）: 用 pe_buy_threshold / pe_sell_threshold 静态判断
    """

    @property
    def name(self) -> str:
        return "4%定投法"

    @property
    def description(self) -> str:
        return (
            "雷牛牛「1+1+4」框架：选优质指数(1) + 合理估值区间(1) + 每跌4%定投1份(4)。\n\n"
            "**买入**: PE分位<30%（低估区），以上次买入价为基准，每下跌4%买入一份，共10份。\n"
            "**卖出**: PE分位>70%（高估区），以上次卖出价为基准，每上涨4%卖出一份。\n"
            "**持有**: PE分位30%~70%（合理区）呆坐不动，不买不卖。\n\n"
            "核心优势：上涨不触发买点（不追涨），下跌正是收集筹码时（不恐惧），"
            "天然规避追涨杀跌。做到涨也开心、跌也开心。\n\n"
            "✅ 已接入 PE 历史分位（legulegu 指数 PE 数据 2005-今）\n"
            "参考年化: 8%-12%。"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return {
            "total_portions": 10,
            "drop_threshold_pct": 0.04,
            "rise_threshold_pct": 0.04,
            "portion_amount": 4000,
            "pe_buy_threshold": 15.0,
            "pe_sell_threshold": 30.0,
            "pe_percentile_buy": 0.30,   # PE分位 < 30% 才买入
            "pe_percentile_sell": 0.70,  # PE分位 > 70% 才卖出
            "use_pe_filter": True,
            "macro_risk_filter": False,   # 极端恐惧时自动暂停买入
        }

    def get_param_descriptions(self) -> dict[str, dict]:
        return {
            "total_portions": {
                "label": "总份数",
                "type": "number",
                "min": 3, "max": 20, "step": 1,
                "help": "将单标的总投资额分成多少份（雷牛牛建议 10 份）",
            },
            "drop_threshold_pct": {
                "label": "下跌触发阈值",
                "type": "slider",
                "min": 0.02, "max": 0.10, "step": 0.005,
                "help": "相对上次买入价下跌多少触发下一次买入（默认4%）",
            },
            "rise_threshold_pct": {
                "label": "上涨触发阈值",
                "type": "slider",
                "min": 0.02, "max": 0.10, "step": 0.005,
                "help": "相对上次卖出价上涨多少触发下一次卖出（默认4%）",
            },
            "portion_amount": {
                "label": "每份金额(元)",
                "type": "number",
                "min": 500, "max": 50000, "step": 500,
                "help": "每份定投金额。建议 = 总仓位上限 ÷ 总份数（如 4万÷10=4000）",
            },
            "pe_buy_threshold": {
                "label": "PE买入线（兜底）",
                "type": "number",
                "min": 5.0, "max": 40.0, "step": 1.0,
                "help": "PE分位不可用时的兜底：PE低于此值允许买入。设0禁用",
            },
            "pe_sell_threshold": {
                "label": "PE卖出线（兜底）",
                "type": "number",
                "min": 10.0, "max": 60.0, "step": 1.0,
                "help": "PE分位不可用时的兜底：PE高于此值允许卖出。设999禁用",
            },
            "pe_percentile_buy": {
                "label": "PE分位买入线",
                "type": "slider",
                "min": 0.05, "max": 0.50, "step": 0.05,
                "help": "PE历史分位低于此值允许买入（默认30%=低估区）。优先于PE绝对值阈值",
            },
            "pe_percentile_sell": {
                "label": "PE分位卖出线",
                "type": "slider",
                "min": 0.50, "max": 0.95, "step": 0.05,
                "help": "PE历史分位高于此值允许卖出（默认70%=高估区）。优先于PE绝对值阈值",
            },
            "use_pe_filter": {
                "label": "启用PE过滤",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启=PE分位/PE阈值过滤（低估买/高估卖） | 关闭=纯价格4%机械式买卖",
            },
            "macro_risk_filter": {
                "label": "宏观风险过滤",
                "type": "select",
                "options": ["True", "False"],
                "help": "开启后，极端恐惧时自动暂停买入信号（需宏观温度计数据）",
            },
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        pe_value: float | None = None,
        pe_percentile: "PEPercentile | None" = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate buy/sell signals following the 4% DCA rules.

        Args:
            df: OHLCV DataFrame sorted date ascending.
            pe_value: Current PE(TTM) snapshot (fallback, used when pe_percentile is None).
            pe_percentile: PEPercentile from PE history (preferred). When available,
                uses historical PE percentile for zone determination instead of
                static PE thresholds.
            **kwargs: Strategy parameter overrides.

        Returns:
            DataFrame with appended signal columns.
        """
        params = {**self.get_default_params(), **kwargs}
        total_portions = int(params["total_portions"])
        drop_pct = float(params["drop_threshold_pct"])
        rise_pct = float(params["rise_threshold_pct"])
        portion_amount = float(params["portion_amount"])
        pe_buy_threshold = float(params["pe_buy_threshold"])
        pe_sell_threshold = float(params["pe_sell_threshold"])
        pe_pct_buy = float(params["pe_percentile_buy"])
        pe_pct_sell = float(params["pe_percentile_sell"])
        use_pe = str(params["use_pe_filter"]).lower() in ("true", "1", "yes")

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # --- Init signal columns ---
        df["signal"] = "hold"
        df["signal_price"] = df["close"]
        df["signal_shares"] = 0
        df["signal_reason"] = ""

        # --- Determine zones ---
        zone_source = "none"  # "percentile" | "threshold" | "pure_price" | "none"

        if use_pe:
            # Priority 1: PE historical percentile
            if pe_percentile is not None and pe_percentile.pe_percentile is not None:
                pct = pe_percentile.pe_percentile
                can_buy = pct < pe_pct_buy * 100
                can_sell = pct >= pe_pct_sell * 100
                in_hold_zone = not can_buy and not can_sell
                zone_source = "percentile"
            # Priority 2: PE snapshot threshold (fallback)
            elif pe_value is not None and pe_value > 0:
                can_buy = pe_value < pe_buy_threshold
                can_sell = pe_value >= pe_sell_threshold
                in_hold_zone = (
                    pe_buy_threshold <= pe_value < pe_sell_threshold
                )
                zone_source = "threshold"
            else:
                # No PE data at all — can't determine zone, use pure price
                can_buy = True
                can_sell = True
                in_hold_zone = False
                zone_source = "pure_price"
        else:
            # Pure price mode — both buy and sell allowed
            can_buy = True
            can_sell = True
            in_hold_zone = False
            zone_source = "pure_price"

        if in_hold_zone:
            # Attach metadata before returning
            df.attrs["portions_bought"] = 0
            df.attrs["portions_sold"] = 0
            df.attrs["total_portions"] = total_portions
            df.attrs["can_buy"] = can_buy
            df.attrs["can_sell"] = can_sell
            df.attrs["zone_source"] = zone_source
            return df

        # --- Forward pass: track trigger levels ---
        portions_bought = 0
        portions_sold = 0
        next_buy_trigger: float | None = None
        next_sell_trigger: float | None = None

        for idx in df.index:
            price = float(df.at[idx, "close"])
            if pd.isna(price) or price <= 0:
                continue

            # ============================================================
            # BUY logic
            # ============================================================
            if can_buy and portions_bought < total_portions:
                if next_buy_trigger is None:
                    # First buy: enter at current market price
                    shares = max(1, int(portion_amount / price))
                    df.at[idx, "signal"] = "buy"
                    df.at[idx, "signal_price"] = price
                    df.at[idx, "signal_shares"] = shares
                    df.at[idx, "signal_reason"] = (
                        f"🔵 4%定投 第1/{total_portions}份 | "
                        f"入场价 {price:.3f} | "
                        + (
                            f"PE分位={pe_percentile.pe_percentile:.0f}%"
                            if pe_percentile is not None and pe_percentile.pe_percentile is not None
                            else (f"PE={pe_value:.1f} < {pe_buy_threshold}" if use_pe and pe_value else "纯价格模式")
                        )
                    )
                    next_buy_trigger = price * (1.0 - drop_pct)
                    portions_bought += 1

                elif price <= next_buy_trigger:
                    # Price dropped to trigger level → buy one more portion
                    trigger_price = next_buy_trigger
                    next_buy_trigger = trigger_price * (1.0 - drop_pct)
                    shares = max(1, int(portion_amount / price))
                    df.at[idx, "signal"] = "buy"
                    df.at[idx, "signal_price"] = price
                    df.at[idx, "signal_shares"] = shares

                    total_drop_pct = (1.0 - price / (next_buy_trigger / (1.0 - drop_pct))) * 100
                    df.at[idx, "signal_reason"] = (
                        f"🔵 4%定投 第{portions_bought + 1}/{total_portions}份 | "
                        f"触发价 {trigger_price:.3f} | 成交 {price:.3f} | "
                        f"下次触发 {next_buy_trigger:.3f}"
                    )
                    portions_bought += 1

            # ============================================================
            # SELL logic
            # ============================================================
            if can_sell and portions_bought > 0 and portions_sold < portions_bought:
                if next_sell_trigger is None:
                    # First sell trigger: start from a reasonable reference.
                    # Use the most recent buy trigger price (next_buy_trigger / (1-drop))
                    # as the cost anchor, and sell when price rises rise_pct above it.
                    # If no buy trigger yet (shouldn't happen), use current price.
                    ref_price = (
                        next_buy_trigger / (1.0 - drop_pct)
                        if next_buy_trigger is not None
                        else price
                    )
                    next_sell_trigger = ref_price * (1.0 + rise_pct)

                    # Only sell immediately if price already above trigger
                    if price >= next_sell_trigger:
                        shares = max(1, int(portion_amount / price))
                        df.at[idx, "signal"] = "sell"
                        df.at[idx, "signal_price"] = price
                        df.at[idx, "signal_shares"] = shares
                        df.at[idx, "signal_reason"] = (
                            f"🔴 4%定投 卖出第{portions_sold + 1}份 | "
                            f"触发价 {next_sell_trigger:.3f} | 成交 {price:.3f} | "
                            + (
                                f"PE分位={pe_percentile.pe_percentile:.0f}%"
                                if pe_percentile is not None and pe_percentile.pe_percentile is not None
                                else (f"PE={pe_value:.1f} ≥ {pe_sell_threshold}" if use_pe and pe_value else "纯价格模式")
                            )
                        )
                        next_sell_trigger = next_sell_trigger * (1.0 + rise_pct)
                        portions_sold += 1

                elif price >= next_sell_trigger:
                    trigger_price = next_sell_trigger
                    next_sell_trigger = trigger_price * (1.0 + rise_pct)
                    shares = max(1, int(portion_amount / price))
                    df.at[idx, "signal"] = "sell"
                    df.at[idx, "signal_price"] = price
                    df.at[idx, "signal_shares"] = shares
                    df.at[idx, "signal_reason"] = (
                        f"🔴 4%定投 卖出第{portions_sold + 1}份 | "
                        f"触发价 {trigger_price:.3f} | 成交 {price:.3f} | "
                        f"下次触发 {next_sell_trigger:.3f}"
                    )
                    portions_sold += 1

        # --- Attach summary metadata ---
        df.attrs["portions_bought"] = portions_bought
        df.attrs["portions_sold"] = portions_sold
        df.attrs["total_portions"] = total_portions
        df.attrs["can_buy"] = can_buy
        df.attrs["can_sell"] = can_sell
        df.attrs["zone_source"] = zone_source
        if pe_percentile is not None and pe_percentile.pe_percentile is not None:
            df.attrs["pe_percentile"] = pe_percentile.pe_percentile

        return df

    # ------------------------------------------------------------------
    # Helper: forward-pass state extraction
    # ------------------------------------------------------------------

    def _compute_state(
        self,
        df: pd.DataFrame,
        pe_value: float | None = None,
        pe_percentile: "PEPercentile | None" = None,
        portfolio_context: dict | None = None,
        **kwargs,
    ) -> dict:
        """Run forward pass through *df* and return the final tracked state.

        When *portfolio_context* is provided (from actual executed trades),
        the forward-pass anchor prices are overridden by real execution
        prices so that future signal computation adapts to what the user
        actually did rather than what the historical simulation expects.

        Returns a dict with keys:
          portions_bought, portions_sold, next_buy_trigger,
          next_sell_trigger, can_buy, can_sell, in_hold_zone,
          is_pure_price_mode, zone_source, total_portions, drop_pct,
          rise_pct, portion_amount, pe_buy_threshold, pe_sell_threshold,
          pe_percentile_buy, pe_percentile_sell, pe_percentile_val,
          anchored_by_portfolio (bool)
        """
        params = {**self.get_default_params(), **kwargs}
        total_portions = int(params["total_portions"])
        drop_pct = float(params["drop_threshold_pct"])
        rise_pct = float(params["rise_threshold_pct"])
        use_pe = str(params.get("use_pe_filter", "True")).lower() in ("true", "1", "yes")
        pe_buy_threshold = float(params["pe_buy_threshold"])
        pe_sell_threshold = float(params["pe_sell_threshold"])
        pe_pct_buy = float(params["pe_percentile_buy"])
        pe_pct_sell = float(params["pe_percentile_sell"])
        portion_amount = float(params["portion_amount"])

        # --- Edge case: no data ---
        if df is None or len(df) == 0:
            return {
                "portions_bought": 0,
                "portions_sold": 0,
                "next_buy_trigger": None,
                "next_sell_trigger": None,
                "can_buy": False,
                "can_sell": False,
                "in_hold_zone": True,
                "is_pure_price_mode": False,
                "zone_source": "none",
                "total_portions": total_portions,
                "drop_pct": drop_pct,
                "rise_pct": rise_pct,
                "portion_amount": portion_amount,
                "pe_buy_threshold": pe_buy_threshold,
                "pe_sell_threshold": pe_sell_threshold,
                "pe_percentile_buy": pe_pct_buy,
                "pe_percentile_sell": pe_pct_sell,
                "pe_percentile_val": None,
            }

        df = df.sort_values("date", ascending=True).reset_index(drop=True).copy()

        # --- Determine zones ---
        zone_source = "none"

        if use_pe:
            # Priority 1: PE historical percentile
            if pe_percentile is not None and pe_percentile.pe_percentile is not None:
                pct = pe_percentile.pe_percentile
                can_buy = pct < pe_pct_buy * 100
                can_sell = pct >= pe_pct_sell * 100
                in_hold_zone = not can_buy and not can_sell
                is_pure_price_mode = False
                zone_source = "percentile"
            # Priority 2: PE snapshot threshold (fallback)
            elif pe_value is not None and pe_value > 0:
                can_buy = pe_value < pe_buy_threshold
                can_sell = pe_value >= pe_sell_threshold
                in_hold_zone = (
                    pe_buy_threshold <= pe_value < pe_sell_threshold
                )
                is_pure_price_mode = False
                zone_source = "threshold"
            else:
                can_buy = True
                can_sell = True
                in_hold_zone = False
                is_pure_price_mode = True
                zone_source = "pure_price"
        else:
            can_buy = True
            can_sell = True
            in_hold_zone = False
            is_pure_price_mode = True
            zone_source = "pure_price"

        # --- Forward pass ---
        portions_bought = 0
        portions_sold = 0
        next_buy_trigger: float | None = None
        next_sell_trigger: float | None = None

        for idx in df.index:
            price = float(df.at[idx, "close"])
            if pd.isna(price) or price <= 0:
                continue

            # BUY logic
            if can_buy and portions_bought < total_portions:
                if next_buy_trigger is None:
                    next_buy_trigger = price * (1.0 - drop_pct)
                    portions_bought += 1
                elif price <= next_buy_trigger:
                    next_buy_trigger = next_buy_trigger * (1.0 - drop_pct)
                    portions_bought += 1

            # SELL logic
            if can_sell and portions_bought > 0 and portions_sold < portions_bought:
                if next_sell_trigger is None:
                    ref_price = (
                        next_buy_trigger / (1.0 - drop_pct)
                        if next_buy_trigger is not None
                        else price
                    )
                    next_sell_trigger = ref_price * (1.0 + rise_pct)
                    if price >= next_sell_trigger:
                        next_sell_trigger = next_sell_trigger * (1.0 + rise_pct)
                        portions_sold += 1
                elif price >= next_sell_trigger:
                    next_sell_trigger = next_sell_trigger * (1.0 + rise_pct)
                    portions_sold += 1

        # --- Portfolio anchoring: override simulated state with actual trades ---
        anchored = False
        ctx = portfolio_context or {}

        if ctx.get("buy_count", 0) > 0 and ctx.get("last_buy_price") is not None:
            actual_buy_price = float(ctx["last_buy_price"])
            actual_buy_count = int(ctx.get("buy_count", 0))
            # Use actual buy count (capped at total_portions)
            portions_bought = min(actual_buy_count, total_portions)
            # Anchor next buy trigger from actual last buy price
            next_buy_trigger = actual_buy_price * (1.0 - drop_pct)
            anchored = True

        if ctx.get("sell_count", 0) > 0 and ctx.get("last_sell_price") is not None:
            actual_sell_price = float(ctx["last_sell_price"])
            actual_sell_count = int(ctx.get("sell_count", 0))
            portions_sold = min(actual_sell_count, portions_bought)
            # Anchor next sell trigger from actual last sell price
            next_sell_trigger = actual_sell_price * (1.0 + rise_pct)
            anchored = True

        return {
            "portions_bought": portions_bought,
            "portions_sold": portions_sold,
            "next_buy_trigger": next_buy_trigger,
            "next_sell_trigger": next_sell_trigger,
            "can_buy": can_buy,
            "can_sell": can_sell,
            "in_hold_zone": in_hold_zone,
            "is_pure_price_mode": is_pure_price_mode,
            "zone_source": zone_source,
            "total_portions": total_portions,
            "drop_pct": drop_pct,
            "rise_pct": rise_pct,
            "portion_amount": portion_amount,
            "pe_buy_threshold": pe_buy_threshold,
            "pe_sell_threshold": pe_sell_threshold,
            "pe_percentile_buy": pe_pct_buy,
            "pe_percentile_sell": pe_pct_sell,
            "pe_percentile_val": (
                pe_percentile.pe_percentile
                if pe_percentile is not None and pe_percentile.pe_percentile is not None
                else None
            ),
            "anchored_by_portfolio": anchored,
        }

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    def get_live_signal(
        self,
        df: pd.DataFrame,
        info: dict,
        pe_value: float | None = None,
        pe_percentile: "PEPercentile | None" = None,
        macro_pulse: "MacroPulse | None" = None,
        portfolio_context: dict | None = None,
        **kwargs,
    ) -> LiveSignal:
        """Generate an actionable trading recommendation from current state.

        Runs a forward pass through *df* to determine how many portions have
        been bought/sold and where the next trigger levels sit, then compares
        the current price to produce a buy / sell / wait / hold action.

        When *portfolio_context* is provided (from actual executed trades),
        the anchor prices are taken from the user's real trade history rather
        than the historical simulation — so the next trigger price reflects
        what they actually paid.

        When *pe_percentile* is available (PE历史分位已接入), it replaces
        static PE thresholds with historical percentile-based zone logic.

        When *macro_pulse* is available, extreme fear (risk_level "high" or
        "extreme") appends a macro risk warning to the reason string. If
        ``macro_risk_filter`` is enabled, buy signals are suppressed entirely.
        """
        state = self._compute_state(
            df, pe_value, pe_percentile,
            portfolio_context=portfolio_context, **kwargs,
        )

        portions_bought = state["portions_bought"]
        next_buy_trigger = state["next_buy_trigger"]
        next_sell_trigger = state["next_sell_trigger"]
        can_buy = state["can_buy"]
        can_sell = state["can_sell"]
        in_hold_zone = state["in_hold_zone"]
        is_pure_price_mode = state["is_pure_price_mode"]
        zone_source = state["zone_source"]
        total_portions = state["total_portions"]
        portion_amount = state["portion_amount"]
        drop_pct = state["drop_pct"]
        rise_pct = state["rise_pct"]
        pe_percentile_val = state["pe_percentile_val"]

        # Current price: prefer real-time quote, fall back to last close
        current_price = info.get("current_price") if info else None
        if current_price is None and df is not None and len(df) > 0:
            df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
            current_price = float(df_sorted.iloc[-1]["close"])

        # ----------------------------------------------------------------
        # Determine action, trigger description, and next trigger price
        # ----------------------------------------------------------------
        action: str
        trigger: str
        next_trigger_price: float | None

        if in_hold_zone:
            action = "hold"
            trigger = "估值合理区，呆坐不动"
            next_trigger_price = None

        elif is_pure_price_mode:
            # Both buy and sell allowed — check each side
            if portions_bought < total_portions:
                if next_buy_trigger is None:
                    action = "buy"
                    trigger = "起始买入信号"
                    next_trigger_price = (
                        current_price * (1.0 - drop_pct) if current_price else None
                    )
                elif current_price is not None and current_price <= next_buy_trigger:
                    action = "buy"
                    trigger = "已跌破4%触发线"
                    next_trigger_price = current_price * (1.0 - drop_pct)
                else:
                    action = "wait_for_drop"
                    trigger = f"等待跌至{next_buy_trigger:.3f}"
                    next_trigger_price = next_buy_trigger
            elif portions_bought > 0:
                if next_sell_trigger is not None and current_price is not None and current_price >= next_sell_trigger:
                    action = "sell"
                    trigger = f"已涨破{next_sell_trigger:.3f}触发线"
                    next_trigger_price = next_sell_trigger * (1.0 + rise_pct)
                elif next_sell_trigger is not None:
                    action = "wait_for_rise"
                    trigger = f"等待涨至{next_sell_trigger:.3f}"
                    next_trigger_price = next_sell_trigger
                else:
                    action = "wait_for_rise"
                    trigger = "等待卖出参考价建立"
                    next_trigger_price = None
            else:
                action = "hold"
                trigger = "等待首笔交易建立参考价"
                next_trigger_price = None

        elif can_buy:
            # PE-buy zone only
            if portions_bought >= total_portions:
                action = "hold"
                trigger = f"已完成全部{total_portions}份买入"
                next_trigger_price = None
            elif next_buy_trigger is None:
                action = "buy"
                trigger = "起始买入信号"
                next_trigger_price = (
                    current_price * (1.0 - drop_pct) if current_price else None
                )
            elif current_price is not None and current_price <= next_buy_trigger:
                action = "buy"
                trigger = "已跌破4%触发线"
                next_trigger_price = current_price * (1.0 - drop_pct)
            else:
                action = "wait_for_drop"
                trigger = f"等待跌至{next_buy_trigger:.3f}"
                next_trigger_price = next_buy_trigger

        elif can_sell:
            # PE-sell zone only
            if portions_bought == 0:
                action = "hold"
                trigger = "无持仓可卖"
                next_trigger_price = None
            elif next_sell_trigger is not None and current_price is not None and current_price >= next_sell_trigger:
                action = "sell"
                trigger = f"已涨破{next_sell_trigger:.3f}触发线"
                next_trigger_price = next_sell_trigger * (1.0 + rise_pct)
            elif next_sell_trigger is not None:
                action = "wait_for_rise"
                trigger = f"等待涨至{next_sell_trigger:.3f}"
                next_trigger_price = next_sell_trigger
            else:
                action = "wait_for_rise"
                trigger = "等待卖出参考价建立"
                next_trigger_price = None

        else:
            # Fallback (e.g. no valid PE in PE mode)
            action = "hold"
            trigger = "无有效信号"
            next_trigger_price = None

        # ----------------------------------------------------------------
        # Urgency
        # ----------------------------------------------------------------
        if action in ("buy", "sell"):
            urgency = "high"
        elif action in ("wait_for_drop", "wait_for_rise"):
            urgency = "medium"
        else:
            urgency = "low"

        # ----------------------------------------------------------------
        # Zone label for display
        # ----------------------------------------------------------------
        if in_hold_zone:
            zone_label = "合理区"
            if pe_percentile_val is not None:
                zone_label = f"合理区（分位 {pe_percentile_val:.0f}%）"
        elif is_pure_price_mode:
            zone_label = "纯价格模式"
        elif can_buy:
            zone_label = "低估区"
            if pe_percentile_val is not None:
                zone_label = f"低估区（分位 {pe_percentile_val:.0f}%）"
        elif can_sell:
            zone_label = "高估区"
            if pe_percentile_val is not None:
                zone_label = f"高估区（分位 {pe_percentile_val:.0f}%）"
        else:
            zone_label = "无PE数据"

        # ----------------------------------------------------------------
        # Suggested shares — dynamic lot-aligned sizing
        # ----------------------------------------------------------------
        suggested_shares = _compute_lot_aligned_shares(
            portion_amount=portion_amount,
            current_price=current_price,
            total_portions=total_portions,
            portions_bought=portions_bought,
            portfolio_context=portfolio_context,
        )
        suggested_amount = (
            suggested_shares * current_price if current_price and suggested_shares else portion_amount
        )

        # ── Macro risk overlay ──
        macro_warning = ""
        if macro_pulse is not None:
            macro_warning = _format_macro_warning(macro_pulse)
            # Optionally suppress buy signals during extreme macro fear
            macro_risk_filter = kwargs.get("macro_risk_filter", False)
            if (
                macro_risk_filter
                and action in ("buy",)
                and macro_pulse.risk_level in ("high", "extreme")
            ):
                action = "hold"
                trigger = f"宏观风险({macro_pulse.risk_level})，暂停买入。{trigger}"

        reason = trigger
        if macro_warning and action != "hold":
            reason = f"{trigger}。{macro_warning}"

        # ── Portfolio anchor indicator ──
        _anchored = state.get("anchored_by_portfolio", False)
        if _anchored and portfolio_context:
            _last_buy = portfolio_context.get("last_buy_price")
            _last_sell = portfolio_context.get("last_sell_price")
            _anchor_note_parts = []
            if _last_buy is not None:
                _anchor_note_parts.append(f"上次买入 ¥{_last_buy:.3f}")
            if _last_sell is not None:
                _anchor_note_parts.append(f"上次卖出 ¥{_last_sell:.3f}")
            if _anchor_note_parts:
                reason = f"{reason}（基于实际成交：{'，'.join(_anchor_note_parts)}）"

        return LiveSignal(
            action=action,
            current_price=current_price,
            suggested_shares=suggested_shares,
            suggested_amount=suggested_amount,
            trigger_description=trigger,
            next_trigger_price=next_trigger_price,
            reason=reason,
            urgency_level=urgency,
            portions_used=portions_bought,
            portions_total=total_portions,
            current_zone=zone_label,
        )

    # ------------------------------------------------------------------
    # Dashboard cards
    # ------------------------------------------------------------------

    def get_dashboard_cards(
        self,
        df: pd.DataFrame,
        info: dict,
        pe_value: float | None = None,
        pe_percentile: "PEPercentile | None" = None,
        macro_pulse: "MacroPulse | None" = None,
        portfolio_context: dict | None = None,
        **kwargs,
    ) -> list[DashboardCard]:
        """Return strategy-specific info cards for the dashboard grid.

        Cards:
        1. 定投进度 — progress
        2. 下次买入触发价 — trigger
        3. 下次卖出触发价 — trigger
        4. PE分位 — pe_percentile or info
        5. 宏观情绪 — macro_pulse (if available)
        """
        state = self._compute_state(
            df, pe_value, pe_percentile,
            portfolio_context=portfolio_context, **kwargs,
        )

        portions_bought = state["portions_bought"]
        total_portions = state["total_portions"]
        next_buy_trigger = state["next_buy_trigger"]
        next_sell_trigger = state["next_sell_trigger"]
        portion_amount = state["portion_amount"]
        pe_buy_threshold = state["pe_buy_threshold"]
        pe_sell_threshold = state["pe_sell_threshold"]
        in_hold_zone = state["in_hold_zone"]
        can_buy = state["can_buy"]
        can_sell = state["can_sell"]
        is_pure_price_mode = state["is_pure_price_mode"]

        # Current price: prefer real-time quote, fall back to last close
        current_price = info.get("current_price") if info else None
        if current_price is None and df is not None and len(df) > 0:
            df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
            current_price = float(df_sorted.iloc[-1]["close"])

        # Derived numbers
        used_amount = portions_bought * portion_amount
        remaining = (total_portions - portions_bought) * portion_amount

        drop_needed_pct: float | None
        if next_buy_trigger is not None and current_price:
            drop_needed_pct = round(
                (current_price - next_buy_trigger) / current_price * 100, 2
            )
        else:
            drop_needed_pct = None

        rise_needed_pct: float | None
        if next_sell_trigger is not None and current_price:
            rise_needed_pct = round(
                (next_sell_trigger - current_price) / current_price * 100, 2
            )
        else:
            rise_needed_pct = None

        # PE percentile info
        pe_percentile_val = state.get("pe_percentile_val")
        zone_source = state.get("zone_source", "none")

        # Zone label
        if in_hold_zone:
            zone_label = "合理区"
        elif is_pure_price_mode:
            zone_label = "纯价格模式"
        elif can_buy:
            zone_label = "低估区"
        elif can_sell:
            zone_label = "高估区"
        else:
            zone_label = "无PE数据"

        # Add percentile suffix if available
        if pe_percentile_val is not None:
            zone_label = f"{zone_label}（分位 {pe_percentile_val:.0f}%）"

        # Build PE card content — richer when percentile data is available
        if pe_percentile is not None and pe_percentile.pe_percentile is not None:
            pe_card_type = "pe_percentile"
            pe_card_content = {
                "current_pe": pe_percentile.current_pe,
                "pe_percentile": pe_percentile.pe_percentile,
                "pe_mean": pe_percentile.pe_mean,
                "pe_median": pe_percentile.pe_median,
                "pe_plus_1std": pe_percentile.pe_plus_1std,
                "pe_minus_1std": pe_percentile.pe_minus_1std,
                "pe_min_5yr": pe_percentile.pe_min_5yr,
                "pe_max_5yr": pe_percentile.pe_max_5yr,
                "data_points": pe_percentile.data_points,
                "date_range": pe_percentile.date_range,
                "index_name": pe_percentile.index_name,
                "zone_label": zone_label,
                "zone_color": pe_percentile.zone_color,
            }
            pe_card_title = f"PE分位 · {pe_percentile.index_name}"
        else:
            pe_card_type = "info"
            pe_card_content = {
                "current_pe": pe_value,
                "pe_buy_threshold": pe_buy_threshold,
                "pe_sell_threshold": pe_sell_threshold,
                "zone_label": zone_label,
            }
            pe_card_title = "PE区间"

        name_slug = self.name

        return [
            DashboardCard(
                card_id=f"{name_slug}_progress",
                title="定投进度",
                card_type="progress",
                content={
                    "portions_bought": portions_bought,
                    "total_portions": total_portions,
                    "used_amount": used_amount,
                    "remaining": remaining,
                    "portion_amount": portion_amount,
                },
                priority=1,
            ),
            DashboardCard(
                card_id=f"{name_slug}_buy_trigger",
                title="下次买入触发价",
                card_type="trigger",
                content={
                    "next_trigger": round(next_buy_trigger, 3) if next_buy_trigger is not None else None,
                    "current_price": round(current_price, 3) if current_price else None,
                    "drop_needed_pct": drop_needed_pct,
                },
                priority=2,
            ),
            DashboardCard(
                card_id=f"{name_slug}_sell_trigger",
                title="下次卖出触发价",
                card_type="trigger",
                content={
                    "next_trigger": round(next_sell_trigger, 3) if next_sell_trigger is not None else None,
                    "current_price": round(current_price, 3) if current_price else None,
                    "rise_needed_pct": rise_needed_pct,
                },
                priority=2,
            ),
            DashboardCard(
                card_id=f"{name_slug}_pe_zone",
                title=pe_card_title,
                card_type=pe_card_type,
                content=pe_card_content,
                priority=2,
            ),
        ] + _build_macro_card(macro_pulse, name_slug)


# ---------------------------------------------------------------------------
# Macro pulse helpers (module-level, shared across strategy methods)
# ---------------------------------------------------------------------------


def _format_macro_warning(pulse: "MacroPulse") -> str:
    """Return a concise macro risk warning string for the live signal."""
    if pulse is None or pulse.total_signals == 0:
        return ""

    risk = pulse.risk_level
    if risk in ("high", "extreme"):
        return (
            f"⚠️ 宏观情绪偏弱（指数 {pulse.overall_sentiment:.2f}），"
            f"建议关注宏观风险"
        )
    if risk == "elevated":
        return (
            f"📊 宏观情绪中性偏弱（指数 {pulse.overall_sentiment:.2f}）"
        )
    return ""


def _build_macro_card(
    pulse: "MacroPulse | None",
    name_slug: str,
) -> list:
    """Build a ``DashboardCard`` list (0 or 1 element) for macro sentiment."""
    if pulse is None or pulse.total_signals == 0:
        return []

    risk_label_map = {
        "extreme": "极端恐惧",
        "high": "高度恐惧",
        "elevated": "偏高风险",
        "low": "情绪正常",
    }

    modules_data = []
    for mod_key, mod_info in pulse.modules_detail.items():
        modules_data.append({
            "icon": _MODULE_ICONS_MAP.get(mod_key, "📌"),
            "label": mod_info.get("label", mod_key),
            "avg": mod_info.get("avg_sentiment", 0.5),
        })

    warning = ""
    if pulse.risk_level in ("high", "extreme"):
        warning = (
            f"⚠️ 宏观情绪处于{pulse.risk_level}区间，"
            f"建议开启 macro_risk_filter 暂停买入"
        )

    return [DashboardCard(
        card_id=f"{name_slug}_macro_pulse",
        title=f"宏观情绪 · {pulse.refreshed_at}",
        card_type="macro_pulse",
        content={
            "overall_sentiment": pulse.overall_sentiment,
            "risk_level": pulse.risk_level,
            "risk_color": pulse.risk_color,
            "risk_label": risk_label_map.get(pulse.risk_level, pulse.risk_level),
            "modules": modules_data,
            "warning": warning,
        },
        priority=2,
    )]


# Small icon map for module keys
_MODULE_ICONS_MAP: dict[str, str] = {
    "monetary": "💰",
    "macro": "📊",
    "geopolitics": "🌍",
    "commodities": "🛢️",
    "ai_tech": "🤖",
}


# ---------------------------------------------------------------------------
# Lot-aligned dynamic portion sizing
# ---------------------------------------------------------------------------


def _compute_lot_aligned_shares(
    portion_amount: float,
    current_price: float | None,
    total_portions: int,
    portions_bought: int,
    portfolio_context: dict | None = None,
) -> int:
    """Compute the recommended number of shares for one DCA portion,
    rounded down to whole lots (100-share increments).

    When *portfolio_context* provides ``available_cash``, the portion
    size is dynamically derived from available capital rather than the
    fixed ``portion_amount`` parameter::

        remaining_portions = total_portions - portions_bought
        budget_per_portion = available_cash / max(remaining_portions, 1)
        raw_shares = budget_per_portion / current_price
        shares = round_down_to_lot(raw_shares)

    Falls back to the static ``portion_amount`` when no portfolio
    context is available.
    """
    if current_price is None or current_price <= 0:
        return 0

    # --- Dynamic sizing from portfolio cash ---
    if portfolio_context and portfolio_context.get("available_cash", 0) > 0:
        available_cash = float(portfolio_context["available_cash"])
        remaining = total_portions - portions_bought
        if remaining < 1:
            remaining = 1

        # Per-portion budget = available cash / remaining portions
        budget_per_portion = available_cash / remaining

        # Safety ceiling: single trade never exceeds 25% of available cash
        max_budget = available_cash * 0.25
        budget = min(budget_per_portion, max_budget)

        raw_shares = int(budget / current_price)
        # Round down to nearest lot (100 shares)
        shares = (raw_shares // 100) * 100
        return max(shares, 0)

    # --- Static portion_amount fallback ---
    if portion_amount > 0:
        raw_shares = int(portion_amount / current_price)
        shares = (raw_shares // 100) * 100
        return max(shares, 0)

    return 0

