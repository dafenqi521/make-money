"""Portfolio dashboard UI — paper-trading position tracker.

Renders portfolio summary cards, trade execution buttons (driven by
strategy live signals), holdings table, and trade history.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.engine.portfolio import PortfolioManager, LOT_SIZE
from src.strategy.signals import LiveSignal
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
    _styler_apply,
)


# ---------------------------------------------------------------------------
# Portfolio summary cards
# ---------------------------------------------------------------------------

def render_portfolio_summary(pm: PortfolioManager) -> None:
    """Render key portfolio metrics in a row of cards."""
    s = pm.summary()
    equity = s["total_equity"]
    pnl = s["total_pnl"]
    pnl_color = SUCCESS if pnl >= 0 else DANGER
    pnl_sign = "+" if pnl > 0 else ""

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div style="padding:12px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.7rem; color:{NEUTRAL};">总资产</div>'
            f'<div style="font-size:1.25rem; font-weight:700; color:{DARK}; '
            f'font-family:{FONT_MONO};">¥{equity:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div style="padding:12px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.7rem; color:{NEUTRAL};">可用现金</div>'
            f'<div style="font-size:1.25rem; font-weight:700; color:{PRIMARY}; '
            f'font-family:{FONT_MONO};">¥{s["cash"]:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f'<div style="padding:12px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.7rem; color:{NEUTRAL};">持仓市值</div>'
            f'<div style="font-size:1.25rem; font-weight:700; color:{DARK}; '
            f'font-family:{FONT_MONO};">¥{s["market_value"]:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f'<div style="padding:12px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.7rem; color:{NEUTRAL};">累计盈亏</div>'
            f'<div style="font-size:1.25rem; font-weight:700; color:{pnl_color}; '
            f'font-family:{FONT_MONO};">{pnl_sign}¥{pnl:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f'<div style="padding:12px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.7rem; color:{NEUTRAL};">总收益率</div>'
            f'<div style="font-size:1.25rem; font-weight:700; color:{pnl_color}; '
            f'font-family:{FONT_MONO};">{pnl_sign}{s["total_return_pct"]}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Trade execution panel
# ---------------------------------------------------------------------------

def render_trade_panel(
    pm: PortfolioManager,
    signal: LiveSignal,
    code: str,
    name: str,
    current_price: float | None,
) -> None:
    """Render buy/sell action buttons driven by the current strategy signal.

    Only shows executable actions when the live signal recommends buy/sell
    AND the portfolio has sufficient cash (buy) or shares (sell).
    """
    if current_price is None or current_price <= 0:
        return

    holding = pm.get_holding(code)
    current_shares = holding.shares if holding else 0

    col_buy, col_sell, col_info = st.columns([1, 1, 2])

    # --- Buy button ---
    with col_buy:
        raw_shares = signal.suggested_shares if signal.action in ("buy",) else 0
        suggested_shares = (raw_shares // LOT_SIZE) * LOT_SIZE  # round to lots
        if signal.action in ("buy",) and suggested_shares > 0:
            suggested_amount = suggested_shares * current_price
            affordable = pm.cash > suggested_amount * 0.1

            lots = suggested_shares // LOT_SIZE
            if affordable:
                disabled = False
                btn_label = f"🟢 执行买入\n{suggested_shares}股 ({lots}手) ≈ ¥{suggested_amount:,.0f}"
                btn_help = signal.reason
            else:
                disabled = True
                btn_label = "🟢 买入 (现金不足)"
                btn_help = f"需要约 ¥{suggested_amount:,.0f}，可用现金 ¥{pm.cash:,.0f}"

            if st.button(
                btn_label, key="exec_buy_btn", use_container_width=True,
                disabled=disabled, type="primary" if not disabled else "secondary",
                help=btn_help,
            ):
                trade = pm.buy(
                    code=code, price=current_price,
                    shares=suggested_shares, name=name,
                    reason=f"[{signal.current_zone}] {signal.reason}",
                )
                if trade:
                    st.session_state["last_trade"] = trade
                    _save_to_db(pm)
                    st.rerun()
                else:
                    st.error("买入失败：现金不足")
        else:
            st.button(
                "买入 (无信号)", key="exec_buy_btn", use_container_width=True,
                disabled=True,
            )

    # --- Sell button ---
    with col_sell:
        if signal.action in ("sell",) and current_shares > 0:
            raw_sell = (signal.suggested_shares // LOT_SIZE) * LOT_SIZE if signal.suggested_shares else 0
            sell_shares = min(raw_sell, current_shares) if raw_sell > 0 else ((current_shares // LOT_SIZE) * LOT_SIZE)
            if sell_shares <= 0:
                sell_shares = (current_shares // LOT_SIZE) * LOT_SIZE

            lots = sell_shares // LOT_SIZE
            if st.button(
                f"🔴 执行卖出\n{sell_shares}股 ({lots}手) ≈ ¥{sell_shares * current_price:,.0f}",
                key="exec_sell_btn", use_container_width=True, type="primary",
                help=signal.reason,
            ):
                trade = pm.sell(
                    code=code, price=current_price,
                    shares=sell_shares, name=name,
                    reason=f"[{signal.current_zone}] {signal.reason}",
                )
                if trade:
                    st.session_state["last_trade"] = trade
                    _save_to_db(pm)
                    st.rerun()
                else:
                    st.error("卖出失败：持仓不足")
        elif signal.action in ("wait_for_rise", "sell") and current_shares == 0:
            st.button(
                "卖出 (无持仓)", key="exec_sell_btn", use_container_width=True,
                disabled=True,
            )
        else:
            st.button(
                f"卖出 (持有{current_shares}股)", key="exec_sell_btn",
                use_container_width=True, disabled=True,
            )

    # --- Info ---
    with col_info:
        if holding and holding.shares > 0:
            lots = holding.shares // LOT_SIZE
            st.markdown(
                f'<div style="padding:8px 12px; font-size:0.8rem; color:{NEUTRAL};">'
                f'持仓: <b style="color:{DARK};">{holding.shares}股 ({lots}手)</b> | '
                f'成本: <b style="color:{DARK};">¥{holding.avg_cost:.3f}</b> | '
                f'浮盈: <b style="color:{SUCCESS if holding.unrealized_pnl >= 0 else DANGER};">'
                f'{"+" if holding.unrealized_pnl >= 0 else ""}¥{holding.unrealized_pnl:,.0f} '
                f'({"+" if holding.unrealized_pnl_pct >= 0 else ""}'
                f'{holding.unrealized_pnl_pct:.2f}%)</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif signal.action in ("buy", "wait_for_drop"):
            st.caption(f"📋 {signal.trigger_description}" if signal.trigger_description else "")
        else:
            st.caption("暂无持仓 | 等待买入信号")


# ---------------------------------------------------------------------------
# Holdings + trade history tabs
# ---------------------------------------------------------------------------

def render_portfolio_details(pm: PortfolioManager) -> None:
    """Render holdings table and trade history in tabs."""
    tab1, tab2 = st.tabs(["📋 持仓明细", "📜 交易记录"])

    with tab1:
        holdings_list = pm.get_holdings_table()
        if holdings_list:
            df = pd.DataFrame(holdings_list)
            df = df.rename(columns={
                "code": "代码", "name": "名称", "shares": "持股数",
                "avg_cost": "成本价", "total_cost": "总成本",
                "current_price": "现价", "market_value": "市值",
                "unrealized_pnl": "浮动盈亏", "unrealized_pnl_pct": "盈亏%",
            })
            # Reorder
            display_cols = ["代码", "名称", "持股数", "成本价", "现价", "市值", "浮动盈亏", "盈亏%"]
            df = df[[c for c in display_cols if c in df.columns]]

            def _pnl_color(val):
                try:
                    v = float(val)
                    return f"color: {SUCCESS}; font-weight: 700;" if v >= 0 else f"color: {DANGER}; font-weight: 700;"
                except (ValueError, TypeError):
                    return ""

            styled = df.style
            if "盈亏%" in df.columns:
                styled = _styler_apply(styled, _pnl_color, ["盈亏%"])
            if "浮动盈亏" in df.columns:
                styled = _styler_apply(styled, _pnl_color, ["浮动盈亏"])

            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("暂无持仓。等待策略信号触发后执行买入。")

    with tab2:
        trade_hist = pm.get_trade_history(n=50)
        if trade_hist:
            df = pd.DataFrame(trade_hist)
            st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("暂无交易记录。")


# ---------------------------------------------------------------------------
# Initial capital setup
# ---------------------------------------------------------------------------

def render_portfolio_setup() -> float:
    """Render initial capital input and return the value.

    Only shown when no portfolio exists yet.
    """
    st.info("💡 尚未创建模拟持仓。设置初始资金开始纸面交易。")
    capital = st.number_input(
        "初始资金 (元)",
        min_value=1000.0, max_value=10_000_000.0,
        value=100_000.0, step=10000.0,
        key="portfolio_initial_capital",
    )
    if st.button("💰 创建模拟账户", type="primary", use_container_width=True):
        pm = PortfolioManager(initial_capital=capital)
        st.session_state["portfolio"] = pm
        st.rerun()
    return capital


# ---------------------------------------------------------------------------
# Main portfolio section renderer
# ---------------------------------------------------------------------------

def render_portfolio_section(
    signal: LiveSignal,
    code: str,
    name: str,
    current_price: float | None,
) -> None:
    """Main entry point: render the full portfolio tracking section.

    Shows setup if no portfolio exists, otherwise shows summary +
    trade panel + details.
    """
    st.divider()
    st.subheader("💰 模拟持仓")

    if "portfolio" not in st.session_state:
        # Try loading from SQLite DB before showing setup
        from src.data.portfolio_db import PortfolioDB
        db = PortfolioDB()
        pm_loaded = db.load()
        if pm_loaded is not None and pm_loaded.total_trades > 0:
            st.session_state["portfolio"] = pm_loaded
            st.toast("📂 已从数据库恢复上次的持仓和交易记录", icon="💾")
        else:
            render_portfolio_setup()
            return

    pm: PortfolioManager = st.session_state["portfolio"]

    # Update current prices in portfolio
    if current_price is not None and current_price > 0:
        pm.update_prices({code: current_price})

    # Persist portfolio across reruns
    st.session_state["portfolio"] = pm

    # --- Summary cards ---
    render_portfolio_summary(pm)

    # --- Trade execution (signal-driven quick buttons) ---
    render_trade_panel(pm, signal, code, name, current_price)

    # --- Manual trade entry (custom price / shares) ---
    render_manual_trade_form(pm, signal, code, name, current_price)

    # Show last trade confirmation
    if "last_trade" in st.session_state:
        t = st.session_state["last_trade"]
        action_label = "买入" if t.action == "buy" else "卖出"
        st.success(
            f"✅ {action_label}成功: {t.shares}股 @ ¥{t.price:.3f} | "
            f"金额 ¥{abs(t.net_amount):,.2f} | "
            + (f"盈亏 ¥{t.pnl:+,.0f}" if t.pnl is not None else "")
        )
        del st.session_state["last_trade"]

    # --- Details ---
    st.caption("")  # spacer
    render_portfolio_details(pm)

    # --- Reset / Re-init ---
    with st.expander("⚙️ 账户管理", expanded=False):
        st.caption(f"当前初始资金: ¥{pm.initial_capital:,.0f} | 佣金: 万三 | ETF免印花税")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            new_capital = st.number_input(
                "重新初始化金额 (元)",
                min_value=1000.0, max_value=10_000_000.0,
                value=float(pm.initial_capital), step=10000.0,
                key="portfolio_reset_capital",
            )
        with col_b:
            st.caption("")  # spacer
            st.caption("")  # spacer
            if st.button("🔄 按此金额重置", type="secondary", use_container_width=True):
                st.session_state["portfolio"] = PortfolioManager(
                    initial_capital=float(new_capital)
                )
                _reset_db()
                st.rerun()


# ---------------------------------------------------------------------------
# Manual trade entry — custom price / shares / notes
# ---------------------------------------------------------------------------


def render_manual_trade_form(
    pm: PortfolioManager,
    signal: LiveSignal,
    code: str,
    name: str,
    current_price: float | None,
) -> None:
    """Render a form that lets the user enter actual trade details.

    Unlike the signal-driven quick buttons, this form allows the user to
    specify the exact price, shares, and amount they actually traded.
    After recording, the portfolio's trade history is used by strategies
    to adjust future signal computation (e.g. next trigger prices).
    """
    if current_price is None or current_price <= 0:
        return

    with st.expander("✏️ 手动录入交易", expanded=False):
        st.caption("如果你的实际成交价/数量与建议不同，在这里手动录入。系统会根据实际成交调整后续决策。")

        direction = st.radio(
            "交易方向",
            options=["buy", "sell"],
            format_func=lambda d: "🟢 买入" if d == "buy" else "🔴 卖出",
            horizontal=True,
            key="manual_trade_direction",
        )

        # Smart defaults based on signal
        if direction == "buy":
            default_shares = signal.suggested_shares if signal.suggested_shares > 0 else 0
            default_amount = signal.suggested_amount if signal.suggested_amount > 0 else 0.0
        else:
            holding = pm.get_holding(code)
            default_shares = holding.shares if holding else 0
            default_amount = default_shares * current_price

        col1, col2, col3 = st.columns(3)
        with col1:
            price = st.number_input(
                "成交单价 (元)",
                min_value=0.001, max_value=99999.999,
                value=round(float(current_price), 3),
                step=0.001,
                format="%.3f",
                key="manual_trade_price",
                help="你实际成交的价格",
            )
        with col2:
            shares = st.number_input(
                "成交股数",
                min_value=100, max_value=10_000_000,
                value=max(default_shares, 100),
                step=100,
                key="manual_trade_shares",
                help="A股最低100股（1手），必须是100的整数倍",
            )
        with col3:
            # Auto-calculate amount but allow override
            calc_amount = price * shares
            amount = st.number_input(
                "成交金额 (元)",
                min_value=1.0, max_value=100_000_000.0,
                value=round(float(calc_amount), 2),
                step=100.0,
                format="%.2f",
                key="manual_trade_amount",
                help="自动 = 单价 × 股数，可手动覆盖",
            )

        reason = st.text_input(
            "交易原因（可选）",
            value=signal.reason if signal.reason else "",
            placeholder="如：按计划定投第3份 / 网格触发 / 手动加仓",
            key="manual_trade_reason",
        )

        notes = st.text_input(
            "备注（可选）",
            placeholder="如：实际在XX券商以4.82成交",
            key="manual_trade_notes",
        )

        if st.button("📝 确认录入", type="primary", use_container_width=True,
                     key="manual_trade_submit_btn"):
            # Validate
            if shares % 100 != 0:
                st.error("⚠️ A股最低交易单位100股（1手），股数必须是100的整数倍")
                return

            full_reason = reason
            if notes:
                full_reason = f"{reason}（{notes}）" if reason else notes
            if not full_reason:
                full_reason = "手动录入"

            if direction == "buy":
                trade = pm.buy(
                    code=code, price=price, shares=shares,
                    name=name, reason=full_reason,
                )
                if trade is None:
                    st.error(f"买入失败：现金不足（可用 ¥{pm.cash:,.0f}，需要约 ¥{price * shares:,.0f}）")
                    return
            else:
                trade = pm.sell(
                    code=code, price=price, shares=shares,
                    name=name, reason=full_reason,
                )
                if trade is None:
                    st.error("卖出失败：持仓不足或数量无效")
                    return

            st.session_state["last_trade"] = trade
            _save_to_db(pm)
            st.rerun()


def get_portfolio_context(pm: PortfolioManager, code: str) -> dict:
    """Extract trading context from portfolio for strategy signal adjustment.

    Returns a dict with keys that strategies can use to anchor their
    signal computation on actual executed trades rather than pure
    historical simulation:

      - last_buy_price: most recent buy price for this code
      - last_sell_price: most recent sell price for this code
      - buy_count: number of buy trades for this code
      - sell_count: number of sell trades for this code
      - holding_shares: current position size
      - holding_avg_cost: average cost basis
      - has_position: whether we currently hold this ETF
      - available_cash: uninvested cash in the portfolio
      - total_equity: cash + market value of all holdings
    """
    ctx: dict = {
        "last_buy_price": None,
        "last_sell_price": None,
        "buy_count": 0,
        "sell_count": 0,
        "holding_shares": 0,
        "holding_avg_cost": 0.0,
        "has_position": False,
        "available_cash": pm.cash,
        "total_equity": pm.summary()["total_equity"],
    }

    # Count trades and find last prices
    for t in pm.trades:
        if t.code != code:
            continue
        if t.action == "buy":
            ctx["last_buy_price"] = t.price
            ctx["buy_count"] += 1
        elif t.action == "sell":
            ctx["last_sell_price"] = t.price
            ctx["sell_count"] += 1

    # Holding info
    holding = pm.get_holding(code)
    if holding and holding.shares > 0:
        ctx["holding_shares"] = holding.shares
        ctx["holding_avg_cost"] = holding.avg_cost
        ctx["has_position"] = True

    return ctx


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------


def _save_to_db(pm: PortfolioManager) -> None:
    """Persist portfolio to SQLite after a trade (fire-and-forget)."""
    try:
        from src.data.portfolio_db import PortfolioDB
        PortfolioDB().save(pm)
    except Exception:
        pass  # Non-critical — portfolio still lives in session_state


def _reset_db() -> None:
    """Wipe the SQLite portfolio database."""
    try:
        from src.data.portfolio_db import PortfolioDB
        PortfolioDB().reset()
    except Exception:
        pass
