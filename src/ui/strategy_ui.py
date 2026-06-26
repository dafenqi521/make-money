"""Strategy backtesting UI — Retro-Futuristic terminal style."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.engine.backtest import BacktestEngine
from src.engine.broker import Broker
from src.engine.risk import RiskManager
from src.strategy.trend_following import TrendFollowingStrategy
from src.strategy.grid_trading import GridTradingStrategy
from src.strategy.value_averaging import ValueAveragingStrategy
from src.strategy.hybrid import HybridStrategy
from src.engine.metrics import compute_drawdown_series
from src.ui.terminal_theme import (
    GREEN, MAGENTA, CYAN, AMBER, VIOLET,
    TEXT, TEXT_DIM, PANEL, CARD, BORDER,
    CHART_COLORS, MA_COLORS,
    FONT_MONO,
    apply_terminal_chart,
    section_header,
    status_badge,
)


STRATEGIES = [
    TrendFollowingStrategy(),
    GridTradingStrategy(),
    ValueAveragingStrategy(),
    HybridStrategy(),
]
STRATEGY_MAP = {s.name: s for s in STRATEGIES}


# ---------------------------------------------------------------------------
# Parameter form
# ---------------------------------------------------------------------------

def _render_param_form(strategy, prefix: str = "") -> dict:
    descs = strategy.get_param_descriptions()
    defaults = strategy.get_default_params()
    values: dict = {}
    for key, default in defaults.items():
        desc = descs.get(key, {})
        label = desc.get("label", key)
        help_text = desc.get("help", "")
        param_type = desc.get("type", "number")
        widget_key = f"{prefix}_{key}"
        if param_type == "select":
            options = desc.get("options", [str(default)])
            idx = options.index(str(default)) if str(default) in options else 0
            values[key] = st.selectbox(label, options, index=idx, key=widget_key, help=help_text)
        elif param_type == "slider":
            values[key] = st.slider(
                label, min_value=float(desc.get("min", 0)),
                max_value=float(desc.get("max", 100)), value=float(default),
                step=float(desc.get("step", 1)), key=widget_key, help=help_text,
            )
        else:
            values[key] = st.number_input(
                label, value=float(default) if isinstance(default, (int, float)) else default,
                min_value=float(desc.get("min", 0)), max_value=float(desc.get("max", 1e9)),
                step=float(desc.get("step", 1)), key=widget_key, help=help_text,
            )
    return values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _render_metrics(result) -> None:
    section_header("PERFORMANCE")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ANNUAL", f"{result.annual_return:+.1%}")
    with c2:
        st.metric("SHARPE", f"{result.sharpe_ratio:.2f}")
    with c3:
        st.metric("MAX DD", f"{result.max_drawdown:.1%}")
    with c4:
        st.metric("CALMAR", f"{result.calmar_ratio:.2f}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("WIN RATE", f"{result.win_rate:.0%}")
    with c2:
        st.metric("TRADES", str(result.total_trades))
    with c3:
        st.metric("W / L", f"{result.winning_trades}/{result.losing_trades}")
    with c4:
        st.metric("AVG HOLD", f"{result.avg_holding_days:.0f}d")


# ---------------------------------------------------------------------------
# Equity curve chart
# ---------------------------------------------------------------------------

def _render_equity_chart(result, title: str = "EQUITY") -> None:
    eq = result.equity_curve
    if eq.empty:
        st.warning("NO EQUITY DATA")
        return

    equity = eq.set_index("date")["equity"]
    dd = compute_drawdown_series(equity)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.65, 0.35],
        subplot_titles=("EQUITY CURVE", "DRAWDOWN"),
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name=title,
            line=dict(color=GREEN, width=2),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=result.initial_capital, line_dash="dash",
                  line_color=TEXT_DIM, opacity=0.4, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="DD",
            fill="tozeroy",
            line=dict(color=MAGENTA, width=1),
        ),
        row=2, col=1,
    )

    apply_terminal_chart(fig, height=500)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="CNY", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1, tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Trade table
# ---------------------------------------------------------------------------

def _render_trade_table(result) -> None:
    trades = result.trades
    if not trades:
        st.info("NO CLOSED TRADES")
        return

    rows = []
    for t in trades:
        rows.append({
            "ENTRY": str(t.entry_date.date()) if t.entry_date else "",
            "EXIT": str(t.exit_date.date()) if t.exit_date else "OPEN",
            "PRICE IN": f"{t.entry_price:.3f}",
            "PRICE OUT": f"{t.exit_price:.3f}" if t.exit_price else "—",
            "QTY": t.shares,
            "PNL": f"{t.pnl:+.0f}" if t.pnl is not None else "—",
            "PNL%": f"{t.pnl_pct:+.1%}" if t.pnl_pct is not None else "—",
            "DAYS": t.holding_days if t.holding_days else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Comparison view
# ---------------------------------------------------------------------------

def _render_comparison(results: list) -> None:
    if not results:
        return

    section_header("COMPARISON")

    table_rows = []
    for r in results:
        table_rows.append({
            "STRATEGY": r.strategy_name,
            "ANNUAL": f"{r.annual_return:+.1%}",
            "SHARPE": f"{r.sharpe_ratio:.2f}",
            "MAX DD": f"{r.max_drawdown:.1%}",
            "CALMAR": f"{r.calmar_ratio:.2f}",
            "WIN%": f"{r.win_rate:.0%}",
            "TRADES": r.total_trades,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    best = max(results, key=lambda r: r.sharpe_ratio)
    st.success(
        f"▶ RECOMMENDED: {best.strategy_name}  |  "
        f"ANNUAL {best.annual_return:+.1%}  |  "
        f"SHARPE {best.sharpe_ratio:.2f}  |  "
        f"MAX DD {best.max_drawdown:.1%}"
    )

    section_header("EQUITY OVERLAY")
    colors = CHART_COLORS[:len(results)]
    fig = go.Figure()
    for i, r in enumerate(results):
        eq = r.equity_curve
        if eq.empty:
            continue
        equity = eq.set_index("date")["equity"]
        norm = equity / equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=norm.index, y=norm.values,
                mode="lines", name=r.strategy_name,
                line=dict(color=colors[i % len(colors)], width=2),
            ),
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color=TEXT_DIM, opacity=0.4)
    apply_terminal_chart(fig, height=420)
    fig.update_yaxes(title_text="NORMALIZED (BASE=1)", tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_strategy_page(df: pd.DataFrame, info: dict) -> None:
    if df.empty:
        st.warning("NO HISTORICAL DATA")
        return

    pe_value = info.get("pe_ttm") or info.get("pe_static")

    # Sidebar strategy selection
    st.sidebar.markdown(f"""
    <div style="font-family:{FONT_MONO}; font-size:0.65rem; color:{TEXT_DIM};
                text-transform:uppercase; letter-spacing:2px;
                margin:16px 0 8px 0;">
        ▸ STRATEGY
    </div>
    """, unsafe_allow_html=True)

    strategy_names = [s.name for s in STRATEGIES]
    selected_name = st.sidebar.radio(
        "STRATEGY", strategy_names, index=2, label_visibility="collapsed",
    )
    strategy = STRATEGY_MAP[selected_name]
    compare_mode = st.sidebar.checkbox("COMPARE ALL 4")

    # Description
    with st.expander(f"▸ {strategy.name.upper()} — BRIEF", expanded=False):
        st.write(strategy.description)

    # PE warning
    if selected_name in ("估值定投", "网格+定投"):
        if pe_value is None:
            st.warning("PE(TTM) UNAVAILABLE — USING BASE AMOUNT (1x)")
        else:
            st.info(f"PE(TTM): {pe_value:.1f}  |  SNAPSHOT ONLY — NOT PERCENTILE-BASED")

    # Parameters
    st.sidebar.markdown(f"""
    <div style="font-family:{FONT_MONO}; font-size:0.65rem; color:{TEXT_DIM};
                text-transform:uppercase; letter-spacing:2px;
                margin:16px 0 8px 0;">
        ▸ PARAMETERS
    </div>
    """, unsafe_allow_html=True)

    if compare_mode:
        st.sidebar.caption("USING DEFAULTS")
        params = {}
    else:
        params = _render_param_form(strategy, prefix="s")

    # Run
    if st.sidebar.button("▶ EXECUTE BACKTEST", type="primary", use_container_width=True):
        with st.spinner("RUNNING..."):
            engine = BacktestEngine(initial_capital=100_000, broker=Broker(), risk_manager=RiskManager())

            if compare_mode:
                results = []
                for s in STRATEGIES:
                    sp = s.get_default_params()
                    r = engine.run(df.copy(), s, pe_value=pe_value, **sp)
                    results.append(r)
                _render_comparison(results)
            else:
                result = engine.run(df.copy(), strategy, pe_value=pe_value, **params)

                _render_metrics(result)

                st.divider()
                tab1, tab2 = st.tabs(["EQUITY CURVE", "TRADE LOG"])
                with tab1:
                    _render_equity_chart(result, title=strategy.name.upper())
                with tab2:
                    st.caption(f"{result.total_trades} TRADES")
                    _render_trade_table(result)

                # Warnings
                if selected_name in ("网格交易", "网格+定投"):
                    grid_p = params if params else strategy.get_default_params()
                    step = grid_p.get("position_per_grid_pct", 0.08)
                    price = info.get("current_price", 10)
                    rm = RiskManager()
                    should, msg = rm.check_step_size(step * price, price)
                    if should:
                        st.warning(msg)

                st.caption(result.summary())
    else:
        st.info("CONFIGURE PARAMETERS AND EXECUTE")
