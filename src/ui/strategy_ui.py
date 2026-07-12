"""策略回测 UI — 参数表单、净值曲线、对比视图"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.engine.metrics import compute_drawdown_series
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK, BG_CARD, BORDER,
    CHART_COLORS, FONT, apply_chart_theme,
)


# ---------------------------------------------------------------------------
# Param form
# ---------------------------------------------------------------------------

def _render_param_form(strategy, prefix: str = "") -> dict:
    """Render strategy parameters.

    Defaults are pre-optimised — most users don't need to change them.
    Advanced tweaking is hidden behind a collapsed expander so the sidebar
    stays clean for first-time users.
    """
    descs = strategy.get_param_descriptions()
    defaults = strategy.get_default_params()

    # ── Quick-access: the two params users actually change ──
    values: dict = {}
    quick_keys = ["total_portions", "portion_amount"]

    for key in quick_keys:
        if key not in defaults:
            continue
        desc = descs.get(key, {})
        label = desc.get("label", key)
        help_text = desc.get("help", "")
        param_type = desc.get("type", "number")
        widget_key = f"{prefix}_{key}"
        default = defaults[key]
        if param_type == "select":
            options = desc.get("options", [str(default)])
            idx = options.index(str(default)) if str(default) in options else 0
            values[key] = st.selectbox(
                label, options, index=idx, key=widget_key, help=help_text,
            )
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

    # ── Advanced params: collapsed by default ──
    remaining = {k: v for k, v in defaults.items() if k not in quick_keys}
    if remaining:
        with st.expander("⚙️ 高级参数（一般不用改）", expanded=False):
            st.caption("以下参数已预设最优值，改之前建议先跑一次回测看看效果。")
            for key, default in remaining.items():
                desc = descs.get(key, {})
                label = desc.get("label", key)
                help_text = desc.get("help", "")
                param_type = desc.get("type", "number")
                widget_key = f"{prefix}_{key}"
                if param_type == "select":
                    options = desc.get("options", [str(default)])
                    idx = options.index(str(default)) if str(default) in options else 0
                    values[key] = st.selectbox(
                        label, options, index=idx, key=widget_key, help=help_text,
                    )
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

    # Ensure every default key has a value (safety net)
    for key in defaults:
        if key not in values:
            values[key] = defaults[key]

    return values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _render_metrics(result) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("年化收益", f"{result.annual_return:+.1%}")
    with c2: st.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
    with c3: st.metric("最大回撤", f"{result.max_drawdown:.1%}")
    with c4: st.metric("Calmar", f"{result.calmar_ratio:.2f}")
    with c5: st.metric("胜率", f"{result.win_rate:.0%}")
    with c6: st.metric("交易次数", str(result.total_trades))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _render_equity_chart(result, title: str = "净值曲线") -> None:
    eq = result.equity_curve
    if eq.empty:
        st.warning("暂无权益数据")
        return

    equity = eq.set_index("date")["equity"]
    dd = compute_drawdown_series(equity)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.65, 0.35],
        subplot_titles=("净值曲线", "回撤"),
    )

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, mode="lines",
        name=title, line=dict(color=CHART_COLORS[0], width=2),
    ), row=1, col=1)
    fig.add_hline(y=result.initial_capital, line_dash="dash",
                  line_color=NEUTRAL, opacity=0.4, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="回撤",
        fill="tozeroy", line=dict(color=DANGER, width=1),
    ), row=2, col=1)

    apply_chart_theme(fig, height=500)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="资产 (元)", row=1, col=1)
    fig.update_yaxes(title_text="回撤", row=2, col=1, tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_trade_table(result) -> None:
    trades = result.trades
    if not trades:
        st.info("无已完成交易")
        return

    rows = []
    for t in trades:
        rows.append({
            "入场日期": str(t.entry_date.date()) if t.entry_date else "",
            "出场日期": str(t.exit_date.date()) if t.exit_date else "持仓中",
            "入场价": f"{t.entry_price:.3f}",
            "出场价": f"{t.exit_price:.3f}" if t.exit_price else "—",
            "股数": t.shares,
            "盈亏(元)": f"{t.pnl:+.0f}" if t.pnl is not None else "—",
            "盈亏%": f"{t.pnl_pct:+.1%}" if t.pnl_pct is not None else "—",
            "持仓天数": t.holding_days if t.holding_days else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_comparison(results: list) -> None:
    if not results: return

    st.subheader("策略对比")

    table_rows = []
    for r in results:
        table_rows.append({
            "策略": r.strategy_name,
            "年化收益": f"{r.annual_return:+.1%}",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "最大回撤": f"{r.max_drawdown:.1%}",
            "Calmar": f"{r.calmar_ratio:.2f}",
            "胜率": f"{r.win_rate:.0%}",
            "交易次数": r.total_trades,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    best = max(results, key=lambda r: r.sharpe_ratio)
    st.success(f"🏆 推荐策略: {best.strategy_name} | 年化 {best.annual_return:+.1%} | Sharpe {best.sharpe_ratio:.2f}")

    st.subheader("净值曲线对比")
    colors = CHART_COLORS[:len(results)]
    fig = go.Figure()
    for i, r in enumerate(results):
        eq = r.equity_curve
        if eq.empty: continue
        equity = eq.set_index("date")["equity"]
        norm = equity / equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, mode="lines",
            name=r.strategy_name, line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=NEUTRAL, opacity=0.4)
    apply_chart_theme(fig, height=420)
    fig.update_yaxes(title_text="归一化净值 (起始=1)", tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# (render_strategy_page has moved to app.py — the strategy-centric layout)
# ---------------------------------------------------------------------------
