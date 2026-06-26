"""Strategy backtesting UI — parameter forms, equity charts, comparison view.

All visual constants from ``src.ui.theme``.
"""

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
from src.ui.theme import (
    PRIMARY,
    SUCCESS,
    DANGER,
    WARNING,
    NEUTRAL,
    DARK,
    BG_CARD,
    BORDER,
    CHART_COLORS,
    apply_chart_theme,
    chart_layout,
    make_equity_colors,
    section_header,
    metric_row,
    info_banner,
    inject_css,
)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

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
    """Render dynamic parameter controls from strategy metadata."""
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
            values[key] = st.selectbox(
                label, options, index=idx, key=widget_key, help=help_text,
            )
        elif param_type == "slider":
            values[key] = st.slider(
                label,
                min_value=float(desc.get("min", 0)),
                max_value=float(desc.get("max", 100)),
                value=float(default),
                step=float(desc.get("step", 1)),
                key=widget_key, help=help_text,
            )
        else:
            values[key] = st.number_input(
                label,
                value=float(default) if isinstance(default, (int, float)) else default,
                min_value=float(desc.get("min", 0)),
                max_value=float(desc.get("max", 1e9)),
                step=float(desc.get("step", 1)),
                key=widget_key, help=help_text,
            )

    return values


# ---------------------------------------------------------------------------
# Metrics display
# ---------------------------------------------------------------------------

def _render_metrics(result) -> None:
    """Render 6-column performance metrics in two tidy rows."""
    section_header("绩效指标")

    metric_row([
        {"label": "年化收益", "value": f"{result.annual_return:+.1%}"},
        {"label": "Sharpe", "value": f"{result.sharpe_ratio:.2f}"},
        {"label": "最大回撤", "value": f"{result.max_drawdown:.1%}"},
        {"label": "Calmar", "value": f"{result.calmar_ratio:.2f}"},
    ])
    metric_row([
        {"label": "胜率", "value": f"{result.win_rate:.0%}"},
        {"label": "总交易", "value": str(result.total_trades)},
        {"label": "盈利/亏损", "value": f"{result.winning_trades}/{result.losing_trades}"},
        {"label": "持仓天数", "value": f"{result.avg_holding_days:.0f}"},
    ])


# ---------------------------------------------------------------------------
# Equity curve chart
# ---------------------------------------------------------------------------

def _render_equity_chart(result, title: str = "净值曲线") -> None:
    """Render Plotly equity curve with drawdown subplot."""
    eq = result.equity_curve
    if eq.empty:
        st.warning("暂无权益数据")
        return

    equity = eq.set_index("date")["equity"]
    dd = compute_drawdown_series(equity)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=("净值曲线", "回撤"),
    )

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name=title,
            line=dict(color=CHART_COLORS[0], width=2),
        ),
        row=1, col=1,
    )

    # Initial capital reference
    fig.add_hline(
        y=result.initial_capital, line_dash="dash",
        line_color=NEUTRAL, opacity=0.5, row=1, col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="回撤",
            fill="tozeroy",
            line=dict(color=DANGER, width=1),
        ),
        row=2, col=1,
    )

    apply_chart_theme(fig)
    fig.update_layout(height=500, showlegend=False)
    fig.update_yaxes(title_text="资产 (元)", row=1, col=1)
    fig.update_yaxes(title_text="回撤", row=2, col=1, tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Trade table
# ---------------------------------------------------------------------------

def _render_trade_table(result) -> None:
    """Render sortable trade history."""
    trades = result.trades
    if not trades:
        info_banner("无已完成交易（策略可能全程持仓未卖出）", kind="info")
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


# ---------------------------------------------------------------------------
# Comparison view
# ---------------------------------------------------------------------------

def _render_comparison(results: list) -> None:
    """Render strategy comparison table and overlaid equity curves."""
    if not results:
        return

    section_header("策略对比")

    # --- Summary table ---
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

    # Highlight best
    best = max(results, key=lambda r: r.sharpe_ratio)
    info_banner(
        f"🏆 推荐策略: **{best.strategy_name}** — "
        f"年化 {best.annual_return:+.1%} | Sharpe {best.sharpe_ratio:.2f} | "
        f"最大回撤 {best.max_drawdown:.1%}",
        kind="success",
    )

    # --- Overlaid equity curves ---
    section_header("净值曲线对比")
    colors = make_equity_colors(len(results))
    fig = go.Figure()

    for i, r in enumerate(results):
        eq = r.equity_curve
        if eq.empty:
            continue
        equity = eq.set_index("date")["equity"]
        normalized = equity / equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=normalized.index, y=normalized.values,
                mode="lines", name=r.strategy_name,
                line=dict(color=colors[i % len(colors)], width=2),
            ),
        )

    fig.add_hline(y=1.0, line_dash="dash", line_color=NEUTRAL, opacity=0.5)
    apply_chart_theme(fig)
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )
    fig.update_yaxes(title_text="归一化净值 (起始=1)", tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_strategy_page(df: pd.DataFrame, info: dict) -> None:
    """Render the complete strategy backtesting page.

    Args:
        df: OHLCV DataFrame from fetch_etf_hist().
        info: Real-time quote dict from fetch_etf_info().
    """
    inject_css()

    if df.empty:
        st.warning("暂无历史数据，请先获取行情数据")
        return

    pe_value = info.get("pe_ttm") or info.get("pe_static")

    # --- Sidebar: strategy selection ---
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ 策略选择")
    strategy_names = [s.name for s in STRATEGIES]
    selected_name = st.sidebar.radio(
        "选择策略",
        strategy_names,
        index=2,
    )
    strategy = STRATEGY_MAP[selected_name]
    compare_mode = st.sidebar.checkbox("☐ 四策略对比", value=False)

    # --- Strategy description ---
    with st.expander(f"📖 {strategy.name} — 策略说明", expanded=False):
        st.write(strategy.description)

    # --- PE info for valuation strategies ---
    if selected_name in ("估值定投", "网格+定投"):
        if pe_value is None:
            info_banner(
                "当前 ETF 无 PE(TTM) 数据（腾讯财经不提供），"
                "估值定投策略将使用基准金额（1倍）执行。",
                kind="warning",
            )
        else:
            st.info(
                f"📊 当前 PE(TTM): **{pe_value:.1f}**  |  "
                f"⚠️ 基于当前PE快照，非历史PE分位，仅供参考。"
            )

    # --- Parameters ---
    st.sidebar.subheader("🎛️ 参数配置")
    if compare_mode:
        st.sidebar.caption("对比模式使用各策略默认参数")
        params = {}
    else:
        params = _render_param_form(strategy, prefix="strat")

    # --- Run ---
    if st.sidebar.button("▶️ 开始回测", type="primary", use_container_width=True):
        with st.spinner("正在运行回测..."):
            engine = BacktestEngine(
                initial_capital=100_000,
                broker=Broker(),
                risk_manager=RiskManager(),
            )

            if compare_mode:
                results = []
                for s in STRATEGIES:
                    sp = s.get_default_params()
                    r = engine.run(df.copy(), s, pe_value=pe_value, **sp)
                    results.append(r)
                _render_comparison(results)

            else:
                result = engine.run(df.copy(), strategy, pe_value=pe_value, **params)

                # Metrics
                st.divider()
                _render_metrics(result)

                # Charts & trades
                st.divider()
                tab1, tab2 = st.tabs(["📈 净值曲线", "📋 交易明细"])
                with tab1:
                    _render_equity_chart(result, title=strategy.name)
                with tab2:
                    st.caption(f"共 {result.total_trades} 笔交易")
                    _render_trade_table(result)

                # Risk warnings
                warnings = []
                if selected_name in ("网格交易", "网格+定投"):
                    grid_p = params if params else strategy.get_default_params()
                    step = grid_p.get("position_per_grid_pct", 0.08)
                    price = info.get("current_price", 10)
                    rm = RiskManager()
                    should, msg = rm.check_step_size(step * price, price)
                    if should:
                        warnings.append(msg)

                if warnings:
                    for w in warnings:
                        info_banner(w, kind="warning")

                # Summary footer
                st.caption(result.summary())

    else:
        info_banner("👈 在左侧配置参数后点击「开始回测」", kind="info")
