"""Streamlit result view for the single ETF-rotation backtest."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.engine.backtest import BacktestResult
from src.ui.terminal_theme import PRIMARY, apply_chart_theme


_BENCHMARK_COLOR = "#475569"
_NEGATIVE_COLOR = "#c26a3d"
_NEUTRAL_COLOR = "#f5f5f4"


def render_backtest_result(result: BacktestResult) -> None:
    """Render summary-first metrics, trends, diagnostics, and details."""

    curve = result.equity_curve
    if curve.empty:
        st.error("回测区间内没有足够的共同交易日，未生成结果。")
        return

    strategy = result.metrics
    benchmark = result.benchmark_metrics
    excess = float(strategy["total_return"]) - float(benchmark["total_return"])
    metric_columns = st.columns(7)
    metric_columns[0].metric("累计收益", f"{strategy['total_return']:+.2%}")
    metric_columns[1].metric("年化收益", f"{strategy['annual_return']:+.2%}")
    metric_columns[2].metric("最大回撤", f"{strategy['max_drawdown']:.2%}")
    metric_columns[3].metric("夏普比率", f"{strategy['sharpe_ratio']:.2f}")
    metric_columns[4].metric(
        f"基准 {result.benchmark_code}", f"{benchmark['total_return']:+.2%}"
    )
    metric_columns[5].metric("累计超额", f"{excess:+.2%}")
    metric_columns[6].metric("年化换手", f"{strategy['annual_turnover']:.1f}x")

    start = pd.Timestamp(curve["date"].min()).date().isoformat()
    end = pd.Timestamp(curve["date"].max()).date().isoformat()
    frozen_signals = (
        int(result.signal_log["frozen"].sum())
        if not result.signal_log.empty and "frozen" in result.signal_log
        else 0
    )
    st.caption(
        f"区间 {start} 至 {end} · {len(curve)} 个交易日 · "
        f"行情覆盖率 {result.coverage:.0%} · {strategy['trade_count']} 笔成交 · "
        f"平均资金暴露 {strategy['average_exposure']:.1%} · "
        f"数据门禁冻结 {frozen_signals} 次"
    )
    if result.coverage < 0.80:
        st.error("历史行情覆盖率低于80%，数据质量门禁会冻结相应调仓；结果不宜用于决策。")
    if result.data_errors:
        with st.expander(f"历史数据提示（{len(result.data_errors)}项）"):
            for code, message in result.data_errors.items():
                st.write(f"- `{code}`：{message}")
    if frozen_signals:
        st.warning(f"回测期间有 {frozen_signals} 个信号日因行情覆盖率或新鲜度不足而冻结调仓。")

    st.subheader("策略与基准净值")
    if len(curve) >= 8:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["equity"],
                mode="lines",
                name="ETF轮动策略",
                line=dict(color=PRIMARY, width=2.2),
                hovertemplate="%{x|%Y-%m-%d}<br>策略 ¥%{y:,.2f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["benchmark_equity"],
                mode="lines",
                name=f"基准 {result.benchmark_code}",
                line=dict(color=_BENCHMARK_COLOR, width=1.8, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>基准 ¥%{y:,.2f}<extra></extra>",
            )
        )
        apply_chart_theme(figure, height=360)
        figure.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        figure.update_xaxes(title="交易日")
        figure.update_yaxes(title="账户资产（元）")
        st.plotly_chart(figure, width="stretch")
        st.caption("信号使用当日收盘数据，模拟成交发生在下一交易日开盘并计入佣金与滑点。")
    else:
        st.info("少于8个交易日，不展示可能产生误导的趋势图。")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("回撤路径")
        drawdown_figure = go.Figure()
        drawdown_figure.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["drawdown"],
                mode="lines",
                name="策略回撤",
                line=dict(color=PRIMARY, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>策略 %{y:.2%}<extra></extra>",
            )
        )
        drawdown_figure.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["benchmark_drawdown"],
                mode="lines",
                name="基准回撤",
                line=dict(color=_BENCHMARK_COLOR, width=1.6, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>基准 %{y:.2%}<extra></extra>",
            )
        )
        apply_chart_theme(drawdown_figure, height=320)
        drawdown_figure.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        drawdown_figure.update_xaxes(title="交易日")
        drawdown_figure.update_yaxes(title="相对历史高点", tickformat=".0%")
        st.plotly_chart(drawdown_figure, width="stretch")

    with chart_right:
        st.subheader("月度收益")
        monthly = result.monthly_returns.copy()
        if len(monthly) >= 4:
            labels = monthly["month"].dt.strftime("%Y-%m")
            values = [
                monthly["strategy_return"].to_numpy(),
                monthly["benchmark_return"].to_numpy(),
            ]
            bound = max(0.01, float(monthly[["strategy_return", "benchmark_return"]].abs().max().max()))
            heatmap = go.Figure(
                go.Heatmap(
                    z=values,
                    x=labels,
                    y=["ETF轮动策略", f"基准 {result.benchmark_code}"],
                    zmin=-bound,
                    zmax=bound,
                    zmid=0,
                    colorscale=[
                        [0.0, _NEGATIVE_COLOR],
                        [0.5, _NEUTRAL_COLOR],
                        [1.0, PRIMARY],
                    ],
                    text=[[f"{value:+.1%}" for value in row] for row in values],
                    texttemplate="%{text}",
                    hovertemplate="%{y}<br>%{x}<br>%{z:+.2%}<extra></extra>",
                    colorbar=dict(title="收益率", tickformat=".0%"),
                )
            )
            apply_chart_theme(heatmap, height=320)
            heatmap.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            heatmap.update_xaxes(title="月份")
            st.plotly_chart(heatmap, width="stretch")
        else:
            st.info("累计到4个完整月份后展示月度收益热力图。")

    detail_metrics = pd.DataFrame(
        [
            {
                "指标": "年化波动率",
                "策略": strategy["annual_volatility"],
                "基准": benchmark["annual_volatility"],
            },
            {
                "指标": "最大回撤",
                "策略": strategy["max_drawdown"],
                "基准": benchmark["max_drawdown"],
            },
            {
                "指标": "夏普比率",
                "策略": strategy["sharpe_ratio"],
                "基准": benchmark["sharpe_ratio"],
            },
            {
                "指标": "卡玛比率",
                "策略": strategy["calmar_ratio"],
                "基准": benchmark["calmar_ratio"],
            },
        ]
    )
    with st.expander("完整指标和交易明细"):
        st.dataframe(detail_metrics, hide_index=True, width="stretch")
        if result.trades.empty:
            st.info("回测期间没有成交。")
        else:
            trade_table = result.trades.copy().sort_values("date", ascending=False)
            trade_table["action"] = trade_table["action"].map(
                {"buy": "买入", "sell": "卖出"}
            )
            trade_table = trade_table.rename(
                columns={
                    "date": "日期",
                    "code": "代码",
                    "name": "名称",
                    "action": "操作",
                    "price": "成交价",
                    "shares": "份额",
                    "amount": "成交额",
                    "commission": "佣金",
                    "pnl": "卖出盈亏",
                    "pnl_pct": "卖出收益率",
                    "reason": "原因",
                }
            )
            st.dataframe(
                trade_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "日期": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "成交价": st.column_config.NumberColumn(format="%.4f"),
                    "成交额": st.column_config.NumberColumn(format="¥%.2f"),
                    "佣金": st.column_config.NumberColumn(format="¥%.2f"),
                    "卖出盈亏": st.column_config.NumberColumn(format="¥%+.2f"),
                    "卖出收益率": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )


def render_parameter_sweep(sweep: pd.DataFrame) -> None:
    if sweep.empty:
        return
    st.subheader("持仓数量稳健性")
    display = sweep.copy()
    display["最大回撤绝对值"] = display["max_drawdown"].abs()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=display["max_positions"].astype(str),
            y=display["total_return"],
            name="累计收益",
            marker_color=PRIMARY,
            hovertemplate="最多%{x}只<br>收益 %{y:+.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=display["max_positions"].astype(str),
            y=display["最大回撤绝对值"],
            name="最大回撤绝对值",
            marker_color=_NEGATIVE_COLOR,
            hovertemplate="最多%{x}只<br>回撤 %{y:.2%}<extra></extra>",
        )
    )
    apply_chart_theme(figure, height=320)
    figure.update_layout(
        barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    figure.update_xaxes(title="最多持有ETF数量")
    figure.update_yaxes(title="收益率 / 回撤绝对值", tickformat=".0%", rangemode="tozero")
    st.plotly_chart(figure, width="stretch")
    st.caption("稳健策略不应只在单一持仓数量下有效；该结果用于发现参数敏感性，不用于挑选历史最优值。")
    st.dataframe(
        sweep,
        hide_index=True,
        width="stretch",
        column_config={
            "max_positions": "最多持有",
            "total_return": st.column_config.NumberColumn("累计收益", format="%+.2f%%"),
            "annual_return": st.column_config.NumberColumn("年化收益", format="%+.2f%%"),
            "max_drawdown": st.column_config.NumberColumn("最大回撤", format="%.2f%%"),
            "sharpe_ratio": st.column_config.NumberColumn("夏普", format="%.2f"),
            "trade_count": "成交笔数",
            "annual_turnover": st.column_config.NumberColumn("年化换手", format="%.1fx"),
        },
    )
