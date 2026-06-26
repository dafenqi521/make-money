"""Dashboard UI components for ETF data display.

Renders ETF overview metrics, interactive price charts, and
historical data tables using Streamlit and Plotly.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render_etf_overview(info: dict) -> None:
    """
    Render ETF overview metrics in a row of metric cards.

    Args:
        info: dict with keys: name, current_price, change_pct, volume.
    """
    name = info.get("name", "未知")
    price = info.get("current_price")
    change = info.get("change_pct")
    volume = info.get("volume")

    st.subheader(f"📊 {name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price_str = f"¥{price:.3f}" if price is not None else "N/A"
        st.metric(label="最新价", value=price_str)

    with col2:
        if change is not None:
            delta_str = f"{change:+.2f}%"
            delta_color = "normal" if change >= 0 else "inverse"
            st.metric(label="涨跌幅", value=f"{change:.2f}%", delta=delta_str)
        else:
            st.metric(label="涨跌幅", value="N/A")

    with col3:
        if volume is not None:
            vol_str = f"{volume/10000:.0f}万手" if volume >= 10000 else f"{volume:.0f}手"
            st.metric(label="成交量", value=vol_str)
        else:
            st.metric(label="成交量", value="N/A")

    with col4:
        st.metric(label="数据来源", value="AKShare")


def render_price_chart(df: pd.DataFrame) -> None:
    """
    Render interactive candlestick chart with volume subplot.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
    """
    if df.empty:
        st.warning("暂无数据可显示")
        return

    # Sort by date ascending for chart
    chart_df = df.sort_values("date", ascending=True).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("K线图", "成交量"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="价格",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1, col=1,
    )

    # Volume bars
    colors = [
        "#ef5350" if chart_df.iloc[i]["close"] >= chart_df.iloc[i]["open"]
        else "#26a69a"
        for i in range(len(chart_df))
    ]
    fig.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["volume"],
            name="成交量",
            marker_color=colors,
            opacity=0.5,
        ),
        row=2, col=1,
    )

    # Layout
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="价格 (元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    """
    Render sortable historical data table.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
    """
    if df.empty:
        st.warning("暂无数据可显示")
        return

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    # Round prices for display
    for col in ["open", "high", "low", "close"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    # Format volume
    if "volume" in display_df.columns:
        display_df["volume"] = display_df["volume"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    # Rename columns for display
    display_df = display_df.rename(columns={
        "date": "日期",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "volume": "成交量",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "日期": st.column_config.TextColumn("日期", width="small"),
            "开盘价": st.column_config.NumberColumn("开盘价", format="%.3f"),
            "最高价": st.column_config.NumberColumn("最高价", format="%.3f"),
            "最低价": st.column_config.NumberColumn("最低价", format="%.3f"),
            "收盘价": st.column_config.NumberColumn("收盘价", format="%.3f"),
            "成交量": st.column_config.TextColumn("成交量", width="medium"),
        },
    )


def render_no_data(symbol: str, error: str = None) -> None:
    """Render a friendly message when no data is available."""
    st.error(f"⚠️ 无法获取 ETF `{symbol}` 的数据")
    if error:
        st.caption(f"错误详情: {error}")
    st.info(
        "请检查代码是否正确。常见 ETF 代码示例：\n\n"
        "- `510300` — 沪深300ETF\n"
        "- `510050` — 上证50ETF\n"
        "- `510500` — 中证500ETF\n"
        "- `159915` — 创业板ETF\n"
        "- `588000` — 科创50ETF"
    )
