"""Dashboard UI components for ETF data display.

Renders ETF overview metrics, bid/ask depth panel, interactive price
charts with moving averages, and historical data tables.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# ── Colour palette ─────────────────────────────────────────────
RED = "#ef5350"
GREEN = "#26a69a"


def _fmt(val, decimals=3):
    """Format a float-or-None value for display."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _fmt_vol(val):
    """Format volume as human-readable string."""
    if val is None:
        return "—"
    if val >= 1_0000_0000:
        return f"{val/1_0000_0000:.1f}亿"
    if val >= 10000:
        return f"{val/10000:.0f}万"
    return f"{int(val)}"


def render_etf_overview(info: dict) -> None:
    """Render ETF overview: name, date, price metrics, volume/amount."""
    name = info.get("name", "未知")
    date = info.get("date") or ""
    time = info.get("time") or ""

    st.subheader(f"📊 {name}")
    st.caption(f"数据时间: {date} {time}".strip())

    price = info.get("current_price")
    change = info.get("change")
    change_pct = info.get("change_pct")
    amplitude = info.get("amplitude")
    prev_close = info.get("prev_close")
    open_price = info.get("open")
    high = info.get("high")
    low = info.get("low")
    volume = info.get("volume")
    amount = info.get("amount")

    # Row 1: Price core
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("最新价", _fmt(price))
    with c2:
        if change is not None and change_pct is not None:
            st.metric(
                "涨跌额 / 涨跌幅",
                f"{change:+.3f}",
                delta=f"{change_pct:+.2f}%",
            )
        else:
            st.metric("涨跌额 / 涨跌幅", "—")
    with c3:
        st.metric("昨收", _fmt(prev_close))
    with c4:
        st.metric("今开", _fmt(open_price))
    with c5:
        st.metric("振幅", _fmt(amplitude, 2) + "%" if amplitude else "—")

    # Row 2: Hi / Lo / Volume / Amount
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("最高", _fmt(high))
    with c2:
        st.metric("最低", _fmt(low))
    with c3:
        st.metric("成交量", _fmt_vol(volume))
    with c4:
        if amount is not None:
            st.metric("成交额", f"{amount/1_0000:.0f}万" if amount >= 10000 else f"{amount:.0f}")
        else:
            st.metric("成交额", "—")


def render_bid_ask_panel(info: dict) -> None:
    """Render 5-level bid/ask (盘口) table."""
    bid_prices = [info.get(f"bid{i}_price") for i in range(1, 6)]
    bid_vols = [info.get(f"bid{i}_volume") for i in range(1, 6)]
    ask_prices = [info.get(f"ask{i}_price") for i in range(1, 6)]
    ask_vols = [info.get(f"ask{i}_volume") for i in range(1, 6)]

    # Check if we have any bid/ask data
    has_data = any(
        v is not None
        for v in bid_prices + ask_prices
    )

    if not has_data:
        st.info("暂无盘口数据（非交易时段可能不提供五档行情）")
        return

    st.subheader("📋 五档盘口")

    # Build rows: 卖5 → 卖1, ------, 买1 → 买5
    rows = []
    for i in range(4, -1, -1):  # ask descending (卖5..卖1)
        rows.append({
            "档位": f"卖{i+1}",
            "价格": ask_prices[i],
            "手数": ask_vols[i],
            "side": "sell",
        })
    rows.append({"档位": "———", "价格": None, "手数": None, "side": "sep"})
    for i in range(5):  # bid ascending display (买1..买5)
        rows.append({
            "档位": f"买{i+1}",
            "价格": bid_prices[i],
            "手数": bid_vols[i],
            "side": "buy",
        })

    # Convert to DataFrame for st.dataframe
    table_data = []
    for r in rows:
        if r["side"] == "sep":
            table_data.append({"": "─────", "价格": "─────", "成交量(手)": "─────"})
            continue
        price_str = _fmt(r["价格"]) if r["价格"] is not None else "—"
        vol_str = _fmt_vol(r["手数"]) if r["手数"] is not None else "—"
        label = f"🔴 {r['档位']}" if r["side"] == "sell" else f"🟢 {r['档位']}"
        table_data.append({"": label, "价格": price_str, "成交量(手)": vol_str})

    display_df = pd.DataFrame(table_data)

    # Colour rows: sell = light red, buy = light green
    def _row_style(row):
        label = row[""]
        if label.startswith("🔴"):
            return ["background-color: #fff5f5"] * len(row)
        elif label.startswith("🟢"):
            return ["background-color: #f5fff5"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(_row_style, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )


def render_price_chart(df: pd.DataFrame) -> None:
    """Render candlestick chart with MA lines and volume subplot."""
    if df.empty:
        st.warning("暂无数据可显示")
        return

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
            name="K线",
            increasing_line_color=RED,
            decreasing_line_color=GREEN,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Moving averages
    for col, color, lw in [("ma5", "#FF9800", 1), ("ma10", "#2196F3", 1), ("ma20", "#9C27B0", 1.5)]:
        if col in chart_df.columns:
            visible = chart_df[col].notna()
            if visible.any():
                fig.add_trace(
                    go.Scatter(
                        x=chart_df.loc[visible, "date"],
                        y=chart_df.loc[visible, col],
                        mode="lines",
                        name=col.upper(),
                        line=dict(color=color, width=lw),
                    ),
                    row=1, col=1,
                )

    # Volume bars
    colors = [
        RED if chart_df.iloc[i]["close"] >= chart_df.iloc[i]["open"]
        else GREEN
        for i in range(len(chart_df))
    ]
    fig.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["volume"],
            name="成交量",
            marker_color=colors,
            opacity=0.4,
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
    """Render sortable historical data table with enriched columns."""
    if df.empty:
        st.warning("暂无数据可显示")
        return

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    # Round numeric columns
    for col in ["open", "high", "low", "close", "ma5", "ma10", "ma20"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: round(x, 4) if pd.notna(x) else None
            )

    if "change_pct" in display_df.columns:
        display_df["change_pct"] = display_df["change_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else ""
        )
    if "amplitude" in display_df.columns:
        display_df["amplitude"] = display_df["amplitude"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
    if "volume" in display_df.columns:
        display_df["volume"] = display_df["volume"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    # Rename for display
    col_names = {
        "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "change_pct": "涨跌幅",
        "amplitude": "振幅", "ma5": "MA5", "ma10": "MA10", "ma20": "MA20",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in col_names.items() if k in display_df.columns}
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def render_no_data(symbol: str, error: str = None) -> None:
    """Render a friendly error when data cannot be fetched."""
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
