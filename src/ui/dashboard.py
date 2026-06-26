"""Dashboard UI — ETF overview, bid/ask, candlestick chart, data table."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.ui.terminal_theme import (
    UP_COLOR, DOWN_COLOR, NEUTRAL, DARK, BORDER,
    MA_COLORS, FONT, apply_chart_theme,
)


def _fmt(val, decimals: int = 3) -> str:
    if val is None: return "—"
    return f"{val:.{decimals}f}"


def _fmt_vol(val) -> str:
    if val is None: return "—"
    if val >= 1_0000_0000: return f"{val / 1_0000_0000:.1f}亿"
    if val >= 10000: return f"{val / 10000:.0f}万"
    return f"{int(val)}"


def _fmt_mcap(val) -> str:
    if val is None: return "—"
    if val >= 10000: return f"{val / 10000:.1f}万亿"
    return f"{val:.0f}亿"


def _fmt_pe(val) -> str:
    if val is None: return "—"
    if val <= 0: return "亏损"
    return f"{val:.1f}"


# ---------------------------------------------------------------------------
# ETF Overview
# ---------------------------------------------------------------------------

def render_etf_overview(info: dict) -> None:
    name = info.get("name", "未知")
    date = info.get("date") or ""
    time = info.get("time") or ""
    source = "腾讯财经" if info.get("pe_ttm") is not None else "新浪"

    st.subheader(name)
    st.caption(f"数据时间: {date} {time}  ·  数据源: {source}")

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
    pe_ttm = info.get("pe_ttm")
    pb = info.get("pb")
    mcap_yi = info.get("mcap_yi")
    turnover_pct = info.get("turnover_pct")

    # Row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("最新价", _fmt(price))
    with c2:
        st.metric("涨跌幅", f"{change_pct:+.2f}%" if change_pct else "—",
                  delta=f"{change:+.3f}" if change else None)
    with c3:
        st.metric("今开", _fmt(open_price))
    with c4:
        st.metric("昨收", _fmt(prev_close))
    with c5:
        st.metric("振幅", f"{amplitude:.2f}%" if amplitude else "—")

    # Row 2
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("最高", _fmt(high))
    with c2:
        st.metric("最低", _fmt(low))
    with c3:
        st.metric("成交量", _fmt_vol(volume))
    with c4:
        st.metric("成交额", f"{amount/1_0000:.0f}万" if amount and amount >= 10000 else (f"{amount:.0f}" if amount else "—"))

    # Row 3 — valuation
    has_val = any(v is not None for v in [pe_ttm, pb, mcap_yi, turnover_pct])
    if has_val:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("PE(TTM)", _fmt_pe(pe_ttm), help="滚动市盈率")
        with c2:
            st.metric("PB", _fmt(pb, 2) if pb else "—", help="市净率")
        with c3:
            st.metric("总市值", _fmt_mcap(mcap_yi))
        with c4:
            st.metric("换手率", f"{turnover_pct:.2f}%" if turnover_pct else "—")
        with c5:
            st.metric("数据源", source)


# ---------------------------------------------------------------------------
# Bid / Ask
# ---------------------------------------------------------------------------

def render_bid_ask_panel(info: dict) -> None:
    bid_prices = [info.get(f"bid{i}_price") for i in range(1, 6)]
    bid_vols = [info.get(f"bid{i}_volume") for i in range(1, 6)]
    ask_prices = [info.get(f"ask{i}_price") for i in range(1, 6)]
    ask_vols = [info.get(f"ask{i}_volume") for i in range(1, 6)]

    has_data = any(v is not None for v in bid_prices + ask_prices)
    if not has_data:
        st.info("暂无盘口数据（非交易时段可能不提供五档行情）")
        return

    st.subheader("五档盘口")

    rows = []
    for i in range(4, -1, -1):
        rows.append({"": f"卖{i+1}", "价格": _fmt(ask_prices[i]) if ask_prices[i] is not None else "—",
                      "成交量(手)": _fmt_vol(ask_vols[i]) if ask_vols[i] is not None else "—", "_side": "sell"})
    rows.append({"": "———", "价格": "———", "成交量(手)": "———", "_side": "sep"})
    for i in range(5):
        rows.append({"": f"买{i+1}", "价格": _fmt(bid_prices[i]) if bid_prices[i] is not None else "—",
                      "成交量(手)": _fmt_vol(bid_vols[i]) if bid_vols[i] is not None else "—", "_side": "buy"})

    display_df = pd.DataFrame(rows)

    def _color_row(r):
        side = r.get("_side", "")
        if side == "sell":
            return ["background-color: #fef2f2; color: #dc2626"] * len(r)
        elif side == "buy":
            return ["background-color: #f0fdf4; color: #16a34a"] * len(r)
        return [""] * len(r)

    st.dataframe(display_df.style.apply(_color_row, axis=1), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Candlestick chart
# ---------------------------------------------------------------------------

def render_price_chart(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("暂无数据")
        return

    chart_df = df.sort_values("date", ascending=True).copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
        subplot_titles=("K线图", "成交量"),
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"], open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"], close=chart_df["close"], name="K线",
            increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
            showlegend=True,
        ), row=1, col=1,
    )

    for col, (color, width) in [
        ("ma5", (MA_COLORS["ma5"], 1)),
        ("ma10", (MA_COLORS["ma10"], 1)),
        ("ma20", (MA_COLORS["ma20"], 1.5)),
    ]:
        if col in chart_df.columns:
            visible = chart_df[col].notna()
            if visible.any():
                fig.add_trace(go.Scatter(
                    x=chart_df.loc[visible, "date"], y=chart_df.loc[visible, col],
                    mode="lines", name=col.upper(), line=dict(color=color, width=width),
                ), row=1, col=1)

    colors = [
        UP_COLOR if chart_df.iloc[i]["close"] >= chart_df.iloc[i]["open"]
        else DOWN_COLOR
        for i in range(len(chart_df))
    ]
    fig.add_trace(go.Bar(
        x=chart_df["date"], y=chart_df["volume"], name="成交量",
        marker_color=colors, opacity=0.35, showlegend=False,
    ), row=2, col=1)

    apply_chart_theme(fig, height=550)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="价格 (元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

def render_data_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("暂无数据")
        return

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    for col in ["open", "high", "low", "close", "ma5", "ma10", "ma20"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: round(x, 4) if pd.notna(x) else None)

    if "change_pct" in display_df.columns:
        display_df["change_pct"] = display_df["change_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    if "amplitude" in display_df.columns:
        display_df["amplitude"] = display_df["amplitude"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    if "volume" in display_df.columns:
        display_df["volume"] = display_df["volume"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")

    col_names = {
        "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "change_pct": "涨跌幅",
        "amplitude": "振幅", "ma5": "MA5", "ma10": "MA10", "ma20": "MA20",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_names.items() if k in display_df.columns})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

def render_no_data(symbol: str, error: str | None = None) -> None:
    st.error(f"⚠️ 无法获取 ETF `{symbol}` 的数据")
    if error:
        st.caption(f"错误详情: {error}")
    st.info(
        "常见 ETF 代码示例：\n\n"
        "- `510300` — 沪深300ETF\n- `510050` — 上证50ETF\n"
        "- `510500` — 中证500ETF\n- `159915` — 创业板ETF\n"
        "- `588000` — 科创50ETF"
    )
