"""Dashboard UI components — ETF overview, bid/ask, candlestick chart, data table.

All visual constants come from ``src.ui.theme`` — no hardcoded colors.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.ui.theme import (
    UP_COLOR,
    DOWN_COLOR,
    UP_BG,
    DOWN_BG,
    NEUTRAL,
    DARK,
    MA_COLORS,
    apply_chart_theme,
    chart_layout,
    section_header,
    metric_row,
    info_banner,
    inject_css,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 3) -> str:
    """Format a float-or-None value for display."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _fmt_vol(val) -> str:
    """Format volume as human-readable Chinese units."""
    if val is None:
        return "—"
    if val >= 1_0000_0000:
        return f"{val / 1_0000_0000:.1f}亿"
    if val >= 10000:
        return f"{val / 10000:.0f}万"
    return f"{int(val)}"


def _fmt_mcap(val) -> str:
    """Format market cap (in 亿) as human-readable string."""
    if val is None:
        return "—"
    if val >= 10000:
        return f"{val / 10000:.1f}万亿"
    return f"{val:.0f}亿"


def _fmt_pe(val) -> str:
    """Format PE — show '亏损' for negative values."""
    if val is None:
        return "—"
    if val <= 0:
        return "亏损"
    return f"{val:.1f}"


# ---------------------------------------------------------------------------
# ETF Overview
# ---------------------------------------------------------------------------

def render_etf_overview(info: dict) -> None:
    """Render ETF overview with price, volume, and valuation metrics.

    Three uniform 4-column rows — no more 5/4/5 chaos.
    """
    inject_css()

    name = info.get("name", "未知")
    date = info.get("date") or ""
    time = info.get("time") or ""
    source = "腾讯财经" if info.get("pe_ttm") is not None else "新浪"

    section_header(
        name,
        subtitle=f"数据时间: {date} {time}  ·  数据源: {source}",
    )

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

    # Row 1 — Price core
    metric_row([
        {"label": "最新价", "value": _fmt(price)},
        {
            "label": "涨跌幅",
            "value": f"{change_pct:+.2f}%" if change_pct else "—",
            "delta": f"{change:+.3f}" if change else None,
        },
        {"label": "今开", "value": _fmt(open_price)},
        {"label": "昨收", "value": _fmt(prev_close)},
    ])

    # Row 2 — Range & volume
    amount_str = (
        f"{amount / 1_0000:.0f}万" if amount and amount >= 10000
        else f"{amount:.0f}" if amount
        else "—"
    )
    metric_row([
        {"label": "最高", "value": _fmt(high)},
        {"label": "最低", "value": _fmt(low)},
        {"label": "成交量", "value": _fmt_vol(volume)},
        {"label": "成交额", "value": amount_str},
    ])

    # Row 3 — Valuation (Tencent only)
    amp_str = f"{amplitude:.2f}%" if amplitude else "—"
    has_val = any(v is not None for v in [pe_ttm, pb, mcap_yi, turnover_pct])
    if has_val:
        metric_row([
            {"label": "PE(TTM)", "value": _fmt_pe(pe_ttm),
             "help": "滚动市盈率，越低越低估"},
            {"label": "PB", "value": _fmt(pb, 2) if pb else "—",
             "help": "市净率，<1 为破净"},
            {"label": "总市值", "value": _fmt_mcap(mcap_yi)},
            {"label": "换手率", "value": f"{turnover_pct:.2f}%" if turnover_pct else "—",
             "help": "日换手率，反映交易活跃度"},
        ])
    else:
        metric_row([
            {"label": "振幅", "value": amp_str},
            {"label": "PE/PB", "value": "—",
             "help": "当前数据源不提供估值数据"},
            {"label": "", "value": ""},
            {"label": "", "value": ""},
        ])


# ---------------------------------------------------------------------------
# Bid / Ask depth panel
# ---------------------------------------------------------------------------

def render_bid_ask_panel(info: dict) -> None:
    """Render 5-level bid/ask depth table with colour-coded rows."""
    bid_prices = [info.get(f"bid{i}_price") for i in range(1, 6)]
    bid_vols = [info.get(f"bid{i}_volume") for i in range(1, 6)]
    ask_prices = [info.get(f"ask{i}_price") for i in range(1, 6)]
    ask_vols = [info.get(f"ask{i}_volume") for i in range(1, 6)]

    has_data = any(v is not None for v in bid_prices + ask_prices)
    if not has_data:
        info_banner("暂无盘口数据（非交易时段可能不提供五档行情）", kind="info")
        return

    section_header("五档盘口")

    # Build display rows: 卖5→卖1, 分离线, 买1→买5
    rows: list[dict] = []
    # Asks — descending
    for i in range(4, -1, -1):
        rows.append({
            " ": f"卖{i + 1}",
            "价格": _fmt(ask_prices[i]) if ask_prices[i] is not None else "—",
            "成交量(手)": _fmt_vol(ask_vols[i]) if ask_vols[i] is not None else "—",
            "_side": "sell",
        })
    rows.append({" ": "———", "价格": "———", "成交量(手)": "———", "_side": "sep"})
    # Bids — ascending
    for i in range(5):
        rows.append({
            " ": f"买{i + 1}",
            "价格": _fmt(bid_prices[i]) if bid_prices[i] is not None else "—",
            "成交量(手)": _fmt_vol(bid_vols[i]) if bid_vols[i] is not None else "—",
            "_side": "buy",
        })

    display_df = pd.DataFrame(rows)

    # Vectorized row styling — red bg for sells, green bg for buys
    def _color_row(r):
        side = r.get("_side", "")
        if side == "sell":
            return [f"background-color: {DOWN_BG}"] * len(r)
        elif side == "buy":
            return [f"background-color: {UP_BG}"] * len(r)
        return [""] * len(r)

    styled = display_df.style.apply(_color_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Candlestick chart with MA lines
# ---------------------------------------------------------------------------

def render_price_chart(df: pd.DataFrame) -> None:
    """Render interactive candlestick chart with MA5/10/20 and volume subplot."""
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

    # --- Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="K线",
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DOWN_COLOR,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # --- Moving averages ---
    for col, (color, width) in [
        ("ma5", (MA_COLORS["ma5"], 1)),
        ("ma10", (MA_COLORS["ma10"], 1)),
        ("ma20", (MA_COLORS["ma20"], 1.5)),
    ]:
        if col in chart_df.columns:
            visible = chart_df[col].notna()
            if visible.any():
                fig.add_trace(
                    go.Scatter(
                        x=chart_df.loc[visible, "date"],
                        y=chart_df.loc[visible, col],
                        mode="lines",
                        name=col.upper(),
                        line=dict(color=color, width=width),
                    ),
                    row=1, col=1,
                )

    # --- Volume bars ---
    colors = [
        UP_COLOR if chart_df.iloc[i]["close"] >= chart_df.iloc[i]["open"]
        else DOWN_COLOR
        for i in range(len(chart_df))
    ]
    fig.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["volume"],
            name="成交量",
            marker_color=colors,
            opacity=0.35,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Apply theme ---
    apply_chart_theme(fig)
    fig.update_layout(height=600)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="价格 (元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Historical data table
# ---------------------------------------------------------------------------

def render_data_table(df: pd.DataFrame) -> None:
    """Render sortable historical data table with formatted columns."""
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

    # Chinese column names
    col_names = {
        "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "change_pct": "涨跌幅",
        "amplitude": "振幅", "ma5": "MA5", "ma10": "MA10", "ma20": "MA20",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in col_names.items() if k in display_df.columns}
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Error / empty state
# ---------------------------------------------------------------------------

def render_no_data(symbol: str, error: str | None = None) -> None:
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
