"""Dashboard UI components — ETF overview, bid/ask, candlestick chart, data table.

Retro-Futuristic terminal aesthetic. All visual constants from terminal_theme.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.ui.terminal_theme import (
    UP_COLOR,
    DOWN_COLOR,
    GREEN,
    MAGENTA,
    TEXT,
    TEXT_DIM,
    PANEL,
    CARD,
    BORDER,
    MA_COLORS,
    FONT_MONO,
    apply_terminal_chart,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 3) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _fmt_vol(val) -> str:
    if val is None:
        return "—"
    if val >= 1_0000_0000:
        return f"{val / 1_0000_0000:.1f}亿"
    if val >= 10000:
        return f"{val / 10000:.0f}万"
    return f"{int(val)}"


# ---------------------------------------------------------------------------
# ETF Overview (used when not handled inline by app.py)
# ---------------------------------------------------------------------------

def render_etf_overview(info: dict) -> None:
    """Minimal overview — the heavy lifting is in app.py inline HTML."""
    pass  # app.py handles this directly with inline HTML for terminal feel


# ---------------------------------------------------------------------------
# Bid / Ask depth panel
# ---------------------------------------------------------------------------

def render_bid_ask_panel(info: dict) -> None:
    """Render 5-level bid/ask depth table with phosphor coloring."""
    bid_prices = [info.get(f"bid{i}_price") for i in range(1, 6)]
    bid_vols = [info.get(f"bid{i}_volume") for i in range(1, 6)]
    ask_prices = [info.get(f"ask{i}_price") for i in range(1, 6)]
    ask_vols = [info.get(f"ask{i}_volume") for i in range(1, 6)]

    has_data = any(v is not None for v in bid_prices + ask_prices)
    if not has_data:
        st.info("NO ORDER BOOK DATA — MARKET CLOSED")
        return

    rows: list[dict] = []
    for i in range(4, -1, -1):
        rows.append({
            " ": f"ASK {i + 1}",
            "PRICE": _fmt(ask_prices[i]) if ask_prices[i] is not None else "—",
            "SIZE": _fmt_vol(ask_vols[i]) if ask_vols[i] is not None else "—",
            "_side": "sell",
        })
    rows.append({" ": "───────", "PRICE": "───────", "SIZE": "───────", "_side": "sep"})
    for i in range(5):
        rows.append({
            " ": f"BID {i + 1}",
            "PRICE": _fmt(bid_prices[i]) if bid_prices[i] is not None else "—",
            "SIZE": _fmt_vol(bid_vols[i]) if bid_vols[i] is not None else "—",
            "_side": "buy",
        })

    display_df = pd.DataFrame(rows)

    def _color_row(r):
        side = r.get("_side", "")
        if side == "sell":
            return [f"background-color: rgba(255,0,110,0.06); color: {MAGENTA}"] * len(r)
        elif side == "buy":
            return [f"background-color: rgba(0,255,65,0.06); color: {GREEN}"] * len(r)
        return [f"color: {TEXT_DIM}"] * len(r)

    styled = display_df.style.apply(_color_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Candlestick chart
# ---------------------------------------------------------------------------

def render_price_chart(df: pd.DataFrame) -> None:
    """Render candlestick chart with MA lines and volume — terminal dark theme."""
    if df.empty:
        st.warning("NO DATA")
        return

    chart_df = df.sort_values("date", ascending=True).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("PRICE", "VOLUME"),
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR,
            decreasing_fillcolor=DOWN_COLOR,
            showlegend=True,
        ),
        row=1, col=1,
    )

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

    colors = [
        UP_COLOR if chart_df.iloc[i]["close"] >= chart_df.iloc[i]["open"]
        else DOWN_COLOR
        for i in range(len(chart_df))
    ]
    fig.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["volume"],
            name="VOL",
            marker_color=colors,
            opacity=0.3,
            showlegend=False,
        ),
        row=2, col=1,
    )

    apply_terminal_chart(fig, height=550)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="DATE", row=2, col=1)
    fig.update_yaxes(title_text="PRICE (CNY)", row=1, col=1)
    fig.update_yaxes(title_text="VOLUME", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Historical data table
# ---------------------------------------------------------------------------

def render_data_table(df: pd.DataFrame) -> None:
    """Render historical data table."""
    if df.empty:
        st.warning("NO DATA")
        return

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

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

    col_names = {
        "date": "DATE", "open": "OPEN", "high": "HI", "low": "LO",
        "close": "CLOSE", "volume": "VOL", "change_pct": "CHG%",
        "amplitude": "AMP%", "ma5": "MA5", "ma10": "MA10", "ma20": "MA20",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in col_names.items() if k in display_df.columns}
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Error / empty state
# ---------------------------------------------------------------------------

def render_no_data(symbol: str, error: str | None = None) -> None:
    """Render friendly error state."""
    st.error(f"SYS.ERR: CANNOT FETCH `{symbol}`")
    if error:
        st.caption(f"TRACE: {error}")
    st.info(
        "VALID ETF CODES:\n\n"
        "`510300` — 沪深300ETF\n"
        "`510050` — 上证50ETF\n"
        "`510500` — 中证500ETF\n"
        "`159915` — 创业板ETF\n"
        "`588000` — 科创50ETF"
    )
