"""Dashboard UI — ETF overview, bid/ask, candlestick chart, data table."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.ui.terminal_theme import (
    UP_COLOR, DOWN_COLOR, NEUTRAL, DARK, BORDER,
    MA_COLORS, FONT, apply_chart_theme,
    BG_CARD, PRIMARY, SUCCESS, DANGER,
    _styler_apply,
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

    with st.container(border=True):
        # Header row: name + source badge
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st.subheader(name)
        with c_right:
            st.caption(f"数据源: {source}  ·  {date} {time}")

        # ── Big price display ──
        color = SUCCESS if (change_pct or 0) >= 0 else DANGER
        sign = "+" if (change_pct or 0) >= 0 else ""
        st.markdown(
            f'<div style="display:flex; align-items:baseline; gap:16px; margin:8px 0 16px 0;">'
            f'<span style="font-size:2.2rem; font-weight:700; color:{DARK};">{_fmt(price)}</span>'
            f'<span style="font-size:1.2rem; color:{color}; font-weight:600;">'
            f'{sign}{change_pct:.2f}%</span>'
            f'<span style="font-size:0.85rem; color:{color};">{change:+.3f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Metric rows ──
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("今开", _fmt(open_price))
        with c2: st.metric("昨收", _fmt(prev_close))
        with c3: st.metric("最高", _fmt(high))
        with c4: st.metric("最低", _fmt(low))
        with c5: st.metric("成交量", _fmt_vol(volume))
        with c6: st.metric("振幅", f"{amplitude:.2f}%" if amplitude else "—")

        # ── Valuation row ──
        has_val = any(v is not None for v in [pe_ttm, pb, mcap_yi, turnover_pct])
        if has_val:
            st.divider()
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("PE(TTM)", _fmt_pe(pe_ttm), help="滚动市盈率")
            with c2: st.metric("PB", _fmt(pb, 2) if pb else "—", help="市净率")
            with c3: st.metric("总市值", _fmt_mcap(mcap_yi))
            with c4: st.metric("换手率", f"{turnover_pct:.2f}%" if turnover_pct else "—")
            with c5: st.metric("成交额", f"{amount/1_0000:.0f}万" if amount and amount >= 10000 else (f"{amount:.0f}" if amount else "—"))


# ---------------------------------------------------------------------------
# Bid / Ask — visual depth bars
# ---------------------------------------------------------------------------

def render_bid_ask_panel(info: dict) -> None:
    bid_prices = [info.get(f"bid{i}_price") for i in range(1, 6)]
    bid_vols = [info.get(f"bid{i}_volume") for i in range(1, 6)]
    ask_prices = [info.get(f"ask{i}_price") for i in range(1, 6)]
    ask_vols = [info.get(f"ask{i}_volume") for i in range(1, 6)]

    has_data = any(v is not None for v in bid_prices + ask_prices)
    if not has_data:
        with st.container(border=True):
            st.caption("暂无盘口数据（非交易时段）")
        return

    # Find max volume for bar scaling
    all_vols = [v for v in bid_vols + ask_vols if v is not None]
    max_vol = max(all_vols) if all_vols else 1

    with st.container(border=True):
        st.caption("五档盘口")

        # ── Ask side (sell pressure) — red tones ──
        for i in range(4, -1, -1):
            p = ask_prices[i]
            v = ask_vols[i]
            if p is None and v is None:
                continue
            bar_pct = (v / max_vol * 100) if v and max_vol > 0 else 0
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; height:22px; '
                f'margin:1px 0;">'
                f'<span style="width:36px; font-size:0.7rem; color:{NEUTRAL};">卖{i+1}</span>'
                f'<span style="width:64px; font-size:0.8rem; color:{DANGER}; '
                f'font-weight:600; text-align:right;">{_fmt(p)}</span>'
                f'<span style="flex:1; height:16px; background:linear-gradient(90deg, '
                f'rgba(220,38,38,0.20) {bar_pct}%, transparent {bar_pct}%); '
                f'border-radius:2px;"></span>'
                f'<span style="width:64px; font-size:0.7rem; color:{NEUTRAL}; '
                f'text-align:right;">{_fmt_vol(v)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Divider ──
        st.markdown(
            f'<div style="border-top:1px dashed {BORDER}; margin:6px 0;"></div>',
            unsafe_allow_html=True,
        )

        # ── Bid side (buy support) — green tones ──
        for i in range(5):
            p = bid_prices[i]
            v = bid_vols[i]
            if p is None and v is None:
                continue
            bar_pct = (v / max_vol * 100) if v and max_vol > 0 else 0
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; height:22px; '
                f'margin:1px 0;">'
                f'<span style="width:36px; font-size:0.7rem; color:{NEUTRAL};">买{i+1}</span>'
                f'<span style="width:64px; font-size:0.8rem; color:{SUCCESS}; '
                f'font-weight:600; text-align:right;">{_fmt(p)}</span>'
                f'<span style="flex:1; height:16px; background:linear-gradient(90deg, '
                f'rgba(22,163,74,0.20) {bar_pct}%, transparent {bar_pct}%); '
                f'border-radius:2px;"></span>'
                f'<span style="width:64px; font-size:0.7rem; color:{NEUTRAL}; '
                f'text-align:right;">{_fmt_vol(v)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Candlestick chart
# ---------------------------------------------------------------------------

def render_price_chart(
    df: pd.DataFrame,
    markers: pd.DataFrame | None = None,
) -> None:
    """Render OHLC candlestick chart with MA overlays and volume subplot.

    Args:
        df: OHLCV DataFrame (will be sorted date-ascending internally).
        markers: Optional DataFrame with columns [date, close, signal, signal_reason]
                 for overlaying buy/sell markers on the chart.
    """
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

    # ── Signal markers overlay ──
    if markers is not None and not markers.empty:
        _overlay_signal_markers(fig, markers)

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


def _overlay_signal_markers(fig, markers: pd.DataFrame) -> None:
    """Add buy (green ▲) and sell (red ▼) markers to the candlestick subplot."""
    buy_df = markers[markers["signal"] == "buy"].copy()
    sell_df = markers[markers["signal"] == "sell"].copy()

    if not buy_df.empty:
        buy_df["hover"] = buy_df.apply(
            lambda r: f"买入<br>{r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])}<br>"
            + (f"{r.get('signal_reason', '')}" if pd.notna(r.get("signal_reason")) else ""),
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=buy_df["date"], y=buy_df["close"],
                mode="markers", name="买入信号",
                marker=dict(symbol="triangle-up", size=10, color=SUCCESS,
                            line=dict(width=1, color="white")),
                text=buy_df["hover"], hoverinfo="text",
                showlegend=True,
            ), row=1, col=1,
        )

    if not sell_df.empty:
        sell_df["hover"] = sell_df.apply(
            lambda r: f"卖出<br>{r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])}<br>"
            + (f"{r.get('signal_reason', '')}" if pd.notna(r.get("signal_reason")) else ""),
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sell_df["date"], y=sell_df["close"],
                mode="markers", name="卖出信号",
                marker=dict(symbol="triangle-down", size=10, color=DANGER,
                            line=dict(width=1, color="white")),
                text=sell_df["hover"], hoverinfo="text",
                showlegend=True,
            ), row=1, col=1,
        )


# ---------------------------------------------------------------------------
# Data table — visual hierarchy
# ---------------------------------------------------------------------------

def render_data_table(df: pd.DataFrame) -> None:
    """Historical data with intentional visual hierarchy.

    Tier 1 (primary):   收盘价 (bold), 涨跌幅 (color-coded)
    Tier 2 (secondary): 日期, 开盘, 最高, 最低
    Tier 3 (reference): MA5/10/20, 成交量 (bar), 振幅 (muted)
    """
    if df.empty:
        st.warning("暂无数据")
        return

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    for col in ["open", "high", "low", "close", "ma5", "ma10", "ma20"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: round(x, 4) if pd.notna(x) else None)

    if "change_pct" in display_df.columns:
        display_df["change_pct"] = display_df["change_pct"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    if "amplitude" in display_df.columns:
        display_df["amplitude"] = display_df["amplitude"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    if "volume" in display_df.columns:
        display_df["volume"] = display_df["volume"].apply(lambda x: int(x) if pd.notna(x) else 0)

    col_names = {
        "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "change_pct": "涨跌幅",
        "amplitude": "振幅", "ma5": "MA5", "ma10": "MA10", "ma20": "MA20",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_names.items() if k in display_df.columns})

    vol_max = int(display_df["成交量"].max()) if "成交量" in display_df.columns else 1

    # Column order
    ordered_cols = []
    for c in ["日期", "收盘", "涨跌幅", "开盘", "最高", "最低", "成交量", "振幅", "MA5", "MA10", "MA20"]:
        if c in display_df.columns:
            ordered_cols.append(c)
    display_df = display_df[ordered_cols]

    # Column config
    column_config = {}
    if "日期" in display_df.columns:
        column_config["日期"] = st.column_config.TextColumn("日期", width="small")
    for col in ["开盘", "最高", "最低"]:
        if col in display_df.columns:
            column_config[col] = st.column_config.NumberColumn(col, format="%.3f", width="small")
    if "收盘" in display_df.columns:
        column_config["收盘"] = st.column_config.NumberColumn("★ 收盘", format="%.3f", width="small")
    if "涨跌幅" in display_df.columns:
        column_config["涨跌幅"] = st.column_config.NumberColumn("涨跌幅", format="%+.2f%%", width="small")
    if "振幅" in display_df.columns:
        column_config["振幅"] = st.column_config.NumberColumn("振幅", format="%.2f%%", width="small")
    if "成交量" in display_df.columns:
        column_config["成交量"] = st.column_config.ProgressColumn(
            "成交量", format="%d", width="medium", min_value=0, max_value=vol_max,
        )
    for col in ["MA5", "MA10", "MA20"]:
        if col in display_df.columns:
            column_config[col] = st.column_config.NumberColumn(col, format="%.3f", width="small")

    # Row styles — use applymap (pandas 2.0 compatible)
    def _style_change(val):
        if val is None or pd.isna(val): return ""
        if val > 0:
            return "color: #16a34a; font-weight: 700; background-color: #f0fdf4"
        elif val < 0:
            return "color: #dc2626; font-weight: 700; background-color: #fef2f2"
        return ""

    def _style_close(val):
        if val is None or pd.isna(val): return ""
        return "font-weight: 700"

    styled = display_df.style
    if "涨跌幅" in display_df.columns:
        styled = _styler_apply(styled, _style_change, ["涨跌幅"])
    if "收盘" in display_df.columns:
        styled = _styler_apply(styled, _style_close, ["收盘"])

    st.dataframe(
        styled, use_container_width=True, hide_index=True,
        column_config=column_config,
    )


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
