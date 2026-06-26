"""ETF Investment Decision System

Retro-Futuristic trading terminal.  Streamlit + Plotly + phosphor.
"""

import streamlit as st
from src.data.fetcher import fetch_etf_hist, fetch_etf_info, fetch_multi_etf_info
from src.ui.dashboard import (
    render_etf_overview,
    render_bid_ask_panel,
    render_price_chart,
    render_data_table,
    render_no_data,
)
from src.ui.strategy_ui import render_strategy_page
from src.ui.terminal_theme import (
    inject_terminal_css,
    ticker_header,
    section_header,
    price_display,
    status_badge,
    GREEN, MAGENTA, TEXT, TEXT_DIM, PANEL, CARD, BORDER, CYAN,
    FONT_MONO,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ETF 投资决策系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject global terminal CSS ───────────────────────────────
inject_terminal_css()

# ── Top Bar (pure HTML) ──────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding:8px 20px; background:{PANEL}; border-bottom:1px solid {BORDER};">
    <div style="display:flex; align-items:center; gap:16px;">
        <span style="font-size:1.1rem; font-weight:700; color:{GREEN};
                     font-family:{FONT_MONO}; letter-spacing:1px;">
            ▸ MAKE_MONEY
        </span>
        <span style="color:{TEXT_DIM}; font-size:0.65rem; font-family:{FONT_MONO};">
            ETF INVESTMENT TERMINAL
        </span>
    </div>
    <span style="color:{TEXT_DIM}; font-size:0.6rem; font-family:{FONT_MONO};">
        v0.2 · SYS.OK
    </span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar (kept minimal — only nav & controls) ─────────────
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:{FONT_MONO}; font-size:0.65rem; color:{TEXT_DIM};
                text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">
        ▸ NAVIGATION
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAV",
        ["MARKET DATA", "STRATEGY LAB"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="font-family:{FONT_MONO}; font-size:0.65rem; color:{TEXT_DIM};
                text-transform:uppercase; letter-spacing:2px;
                margin:16px 0 8px 0;">
        ▸ SYMBOL
    </div>
    """, unsafe_allow_html=True)

    symbol = st.text_input(
        "ETF CODE",
        value="510300",
        placeholder="e.g. 510300",
        label_visibility="collapsed",
    ).strip()

    st.markdown(f"""
    <div style="font-family:{FONT_MONO}; font-size:0.65rem; color:{TEXT_DIM};
                text-transform:uppercase; letter-spacing:2px;
                margin:16px 0 8px 0;">
        ▸ TIMEFRAME
    </div>
    """, unsafe_allow_html=True)

    date_range = st.selectbox(
        "TIMEFRAME",
        options=["1M", "3M", "6M", "1Y", "3Y", "ALL"],
        index=3,
        label_visibility="collapsed",
    )

    date_map = {
        "1M": "20260601", "3M": "20260401", "6M": "20260101",
        "1Y": "20250601", "3Y": "20230601", "ALL": None,
    }
    start_date = date_map[date_range]

    st.markdown(f"""
    <div style="margin-top:24px; padding-top:12px;
                border-top:1px solid {BORDER};">
        <span style="font-family:{FONT_MONO}; font-size:0.55rem;
                     color:{TEXT_DIM};">
            DATA: TENCENT + BAIDU / SINA<br>
            ⚠ RISK DISCLAIMER APPLIES
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ─────────────────────────────────────────────

# Ticker tape (fetch watchlist)
try:
    watchlist = fetch_multi_etf_info(["510300", "510050", "510500", "159915", "588000"])
    ticker_data = [
        {
            "code": c, "name": q.get("name", c),
            "price": q.get("current_price"),
            "change_pct": q.get("change_pct"),
        }
        for c, q in watchlist.items()
    ]
    if ticker_data:
        ticker_header(ticker_data)
except Exception:
    pass  # Ticker is optional — don't block the app

# ── Fetch active symbol ──────────────────────────────────────
if not symbol:
    st.info("ENTER ETF CODE TO BEGIN")
elif len(symbol) != 6 or not symbol.isdigit():
    st.warning("ETF CODE MUST BE 6 DIGITS")
else:
    with st.spinner("FETCHING..."):
        try:
            info = fetch_etf_info(symbol)
            df = fetch_etf_hist(symbol, start_date=start_date)

            if page == "MARKET DATA":
                # ── Market Data Page ──
                name = info.get("name", symbol)
                st.markdown(f"""
                <div style="display:flex; align-items:baseline; gap:12px;
                            margin-bottom:4px;">
                    <span style="font-family:{FONT_MONO}; font-size:0.75rem;
                                 color:{TEXT_DIM};">{symbol}</span>
                    <span style="font-family:{FONT_MONO}; font-size:1.0rem;
                                 color:{TEXT};">{name}</span>
                </div>
                """, unsafe_allow_html=True)

                # Price
                price_display(
                    info.get("current_price") or 0,
                    info.get("change_pct"),
                )

                # Quick stats bar
                pe = info.get("pe_ttm")
                pb = info.get("pb")
                mcap = info.get("mcap_yi")
                turnover = info.get("turnover_pct")
                stats_html = (
                    f'<span style="color:{TEXT_DIM};">PE(TTM)</span> '
                    f'<span style="color:{TEXT}; margin-right:20px;">'
                    f'{pe:.1f}</span>' if pe else ''
                ) + (
                    f'<span style="color:{TEXT_DIM};">PB</span> '
                    f'<span style="color:{TEXT}; margin-right:20px;">'
                    f'{pb:.2f}</span>' if pb else ''
                ) + (
                    f'<span style="color:{TEXT_DIM};">MCAP</span> '
                    f'<span style="color:{TEXT}; margin-right:20px;">'
                    f'{mcap:.0f}亿</span>' if mcap else ''
                ) + (
                    f'<span style="color:{TEXT_DIM};">TURN</span> '
                    f'<span style="color:{TEXT};">'
                    f'{turnover:.2f}%</span>' if turnover else ''
                )
                st.markdown(
                    f'<p style="font-family:{FONT_MONO}; font-size:0.7rem; '
                    f'margin:8px 0 16px 0;">{stats_html}</p>',
                    unsafe_allow_html=True,
                )

                # Hi/Lo/Vol/Amt bar
                hi = info.get("high")
                lo = info.get("low")
                vol = info.get("volume")
                amt = info.get("amount")
                amp = info.get("amplitude")
                st.markdown(
                    f'<div style="display:flex; gap:24px; '
                    f'font-family:{FONT_MONO}; font-size:0.7rem; '
                    f'margin-bottom:12px;">'
                    f'<span><span style="color:{TEXT_DIM};">HI</span> '
                    f'<span style="color:{TEXT};">{hi:.3f}</span></span>'
                    f'<span><span style="color:{TEXT_DIM};">LO</span> '
                    f'<span style="color:{TEXT};">{lo:.3f}</span></span>'
                    f'<span><span style="color:{TEXT_DIM};">AMP</span> '
                    f'<span style="color:{TEXT};">{amp:.2f}%</span></span>'
                    f'<span><span style="color:{TEXT_DIM};">VOL</span> '
                    f'<span style="color:{TEXT};">{vol or "—"}</span></span>'
                    f'<span><span style="color:{TEXT_DIM};">AMT</span> '
                    f'<span style="color:{TEXT};">{amt or "—"}</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Bid/Ask
                section_header("ORDER BOOK")
                render_bid_ask_panel(info)

                # Chart
                section_header("PRICE CHART")
                render_price_chart(df)

                # Data table
                section_header("HISTORY")
                st.caption(f"{len(df)} RECORDS")
                render_data_table(df)

            else:
                # ── Strategy Lab Page ──
                st.markdown(f"""
                <div style="display:flex; align-items:baseline; gap:12px;
                            margin-bottom:12px;">
                    <span style="font-family:{FONT_MONO}; font-size:0.75rem;
                                 color:{TEXT_DIM};">{symbol}</span>
                    <span style="font-family:{FONT_MONO}; font-size:0.9rem;
                                 color:{TEXT};">STRATEGY LAB</span>
                    {status_badge("BACKTEST ENGINE ONLINE", "success")}
                </div>
                """, unsafe_allow_html=True)

                render_strategy_page(df, info)

        except ValueError as e:
            render_no_data(symbol, str(e))
        except Exception as e:
            st.error(f"SYS.ERR: {e}")
