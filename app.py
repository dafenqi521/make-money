"""ETF Investment Decision System — v0.2

A Streamlit app for ETF data display and strategy backtesting.
"""

import streamlit as st
from src.data.fetcher import fetch_etf_hist, fetch_etf_info
from src.ui.dashboard import (
    render_etf_overview,
    render_bid_ask_panel,
    render_price_chart,
    render_data_table,
    render_no_data,
)
from src.ui.strategy_ui import render_strategy_page
from src.ui.theme import inject_css, NEUTRAL

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ETF 投资决策系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS once per session
inject_css()

# ── Sidebar: brand header ────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="display:flex; align-items:center; gap:10px; '
        'padding:8px 0 16px 0;">'
        '<span style="font-size:1.5rem; font-weight:700; color:#1a56db;">'
        'ETF 投资决策系统</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f'<span style="color:{NEUTRAL};">v0.2 · 数据展示 + 策略回测</span>',
               unsafe_allow_html=True)

# ── Sidebar: controls ────────────────────────────────────────
symbol = st.sidebar.text_input(
    "ETF 代码",
    value="510300",
    placeholder="输入 ETF 代码，如 510300",
    help="输入6位数字代码，如 沪深300ETF: 510300",
).strip()

st.sidebar.subheader("📅 时间范围")
date_range = st.sidebar.selectbox(
    "选择周期",
    options=["近1个月", "近3个月", "近6个月", "近1年", "近3年", "全部"],
    index=3,
)

date_map = {
    "近1个月": "20260601",
    "近3个月": "20260401",
    "近6个月": "20260101",
    "近1年": "20250601",
    "近3年": "20230601",
    "全部": None,
}
start_date = date_map[date_range]

st.sidebar.divider()

# ── Sidebar: navigation ──────────────────────────────────────
page = st.sidebar.radio(
    "📌 页面导航",
    ["📈 行情数据", "📊 策略回测"],
)

st.sidebar.divider()
st.sidebar.caption(
    f'<span style="color:{NEUTRAL}; font-size:0.8rem;">'
    "数据源: 腾讯财经 + 百度 / 新浪 (AKShare)<br><br>"
    "⚠️ 投资有风险，本系统仅供学习参考"
    "</span>",
    unsafe_allow_html=True,
)

# ── Main content ─────────────────────────────────────────────
if not symbol:
    st.info("👈 请在左侧输入 ETF 代码开始查询")
elif len(symbol) != 6 or not symbol.isdigit():
    st.warning("ETF 代码应为6位数字，如 `510300`")
else:
    with st.spinner(f"正在获取 {symbol} 的数据..."):
        try:
            info = fetch_etf_info(symbol)
            df = fetch_etf_hist(symbol, start_date=start_date)

            if page == "📈 行情数据":
                st.title("行情数据")
                render_etf_overview(info)
                st.divider()
                render_bid_ask_panel(info)
                st.divider()
                tab1, tab2 = st.tabs(["📊 K线图 (含均线)", "📋 数据明细"])
                with tab1:
                    render_price_chart(df)
                with tab2:
                    st.caption(f"共 {len(df)} 条记录")
                    render_data_table(df)

            else:
                st.title("策略回测")
                render_strategy_page(df, info)

        except ValueError as e:
            render_no_data(symbol, str(e))
        except Exception as e:
            st.error(f"发生未预期的错误: {e}")
            st.caption("请检查网络连接或稍后重试")
