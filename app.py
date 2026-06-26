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

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ETF 投资决策系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: common controls ─────────────────────────────────
st.sidebar.title("📈 ETF 投资决策系统")
st.sidebar.caption("v0.2 — 数据展示 + 策略回测")

symbol = st.sidebar.text_input(
    "ETF 代码",
    value="510300",
    placeholder="输入 ETF 代码，如 510300",
    help="输入6位数字代码，如 沪深300ETF: 510300",
).strip()

# Date range controls
st.sidebar.subheader("📅 时间范围")
date_range = st.sidebar.selectbox(
    "选择周期",
    options=["近1个月", "近3个月", "近6个月", "近1年", "近3年", "全部"],
    index=3,
)

# Dynamic date range based on selection
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

# ── Page navigation ──────────────────────────────────────────
page = st.sidebar.radio(
    "📌 页面导航",
    ["📈 行情数据", "📊 策略回测"],
)

st.sidebar.divider()
st.sidebar.caption(
    "数据源: 腾讯财经 + 百度 / 新浪 (AKShare)\n\n"
    "⚠️ 投资有风险，本系统仅供学习参考"
)

# ── Fetch data (shared across pages) ─────────────────────────
if not symbol:
    st.info("👈 请在左侧输入 ETF 代码开始查询")
elif len(symbol) != 6 or not symbol.isdigit():
    st.warning("ETF 代码应为6位数字，如 `510300`")
else:
    with st.spinner(f"正在获取 {symbol} 的数据..."):
        try:
            info = fetch_etf_info(symbol)
            df = fetch_etf_hist(symbol, start_date=start_date)

            # ── Page: 行情数据 ─────────────────────────────────
            if page == "📈 行情数据":
                st.title("📈 ETF 行情数据")

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

            # ── Page: 策略回测 ─────────────────────────────────
            else:
                st.title("📊 策略回测")
                render_strategy_page(df, info)

        except ValueError as e:
            render_no_data(symbol, str(e))
        except Exception as e:
            st.error(f"发生未预期的错误: {e}")
            st.caption("请检查网络连接或稍后重试")
