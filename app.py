"""ETF 投资决策系统 — 专业金融仪表盘"""

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
from src.ui.terminal_theme import (
    inject_css,
    PRIMARY,
    DARK,
    NEUTRAL,
    BG_CARD,
    BORDER,
    SUCCESS,
    DANGER,
    FONT,
    FONT_MONO,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ETF 投资决策系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; '
        f'padding:4px 0 12px 0;">'
        f'<span style="font-size:1.2rem; font-weight:700; color:{PRIMARY}; '
        f'font-family:{FONT};">ETF 投资决策系统</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption("数据展示 + 策略回测")

    st.divider()

    symbol = st.text_input(
        "ETF 代码",
        value="510300",
        placeholder="输入代码，如 510300",
    ).strip()

    date_range = st.selectbox(
        "时间范围",
        options=["近1个月", "近3个月", "近6个月", "近1年", "近3年", "全部"],
        index=3,
    )
    date_map = {
        "近1个月": "20260601", "近3个月": "20260401", "近6个月": "20260101",
        "近1年": "20250601", "近3年": "20230601", "全部": None,
    }
    start_date = date_map[date_range]

    st.divider()
    st.caption("数据源: 腾讯财经 + 百度 / 新浪\n\n⚠️ 投资有风险，本系统仅供学习参考")

# ── Fetch data ───────────────────────────────────────────────
if not symbol:
    st.info("👈 请在左侧输入 ETF 代码开始查询")
elif len(symbol) != 6 or not symbol.isdigit():
    st.warning("ETF 代码应为 6 位数字，如 `510300`")
else:
    with st.spinner(f"正在获取 {symbol} 的数据..."):
        try:
            info = fetch_etf_info(symbol)
            df = fetch_etf_hist(symbol, start_date=start_date)

            # ── Fluid tab navigation ─────────────────────────
            tab1, tab2 = st.tabs(["📈 行情数据", "📊 策略回测"])

            with tab1:
                render_etf_overview(info)
                st.divider()
                render_bid_ask_panel(info)
                st.divider()
                sub1, sub2 = st.tabs(["K线图", "数据明细"])
                with sub1:
                    render_price_chart(df)
                with sub2:
                    st.caption(f"共 {len(df)} 条记录")
                    render_data_table(df)

            with tab2:
                render_strategy_page(df, info)

        except ValueError as e:
            render_no_data(symbol, str(e))
        except Exception as e:
            st.error(f"发生未预期的错误: {e}")
