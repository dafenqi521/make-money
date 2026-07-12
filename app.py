"""ETF 投资决策系统 — 策略优先的统一仪表盘"""

import streamlit as st
from src.data.fetcher import fetch_etf_hist, fetch_etf_info
from src.strategy.registry import get_registry
from src.strategy.base import BaseStrategy
from src.engine.backtest import BacktestEngine
from src.engine.broker import Broker
from src.engine.risk import RiskManager
from src.ui.dashboard import (
    render_etf_overview,
    render_bid_ask_panel,
    render_price_chart,
    render_data_table,
    render_no_data,
)
from src.ui.strategy_ui import (
    _render_param_form,
    _render_metrics,
    _render_equity_chart,
    _render_trade_table,
)
from src.ui.strategy_dashboard import (
    render_strategy_header,
    render_live_signal,
    render_dashboard_cards,
)
from src.ui.etf_screener import render_etf_screener
from src.ui.portfolio_view import render_portfolio_section, get_portfolio_context
from src.ui.pe_band import render_pe_percentile_overview, render_pe_band
from src.data.pe_history import get_etf_pe_percentile, PEPercentile
from src.data.index_map import has_pe_data
from src.data.macro_pulse import get_macro_pulse, MacroPulse
from src.ui.macro_thermometer import render_macro_thermometer, render_mini_indicator
from src.ui.signal_panel import compute_daily_signal, render_signal_panel
from src.ui.optimizer_ui import render_optimizer
from src.ui.notify_settings import render_notify_settings
from src.ui.terminal_theme import (
    inject_css,
    PRIMARY,
    DARK,
    NEUTRAL,
    SUCCESS,
    DANGER,
    FONT,
)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF 投资决策系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Strategy registry ────────────────────────────────────────────
registry = get_registry()

# ── Macro pulse (global, symbol-independent — fetch once eagerly) ─
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_macro_pulse() -> MacroPulse | None:
    """Cache wrapper so sidebar + main area share the same fetch."""
    return get_macro_pulse()

_macro_pulse = _cached_macro_pulse()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    # 1. Title
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; '
        f'padding:4px 0 8px 0;">'
        f'<span style="font-size:1.2rem; font-weight:700; color:{PRIMARY}; '
        f'font-family:{FONT};">ETF 投资决策系统</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption("策略优先 · 数据驱动")

    st.divider()

    # 2. Strategy selector — ALWAYS FIRST
    strategy_names = registry.get_names()
    # Default to 4%定投法 if available, else index 0
    default_idx = (
        strategy_names.index("4%定投法") if "4%定投法" in strategy_names else 0
    )
    selected_name = st.selectbox(
        "🔄 当前策略",
        strategy_names,
        index=default_idx,
        key="selected_strategy",
    )
    strategy = registry.get_by_name(selected_name)

    st.divider()

    # 3. ETF Screener — strategy-aware
    with st.expander("🔍 筛选适配 ETF", expanded=False):
        st.caption(f"按「{strategy.name}」策略评分排序")
        if st.button("开始筛选 ETF", type="secondary", use_container_width=True,
                     key="sidebar_screener_btn"):
            st.session_state["run_screener"] = True

    # 4. ETF code (may be auto-filled by screener)
    # Check if screener selected a code
    screener_code = st.session_state.get("screener_selected_code", "")
    default_code = screener_code if screener_code else "510300"
    symbol = st.text_input(
        "ETF 代码",
        value=default_code,
        placeholder="输入代码，如 510300",
        key="etf_code_input",
    ).strip()

    # 5. Date range
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

    # 5. Strategy description
    with st.expander(f"📖 {strategy.name} — 策略说明", expanded=False):
        st.write(strategy.description)

    # 6. Dynamic parameter form — collapsed, defaults just work
    # Apply optimized params if user clicked "apply"
    if st.session_state.pop("optimizer_applied", False):
        best_params = st.session_state.get("optimizer_best_params", {})
        defaults = strategy.get_default_params()
        for key, val in best_params.items():
            widget_key = f"live_{key}"
            if key in defaults and isinstance(defaults[key], bool):
                st.session_state[widget_key] = "True" if val else "False"
            elif key in defaults and isinstance(defaults[key], str):
                st.session_state[widget_key] = str(val)
            else:
                st.session_state[widget_key] = val

    with st.expander("⚙️ 参数配置（可选，默认已优化）", expanded=False):
        st.caption("不改也能直接用。想调的话，改完点「开始回测」看效果。")
        params = _render_param_form(strategy, prefix="live")

    # 7. Backtest button
    st.divider()
    run_backtest = st.button(
        "▶ 开始回测", type="primary", use_container_width=True,
    )

    # 8. Macro mini indicator
    if _macro_pulse is not None and _macro_pulse.total_signals > 0:
        render_mini_indicator(_macro_pulse)

    # 8.5. Notification settings
    render_notify_settings()

    # 9. Footer
    st.divider()
    st.caption("数据源: 腾讯财经 + 百度 / 新浪\n\n⚠️ 投资有风险，本系统仅供学习参考")

# ── Main area ────────────────────────────────────────────────────

# ── ETF Screener section ────────────────────────────────────────
run_screener = st.session_state.pop("run_screener", False)

if run_screener or (not symbol and st.session_state.get("screener_results") is None):
    selected = render_etf_screener(selected_name)
    if selected:
        st.session_state["screener_selected_code"] = selected
        st.rerun()

# ── ETF Dashboard section ────────────────────────────────────────
if not symbol:
    if st.session_state.get("screener_results") is not None:
        st.info("👆 从筛选结果中选择一只 ETF，或在左侧手动输入代码")
    else:
        st.info("👈 请在左侧选择策略并输入 ETF 代码开始查询，或使用「筛选适配 ETF」功能")
elif len(symbol) != 6 or not symbol.isdigit():
    st.warning("ETF 代码应为 6 位数字，如 `510300`")
else:
    # ── Fetch data ────────────────────────────────────────────────
    with st.spinner(f"正在获取 {symbol} 的数据..."):
        try:
            info = fetch_etf_info(symbol)
            df = fetch_etf_hist(symbol, start_date=start_date)
        except ValueError as e:
            render_no_data(symbol, str(e))
            st.stop()
        except Exception as e:
            st.error(f"发生未预期的错误: {e}")
            st.stop()

    pe_value = info.get("pe_ttm") or info.get("pe_static")

    # ── PE历史分位 ───────────────────────────────────────────
    pe_percentile: PEPercentile | None = None
    if has_pe_data(symbol):
        with st.spinner("正在加载PE历史分位数据..."):
            pe_percentile = get_etf_pe_percentile(symbol, current_pe=pe_value)

    # ── 宏观情绪温度计 ──────────────────────────────────────
    macro_pulse = _macro_pulse

    # ── 1. ETF Overview ───────────────────────────────────────────
    render_etf_overview(info)

    # ── 1.5 PE Percentile Overview (when available) ────────────────
    if pe_percentile is not None:
        render_pe_percentile_overview(pe_percentile)

    # ── 1.7 Multi-Factor Daily Signal Panel ───────────────────────
    daily_signal = compute_daily_signal(
        df, info, pe_percentile=pe_percentile, macro_pulse=macro_pulse,
    )
    render_signal_panel(daily_signal)

    # ── 1.8 Extract portfolio context (for strategy anchoring) ────
    _pf_ctx = None
    if "portfolio" in st.session_state:
        _pf_ctx = get_portfolio_context(st.session_state["portfolio"], symbol)

    # ── 2. Live Signal ────────────────────────────────────────────
    try:
        live_signal = strategy.get_live_signal(
            df, info, pe_value=pe_value, pe_percentile=pe_percentile,
            macro_pulse=macro_pulse, portfolio_context=_pf_ctx, **params,
        )
    except Exception:
        # Fallback: minimal signal if strategy doesn't support live mode
        from src.strategy.signals import LiveSignal
        live_signal = LiveSignal(
            action="hold",
            current_price=info.get("current_price"),
            reason="无法生成实时信号",
        )

    # ── 3. Strategy Header ────────────────────────────────────────
    render_strategy_header(strategy, info, live_signal)

    # ── 4. Live Signal Card ───────────────────────────────────────
    render_live_signal(live_signal)

    # ── 5. Dashboard Cards ────────────────────────────────────────
    try:
        cards = strategy.get_dashboard_cards(
            df, info, pe_value=pe_value, pe_percentile=pe_percentile,
            macro_pulse=macro_pulse, portfolio_context=_pf_ctx, **params,
        )
        render_dashboard_cards(cards)
    except Exception:
        pass  # Cards are optional enhancement

    # ── 6. Portfolio / Paper Trading ───────────────────────────────
    name = info.get("name", f"ETF {symbol}")
    current_price = info.get("current_price")
    render_portfolio_section(live_signal, symbol, name, current_price)

    st.divider()

    # ── 6.5 PE Band Chart (when PE history available) ───────────
    if pe_percentile is not None:
        with st.expander(
            f"📈 PE Band · {pe_percentile.index_name}（历史PE走势图）",
            expanded=False,
        ):
            render_pe_band(symbol, pp=pe_percentile)

    # ── 6.7 Macro Thermometer (when data available) ────────────
    if macro_pulse is not None and macro_pulse.total_signals > 0:
        with st.expander(
            f"🌡️ 宏观情绪温度计 · 更新于 {macro_pulse.refreshed_at}",
            expanded=False,
        ):
            render_macro_thermometer(macro_pulse)

    # ── 7. K-line Chart with signal markers ───────────────────────
    try:
        markers = strategy.get_signal_markers(
            df, pe_value=pe_value, pe_percentile=pe_percentile, **params,
        )
    except Exception:
        markers = None
    if markers is not None and not markers.empty:
        st.caption(f"图表标注: {len(markers[markers['signal']=='buy'])} 个买入信号, "
                   f"{len(markers[markers['signal']=='sell'])} 个卖出信号")
    render_price_chart(df, markers=markers)

    # ── 8. Bid/Ask Panel (collapsed) ──────────────────────────────
    with st.expander("📊 五档盘口", expanded=False):
        render_bid_ask_panel(info)

    # ── 9. Backtest Results ───────────────────────────────────────
    if run_backtest:
        st.divider()
        st.subheader("📈 回测结果")

        with st.spinner("正在运行回测..."):
            try:
                engine = BacktestEngine(
                    initial_capital=100_000,
                    broker=Broker(),
                    risk_manager=RiskManager(),
                )
                result = engine.run(
                    df.copy(), strategy, pe_value=pe_value,
                    pe_percentile=pe_percentile, **params,
                )

                # Cache result so it survives reruns
                st.session_state["backtest_result"] = result
                st.session_state["backtest_strategy"] = strategy.name
                st.session_state["backtest_params_hash"] = hash(str(sorted(params.items())))

                # Metrics
                _render_metrics(result)

                # Equity chart + trades
                col1, col2 = st.columns([3, 2])
                with col1:
                    _render_equity_chart(result, title=strategy.name)
                with col2:
                    st.caption(f"共 {result.total_trades} 笔交易")
                    _render_trade_table(result)

                st.caption(result.summary())

            except Exception as e:
                st.error(f"回测运行失败: {e}")

    elif "backtest_result" in st.session_state:
        # Show stale result with a warning
        cached_strategy = st.session_state.get("backtest_strategy", "")
        if cached_strategy == strategy.name:
            st.divider()
            st.subheader("📈 上次回测结果")
            st.caption("参数未变，显示缓存结果。点击「开始回测」刷新。")
            result = st.session_state["backtest_result"]

            _render_metrics(result)
            col1, col2 = st.columns([3, 2])
            with col1:
                _render_equity_chart(result, title=strategy.name)
            with col2:
                st.caption(f"共 {result.total_trades} 笔交易")
                _render_trade_table(result)
        else:
            st.info("策略已切换，点击「开始回测」查看新策略的回测结果")

    # ── 9.5 Strategy Parameter Optimizer ──────────────────────────
    # Get the strategy class from the registry for the optimizer
    strategy_cls = type(strategy)
    initial_cap = st.session_state["portfolio"].initial_capital if "portfolio" in st.session_state else 100_000
    render_optimizer(
        df, strategy, strategy_cls,
        pe_value=pe_value, pe_percentile=pe_percentile,
        initial_capital=initial_cap,
    )

    # ── 10. Data Table (collapsed) ────────────────────────────────
    st.divider()
    with st.expander(f"📋 历史数据明细 (共 {len(df)} 条记录)", expanded=False):
        render_data_table(df)
