"""ETF 投资决策系统 — 策略优先的统一仪表盘"""

from __future__ import annotations

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
    BORDER,
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

# ── Auto Trader — background thread, starts on app load ─────────────
@st.cache_resource(show_spinner=False)
def _get_auto_trader():
    """Singleton auto-trader that runs in a daemon thread."""
    import threading
    import time as _time
    from datetime import datetime as _dt
    from src.engine.auto_trader import AutoTrader

    trader = AutoTrader(initial_capital=4000)
    status = {
        "running": True,
        "last_result": None,
        "last_error": None,
        "log_lines": [],
        "started_at": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    def _loop():
        while True:
            try:
                if not trader.is_market_open():
                    wait = trader.seconds_until_next_open()
                    if wait > 300:
                        status["log_lines"].append(
                            f"⏰ 休市，{wait/3600:.1f}h 后恢复"
                        )
                    _time.sleep(min(wait, 3600))
                    continue

                result = trader.run_once()
                status["last_result"] = result

                ts = result["timestamp"][11:19]
                trades = f"卖{result['exits_triggered']}买{result['entries_executed']}"
                msg = f"{ts} | {trades} | 权益¥{result['equity']:,.0f} | 持仓{result['positions']}"
                status["log_lines"].append(msg)

                # Keep last 50 lines
                if len(status["log_lines"]) > 50:
                    status["log_lines"] = status["log_lines"][-50:]

                _time.sleep(180)  # 3 min between scans
            except Exception as e:
                status["last_error"] = str(e)
                _time.sleep(60)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return trader, status


_auto_trader, _auto_status = _get_auto_trader()

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
    # Default to 短线波段 if available, else 4%定投法, else index 0
    default_idx = 0
    preferred = ["短线动量", "快速波段", "4%定投法", "短线波段"]
    for p in preferred:
        if p in strategy_names:
            default_idx = strategy_names.index(p)
            break
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

    # 4. ETF code — auto-select for band strategy, manual for others
    screener_code = st.session_state.get("screener_selected_code", "")

    if strategy.name == "短线波段":
        # Auto-select best ETF for short-term band trading.
        # Lock to current ETF while holding a position; re-scan on empty.
        st.caption("🔍 短线波段 · 自动选标的")

        # Check if we have an open position
        has_position = False
        holding_code = None
        if "portfolio" in st.session_state:
            pm = st.session_state["portfolio"]
            for h in pm.list_holdings() if hasattr(pm, "list_holdings") else []:
                if h.shares > 0:
                    has_position = True
                    holding_code = h.code
                    break

        # If we had a position but now it's gone → clear cache to trigger re-scan
        prev_code = st.session_state.get("band_selected_etf")
        if prev_code and not has_position:
            # Position was closed — force re-scan next time
            if st.session_state.get("band_had_position"):
                st.session_state.pop("band_selected_etf", None)
                st.session_state.pop("band_selected_info", None)
        st.session_state["band_had_position"] = has_position

        # Re-select when user clicks
        if st.button("🔄 重新扫描最优标的", type="secondary", use_container_width=True,
                     key="band_rescan_btn"):
            st.session_state.pop("band_selected_etf", None)
            st.session_state.pop("band_selected_info", None)
            st.session_state["band_full_scan"] = True  # manual = full backtest

        # Lock to current holding if in position
        if has_position and holding_code:
            symbol = holding_code
            st.success(f"📌 持仓中：{holding_code}（卖出后自动换标的）")
            st.session_state["band_selected_etf"] = holding_code
        else:
            # No position — auto-scan for best ETF
            selected_etf = st.session_state.get("band_selected_etf")
            selected_info = st.session_state.get("band_selected_info")

            if selected_etf is None:
                full_scan = st.session_state.pop("band_full_scan", False)
                mode_label = "深度扫描(含回测)" if full_scan else "快速扫描"
                with st.spinner(f"正在{mode_label}候选ETF..."):
                    from src.strategy.short_term_band import ShortTermBandStrategy
                    best = ShortTermBandStrategy.select_best_etf(quick=not full_scan)
                    if best:
                        selected_etf = best["code"]
                        selected_info = best
                        st.session_state["band_selected_etf"] = selected_etf
                        st.session_state["band_selected_info"] = selected_info

            if selected_info:
                price_str = f"¥{selected_info.get('current_price', '?'):.3f}" if selected_info.get("current_price") else "?"
                amp_str = f"{selected_info.get('amplitude', 0):.1f}%" if selected_info.get("amplitude") else "?"
                st.info(
                    f"📌 **{selected_info.get('code')} {selected_info.get('name_from_api', selected_info.get('name', ''))}**\n\n"
                    f"当前价：{price_str} | 振幅：{amp_str} | 评分：{selected_info.get('score', '?')}"
                )

            symbol = selected_etf if selected_etf else "510300"

        st.session_state["band_current_code"] = symbol
    elif strategy.name == "快速波段":
        st.caption("⚡ 快速波段 · PE+技术面双维度选基")

        # ── Capital input ──
        total_capital = st.number_input(
            "💰 你的总资金（元）",
            min_value=500, max_value=100000, value=2000, step=500,
            key="fastband_capital",
            help="用于计算实际买卖股数（100股整数倍）",
        )

        # Check position
        has_position = False
        holding_code = None
        holding_info = None
        if "portfolio" in st.session_state:
            pm = st.session_state["portfolio"]
            for h in pm.list_holdings() if hasattr(pm, "list_holdings") else []:
                if h.shares > 0:
                    has_position = True
                    holding_code = h.code
                    holding_info = h
                    break

        if st.button("🔄 重新扫描", type="secondary", use_container_width=True,
                     key="fastband_scan_btn"):
            st.session_state.pop("fastband_top5", None)
            st.session_state.pop("fastband_selected_symbol", None)

        if has_position and holding_code:
            symbol = holding_code
            h = holding_info
            cost = h.avg_cost if hasattr(h, 'avg_cost') else 0
            shares = h.shares if hasattr(h, 'shares') else 0
            mkt_val = shares * float(st.session_state.get("_last_price", 0) or 0)
            pnl_pct = (mkt_val / (cost * shares) - 1) if cost and shares else 0

            st.success(f"📌 持仓：{holding_code} | {shares}股 | 成本¥{cost:.3f}")
            if pnl_pct != 0:
                pnl_color = "green" if pnl_pct > 0 else "red"
                st.caption(f"浮动盈亏：:{pnl_color}[{pnl_pct:+.1%}]")

            # Sell recommendation based on P&L
            take_profit = 0.03
            stop_loss = 0.02
            if pnl_pct >= take_profit:
                st.warning(f"🎯 已触发止盈（+{pnl_pct:.1%} ≥ +{take_profit:.0%}），建议**全部卖出** {shares} 股")
            elif pnl_pct <= -stop_loss:
                st.error(f"🛑 已触发止损（{pnl_pct:.1%} ≤ -{stop_loss:.0%}），建议**全部卖出** {shares} 股")
            else:
                st.info(f"📊 继续持有 | 止盈+{take_profit:.0%} | 止损-{stop_loss:.0%}")

            st.session_state["fastband_selected_symbol"] = holding_code
        else:
            # Show current pick compactly
            current_pick = st.session_state.get("fastband_selected_symbol", "")
            if current_pick:
                pick_info = st.session_state.get("fastband_selected_info", {})
                pick_name = pick_info.get("name_from_api", pick_info.get("name", ""))
                pick_action = pick_info.get("action", "?")
                st.success(f"✅ {current_pick} {pick_name} | {pick_action}")
            else:
                st.info("👇 主区域选一只")

            # Manual input fallback
            manual = st.text_input(
                "或手动输入代码",
                value=current_pick if current_pick else "",
                placeholder="如 510300",
                key="fastband_manual_input",
            ).strip()
            symbol = manual if manual else current_pick if current_pick else ""

        # ── Position sizing suggestion (when ETF selected and not holding) ──
        if not has_position and symbol and len(symbol) == 6:
            pick_info = st.session_state.get("fastband_selected_info", {})
            entry_raw = pick_info.get("entry_score_raw", 0)
            cp = pick_info.get("current_price", 0)

            if cp > 0 and entry_raw > 0:
                position_pct = 0.5  # default 50%
                if entry_raw >= 7:
                    position_pct = 0.8
                    label = "重仓"
                elif entry_raw >= 5:
                    position_pct = 0.5
                    label = "半仓"
                elif entry_raw >= 3:
                    position_pct = 0.3
                    label = "轻仓"
                else:
                    position_pct = 0.0
                    label = "不入"

                if position_pct > 0:
                    budget = total_capital * position_pct
                    shares = max(100, int(budget / cp) // 100 * 100)
                    amount = shares * cp
                    st.info(
                        f"💡 建议**{label}**（{position_pct:.0%}仓位）：买入 **{shares} 股** ≈ ¥{amount:.0f} "
                        f"（{total_capital}元资金的{position_pct:.0%}）"
                    )

    elif strategy.name == "4%定投法":
        st.caption("🎯 4%定投法 · 智能选基")

        if st.button("🔄 重新扫描", type="secondary", use_container_width=True,
                     key="dca_scan_btn"):
            st.session_state.pop("dca_top3", None)
            st.session_state.pop("dca_selected_symbol", None)
            st.session_state.pop("dca_selected_name", None)

        # Show current pick compactly
        current_pick = st.session_state.get("dca_selected_symbol", "")
        if current_pick:
            pick_name = st.session_state.get("dca_selected_name", "")
            st.success(f"✅ {current_pick} {pick_name}")
        else:
            st.info("👇 主区域选一只")

        # Manual input fallback
        manual = st.text_input(
            "或手动输入代码",
            value=current_pick if current_pick else "",
            placeholder="如 510300",
            key="dca_manual_input",
        ).strip()
        symbol = manual if manual else current_pick if current_pick else ""
    else:
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

    # 8.6. Auto-trader status (always visible)
    st.divider()
    with st.expander("🤖 自动交易状态", expanded=True):
        if _auto_status["last_result"] is not None:
            last = _auto_status["last_result"]
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("💰 权益", f"¥{last['equity']:,.2f}")
            with col_b:
                st.metric("💵 现金", f"¥{last['cash']:,.2f}")

            st.caption(
                f"📊 持仓 {last['positions']} 个 | "
                f"最近扫描 {last['scanned']} 个标的 | "
                f"{last['timestamp'][11:19]}"
            )
        else:
            st.caption("⏳ 等待首次扫描...")

        # Last 5 log lines
        if _auto_status["log_lines"]:
            st.caption("---")
            for line in _auto_status["log_lines"][-5:]:
                st.caption(f"• {line}")

        if _auto_status.get("last_error"):
            st.error(f"⚠️ {_auto_status['last_error']}")

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

# --- 短线动量 Auto-Trader Dashboard (default landing page) ---
_selected_from_dashboard = st.session_state.pop("_at_select_code", "")

if strategy.name == "短线动量" and not _selected_from_dashboard and (not symbol or len(symbol) != 6 or st.session_state.get("_at_show_dashboard", True)):
    if st.session_state.get("_at_show_dashboard", True):
        st.header("🤖 自动交易 · 实时看板")

        # ── Portfolio overview ──
        last = _auto_status["last_result"]
        if last is not None:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("💰 总权益", f"¥{last['equity']:,.2f}")
            with c2:
                st.metric("💵 可用现金", f"¥{last['cash']:,.2f}")
            with c3:
                st.metric("📊 持仓数", f"{last['positions']} 个")
            with c4:
                pnl = last['equity'] - 4000
                st.metric("📈 浮动盈亏", f"¥{pnl:+,.2f}",
                         delta=f"{pnl/4000:+.1%}" if pnl != 0 else None)

        # ── Get detailed signals ──
        try:
            dash = _auto_trader.get_dashboard_signals()
        except Exception as e:
            st.warning(f"⚠️ 获取交易信号失败，请稍后刷新。详情：{e}")
            dash = {"positions": {}, "watchlist": {}}

        # ── Current positions (with exit advice) ──
        if dash.get("positions"):
            st.subheader("📌 当前持仓")
            for code, ps in dash["positions"].items():
                action = ps["action"]
                action_color = {
                    "sell": ("🔴 建议卖出", "#dc2626"),
                    "hold": ("🟡 继续持有", "#f59e0b"),
                }.get(action, ("⚪ 等待", "#64748b"))

                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.5])
                    with c1:
                        st.markdown(f"**{code}** {ps['name']}")
                        pnl_str = f"{ps['pnl_pct']:+.2%}" if ps['pnl_pct'] is not None else "—"
                        pnl_color = "#16a34a" if (ps['pnl_pct'] or 0) >= 0 else "#dc2626"
                        st.markdown(
                            f"现价 ¥{ps['current_price']:.3f} | "
                            f"成本 ¥{ps['avg_cost']:.3f} | "
                            f"<span style='color:{pnl_color};font-weight:600;'>{pnl_str}</span>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"<span style='color:{action_color[1]};font-weight:600;'>{action_color[0]}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(ps["reason"][:80])
                    with c3:
                        if ps.get("suggested_price_low") and ps.get("suggested_price_high"):
                            st.caption(
                                f"💰 建议卖出价：¥{ps['suggested_price_low']:.3f} ~ ¥{ps['suggested_price_high']:.3f}"
                            )
                        st.caption(f"📦 建议卖出：100 股 ≈ ¥{ps['current_price'] * 100:,.0f}" if ps['current_price'] else "—")
                    with c4:
                        if st.button(f"📊 详情", key=f"_pos_{code}", use_container_width=True):
                            st.session_state["_at_select_code"] = code
                            st.session_state["_at_show_dashboard"] = False
                            st.rerun()
        else:
            st.info("📭 暂无持仓，等待买入信号...")

        # ── Watchlist ranking (with entry advice) ──
        if dash["watchlist"]:
            st.subheader("🎯 候选池 Top 5")
            st.caption("开盘和午休各刷新。前日涨幅≥1.5% + 量能确认 + MA20上方 — 按综合评分排列。")

            for code, ws in dash["watchlist"].items():
                action = ws["action"]
                if action == "buy":
                    action_label = "🟢 建议买入"
                    action_clr = "#16a34a"
                elif action == "wait":
                    action_label = "⏳ 等待信号"
                    action_clr = "#64748b"
                else:
                    action_label = f"⚪ {action}"
                    action_clr = "#94a3b8"

                score_str = f"评分 {ws['score']:.0f}" if ws.get("score") is not None else ""

                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.5])
                    with c1:
                        st.markdown(f"**{code}** {ws['name']}")
                        price_str = f"¥{ws['current_price']:.3f}" if ws.get("current_price") else "—"
                        st.caption(f"现价 {price_str} | {score_str}")
                    with c2:
                        st.markdown(
                            f"<span style='color:{action_clr};font-weight:600;'>{action_label}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(ws["reason"][:80] if ws.get("reason") else "")
                    with c3:
                        if ws.get("suggested_price_low") and ws.get("suggested_price_high"):
                            st.caption(
                                f"💰 建议买入价：¥{ws['suggested_price_low']:.3f} ~ ¥{ws['suggested_price_high']:.3f}"
                            )
                        if ws.get("current_price"):
                            st.caption(f"📦 建议买入：100 股 ≈ ¥{ws['current_price'] * 100:,.0f}")
                    with c4:
                        if st.button(f"🔍 分析", key=f"_wl_{code}", use_container_width=True):
                            st.session_state["_at_select_code"] = code
                            st.session_state["_at_show_dashboard"] = False
                            st.rerun()
        else:
            st.info("🎯 候选池：等待完整扫描（开盘/午休各一次）...")

        # ── Recent logs ──
        if _auto_status["log_lines"]:
            with st.expander("📋 最近日志", expanded=False):
                for line in _auto_status["log_lines"][-10:]:
                    st.caption(f"• {line}")

        st.divider()
        st.caption("💡 在侧边栏输入ETF代码可查看任意基金详情")
        st.stop()

# If coming from dashboard with a selected code, use it
if _selected_from_dashboard:
    symbol = _selected_from_dashboard
    st.session_state["_at_show_dashboard"] = False

# --- 4% Fast Band: show Top 5 ranking when no ETF selected ---
if strategy.name == "快速波段" and (not symbol or len(symbol) != 6):
    st.header("⚡ 快速波段 · 25只宽基全量排名")

    # Ensure scan has run
    if st.session_state.get("fastband_top5") is None:
        with st.spinner("正在扫描25只宽基ETF（入场时机+波动+流动性+PE安全边际）..."):
            from src.strategy.fast_band_4pct import FastBand4PctStrategy
            st.session_state["fastband_top5"] = FastBand4PctStrategy.select_top_etfs(25)

    top5 = st.session_state.get("fastband_top5", [])

    if top5:
        # ── Overall summary ──
        strong_buys = [e for e in top5 if e.get("action_color") in ("strong_buy", "buy")]
        watches = [e for e in top5 if e.get("action_color") in ("watch", "speculative")]
        avoids = [e for e in top5 if e.get("action_color") in ("avoid", "skip")]

        summary_parts = []
        if strong_buys:
            summary_parts.append(f"🔥 **{len(strong_buys)}只建议买入**：{'、'.join(e['code'] for e in strong_buys)}")
        if watches:
            summary_parts.append(f"⏳ **{len(watches)}只值得关注**：{'、'.join(e['code'] for e in watches)}")
        if avoids:
            summary_parts.append(f"❌ **{len(avoids)}只建议回避**：{'、'.join(e['code'] for e in avoids)}")

        if summary_parts:
            st.info("  |  ".join(summary_parts))

        st.caption("入场时机(50%) + 波动性(25%) + 流动性(15%) + PE安全边际(10%)，按综合分排列")

        for i, etf in enumerate(top5):
            action = etf.get("action", "?")
            action_detail = etf.get("action_detail", "")
            action_color = etf.get("action_color", "skip")
            pe_badge = etf.get("pe_badge", "⚪ PE无数据")
            entry_raw = etf.get("entry_score_raw", 0)

            # Color bar based on action
            color_map = {
                "strong_buy": ("#16a34a", "#f0fdf4"),
                "buy": ("#16a34a", "#f0fdf4"),
                "speculative": ("#f59e0b", "#fffbeb"),
                "watch": ("#0891b2", "#ecfeff"),
                "wait": ("#64748b", "#f8fafc"),
                "track": ("#7c3aed", "#f5f3ff"),
                "avoid": ("#dc2626", "#fef2f2"),
                "skip": ("#94a3b8", "#f8fafc"),
            }
            bar_color, bar_bg = color_map.get(action_color, ("#94a3b8", "#f8fafc"))

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.2, 2.3, 2.5, 1.5])

                with c1:
                    st.markdown(
                        f'<div style="background:{bar_bg}; border-left:4px solid {bar_color}; '
                        f'padding:12px 8px; border-radius:4px; text-align:center;">'
                        f'<div style="font-size:1.6rem; font-weight:700; color:{bar_color};">{action}</div>'
                        f'<div style="font-size:0.7rem; color:#64748b;">综合 {etf["score"]:.0f}/100</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown(f"**{etf['code']}** {etf.get('name_from_api', etf['name'])}")
                    st.caption(f"💰 ¥{etf['current_price']:.3f}  |  {pe_badge}")
                    st.caption(f"🎯 入场{entry_raw:.0f}/10  |  振幅{etf['amplitude']:.1f}%")

                with c3:
                    st.caption(f"📊 入场{entry_raw:.0f} + 波动{etf['volatility_score']:.0f} + 流动{etf['liquidity_score']:.0f} + PE{etf['pe_score']:.0f}")
                    st.caption(f"💡 {action_detail}")
                    details = etf.get("score_details", [])
                    if details:
                        st.caption(" | ".join(details[:2]))

                with c4:
                    if st.button(f"📊 分析 {etf['code']}", key=f"fastband_pick_{i}",
                                 type="primary" if action_color in ("strong_buy", "buy") else "secondary",
                                 use_container_width=True):
                        st.session_state["fastband_selected_symbol"] = etf["code"]
                        st.session_state["fastband_selected_info"] = etf
                        st.rerun()
    else:
        st.warning("扫描失败，请在侧边栏手动输入 ETF 代码")

    # Custom input below Top 5
    st.divider()
    with st.expander("🔍 或者自己输入任意 ETF 代码", expanded=False):
        custom = st.text_input(
            "输入 6 位代码",
            placeholder="如 159915（创业板ETF）",
            key="fastband_custom_input",
        ).strip()
        if custom and len(custom) == 6 and custom.isdigit():
            if st.button("📊 分析这个 ETF", key="fastband_custom_go", type="primary",
                         use_container_width=True):
                st.session_state["fastband_selected_symbol"] = custom
                st.rerun()
        elif custom:
            st.caption("代码应为 6 位数字")

    st.stop()

# --- 4% DCA: show Top 3 ranking when no ETF selected ---
if strategy.name == "4%定投法" and (not symbol or len(symbol) != 6):
    st.header("🎯 4%定投法 · 为你筛选 Top 3 定投标的")

    # Ensure scan has run
    if st.session_state.get("dca_top3") is None:
        with st.spinner("正在扫描 25 只宽基 ETF（PE估值+近期走势）..."):
            from src.strategy.four_percent_dca import select_top_dca_etfs
            st.session_state["dca_top3"] = select_top_dca_etfs(3)

    top3 = st.session_state.get("dca_top3", [])

    if top3:
        st.caption("按 PE低估（50%）+ 近期跌幅（30%）+ 宽基加分（20%）排名，点击一只开始分析")

        for i, etf in enumerate(top3):
            medal = ["🥇", "🥈", "🥉"][i]

            # PE status
            pe_pct = etf.get("pe_percentile")
            if pe_pct is not None:
                if pe_pct < 30:
                    pe_badge = f"🟢 PE分位 {pe_pct:.0f}% 低估"
                elif pe_pct < 70:
                    pe_badge = f"🟡 PE分位 {pe_pct:.0f}% 合理"
                else:
                    pe_badge = f"🔴 PE分位 {pe_pct:.0f}% 高估"
            else:
                pe_badge = "⚪ PE无数据"

            ret = etf.get("recent_return_pct", 0)
            if ret < -2:
                trend_badge = f"📉 近5日 {ret:+.1f}%（定投良机）"
            elif ret < 0:
                trend_badge = f"📉 近5日 {ret:+.1f}%"
            elif ret > 2:
                trend_badge = f"📈 近5日 {ret:+.1f}%（等回调）"
            else:
                trend_badge = f"➡️ 近5日 {ret:+.1f}%"

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.5, 2, 2, 1.5])

                with c1:
                    st.markdown(f"## {medal}")
                    st.metric("评分", f"{etf['score']:.0f}/100")

                with c2:
                    st.markdown(f"**{etf['code']}** {etf['name']}")
                    st.caption(f"💰 ¥{etf['current_price']:.3f}")
                    st.caption(pe_badge)

                with c3:
                    st.caption(trend_badge)
                    st.caption(etf["reason"])

                    # Score breakdown
                    pe_s = etf.get("pe_score", 0)
                    dec_s = etf.get("decline_score", 0)
                    bonus = etf.get("type_bonus", 0)
                    st.caption(f"PE{pe_s:.0f} + 跌幅{dec_s:.0f} + 宽基{bonus:.0f}")

                with c4:
                    if st.button(f"📊 查看 {etf['code']}", key=f"dca_analyze_{i}",
                                 type="primary", use_container_width=True):
                        st.session_state["dca_selected_symbol"] = etf["code"]
                        st.session_state["dca_selected_name"] = etf["name"]
                        st.rerun()
    else:
        st.warning("扫描失败，请在下方手动输入 ETF 代码")

    # --- Custom ETF input (below Top 3) ---
    st.divider()
    with st.expander("🔍 或者自己输入任意 ETF 代码", expanded=False):
        custom = st.text_input(
            "输入 6 位代码",
            placeholder="如 510880（中证红利）",
            key="dca_custom_input",
        ).strip()
        if custom and len(custom) == 6 and custom.isdigit():
            if st.button("📊 分析这个 ETF", key="dca_custom_go", type="primary",
                         use_container_width=True):
                st.session_state["dca_selected_symbol"] = custom
                st.session_state["dca_selected_name"] = custom
                st.rerun()
        elif custom:
            st.caption("代码应为 6 位数字")

    st.stop()

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

    # ── Back to dashboard (for 短线动量 auto-trader) ──────────────
    if strategy.name == "短线动量" and not st.session_state.get("_at_show_dashboard", True):
        if st.button("← 返回自动交易看板", key="_back_to_dashboard"):
            st.session_state["_at_show_dashboard"] = True
            st.rerun()

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
    # Skip for band strategies — PE valuation is irrelevant for short-term trading
    if pe_percentile is not None and strategy.name not in ("短线波段", "快速波段"):
        render_pe_percentile_overview(pe_percentile)

    # ── 1.7 Portfolio context (needed by both signal panel & strategy) ──
    _pf_ctx = None
    _has_position = False
    if "portfolio" in st.session_state:
        _pf_ctx = get_portfolio_context(st.session_state["portfolio"], symbol)
        _has_position = _pf_ctx.get("has_position", False) if _pf_ctx else False

    # ── 1.8 Multi-Factor Daily Signal Panel ───────────────────────
    # Skip for band strategies — their own live signal + dashboard cards are the authority
    if strategy.name not in ("短线波段", "快速波段"):
        daily_signal = compute_daily_signal(
            df, info, pe_percentile=pe_percentile, macro_pulse=macro_pulse,
            has_position=_has_position,
        )
        render_signal_panel(daily_signal)

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
    # Skip for band strategies
    if pe_percentile is not None and strategy.name not in ("短线波段", "快速波段"):
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
