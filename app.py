"""Standalone Streamlit app for ETF scanning and one paper account."""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.portfolio_db import PortfolioDB
from src.engine.backtest import (
    BacktestSettings,
    fetch_backtest_histories,
    run_parameter_sweep,
    run_rotation_backtest,
)
from src.engine.metrics import compute_drawdown_series
from src.engine.paper_trading import (
    RebalancePlan,
    build_rebalance_plan,
    execute_rebalance_plan,
)
from src.engine.portfolio import PortfolioManager
from src.engine.rotation_scanner import DEFAULT_ETF_POOL, scan_etf_pool
from src.strategy.etf_rotation import RotationConfig
from src.ui.backtest_dashboard import render_backtest_result, render_parameter_sweep
from src.ui.terminal_theme import PRIMARY, apply_chart_theme, inject_css


st.set_page_config(
    page_title="场内ETF轮动模拟账户",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


CATEGORY_LABELS = {
    "domestic_broad": "境内宽基",
    "domestic_sector": "境内行业",
    "overseas_equity": "海外权益",
    "commodity": "商品",
    "bond": "债券",
    "other": "其他",
}
ACTION_LABELS = {"buy": "买入", "sell": "卖出", "hold": "持有"}


def _parse_codes(text: str) -> tuple[str, ...]:
    codes = re.findall(r"(?<!\d)\d{6}(?!\d)", text or "")
    return tuple(dict.fromkeys(codes))


def _runtime_setting(name: str) -> str | None:
    """Read Streamlit secrets first, then environment variables."""

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else os.getenv(name)


@st.cache_resource
def _database(database_url: str | None) -> PortfolioDB:
    return PortfolioDB(database_url=database_url)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_scan(
    pool_key: tuple[str, ...],
    max_positions: int,
    cash_reserve: float,
    max_position_weight: float,
    min_avg_amount: float,
    min_daily_amount: float,
    correlation_threshold: float,
):
    default_map = {entry["code"]: dict(entry) for entry in DEFAULT_ETF_POOL}
    pool = [default_map.get(code, {"code": code}) for code in pool_key]
    config = RotationConfig(
        max_positions=max_positions,
        cash_reserve=cash_reserve,
        max_position_weight=max_position_weight,
        min_avg_amount=min_avg_amount,
        min_daily_amount=min_daily_amount,
        correlation_threshold=correlation_threshold,
    )
    return scan_etf_pool(pool=pool, config=config)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_backtest(
    pool_key: tuple[str, ...],
    start_date_value: str,
    end_date_value: str,
    initial_capital: float,
    slippage_pct: float,
    signal_frequency: str,
    benchmark_code: str,
    max_positions: int,
    cash_reserve: float,
    max_position_weight: float,
    min_avg_amount: float,
    min_daily_amount: float,
    correlation_threshold: float,
):
    default_map = {entry["code"]: dict(entry) for entry in DEFAULT_ETF_POOL}
    pool = [default_map.get(code, {"code": code}) for code in pool_key]
    start_value = date.fromisoformat(start_date_value)
    end_value = date.fromisoformat(end_date_value)
    data = fetch_backtest_histories(pool, start_value, end_value)
    config = RotationConfig(
        max_positions=max_positions,
        cash_reserve=cash_reserve,
        max_position_weight=max_position_weight,
        min_avg_amount=min_avg_amount,
        min_daily_amount=min_daily_amount,
        correlation_threshold=correlation_threshold,
    )
    settings = BacktestSettings(
        start_date=start_value,
        end_date=end_value,
        initial_capital=initial_capital,
        slippage_pct=slippage_pct,
        signal_frequency=signal_frequency,
        benchmark_code=benchmark_code,
    )
    result = run_rotation_backtest(
        data.histories,
        data.metadata,
        config,
        settings,
        data_errors=data.errors,
        requested_count=data.requested_count,
    )
    return data, result, config, settings


def _set_flash(level: str, message: str) -> None:
    st.session_state["paper_flash"] = (level, message)


def _show_flash() -> None:
    flash = st.session_state.pop("paper_flash", None)
    if not flash:
        return
    level, message = flash
    getattr(st, level, st.info)(message)


def _save_account(
    db: PortfolioDB,
    portfolio: PortfolioManager,
    data_as_of: date | None = None,
) -> None:
    if not db.save(portfolio):
        raise RuntimeError("模拟账户保存失败")
    db.record_snapshot(
        portfolio,
        snapshot_date=date.today().isoformat(),
        data_as_of=data_as_of.isoformat() if data_as_of else None,
    )


def _account_max_drawdown(curve: pd.DataFrame) -> float:
    if curve.empty:
        return 0.0
    return float(compute_drawdown_series(curve["equity"]).min())


def _win_rate(portfolio: PortfolioManager) -> float | None:
    sells = [trade for trade in portfolio.trades if trade.action == "sell" and trade.pnl is not None]
    if not sells:
        return None
    return sum(trade.pnl > 0 for trade in sells) / len(sells)


def _display_plan(plan: RebalancePlan) -> pd.DataFrame:
    frame = plan.orders.copy()
    if frame.empty:
        return frame
    frame["操作"] = frame["action"].map(ACTION_LABELS)
    frame["目标仓位"] = frame["target_weight"] * 100
    return frame[
        [
            "code", "name", "操作", "current_shares", "target_shares",
            "delta_shares", "reference_price", "目标仓位",
            "estimated_amount", "reason",
        ]
    ].rename(
        columns={
            "code": "代码",
            "name": "名称",
            "current_shares": "当前份额",
            "target_shares": "目标份额",
            "delta_shares": "调整份额",
            "reference_price": "参考收盘价",
            "estimated_amount": "预计金额",
            "reason": "原因",
        }
    )


def _display_targets(targets: pd.DataFrame, capital: float) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    frame = targets.copy()
    frame["类别"] = frame["category"].map(CATEGORY_LABELS).fillna("其他")
    frame["评分"] = frame["score"].round(1)
    frame["目标仓位"] = frame["target_weight"] * 100
    frame["建议金额"] = frame["target_weight"] * capital
    frame["参考价格"] = frame["close"].round(4)
    frame["建议份额"] = (
        np.floor(frame["建议金额"] / (frame["close"] * 100)) * 100
    ).fillna(0).astype(int)
    return frame[
        ["code", "name", "类别", "评分", "目标仓位", "参考价格", "建议份额", "建议金额"]
    ].rename(columns={"code": "代码", "name": "名称"})


def _display_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    frame = rankings.copy()
    frame["状态"] = frame["eligible"].map({True: "合格", False: "淘汰"})
    frame["类别"] = frame["category"].map(CATEGORY_LABELS).fillna("其他")
    frame["评分"] = frame["score"].round(1)
    for source, target in (
        ("return20", "20日收益"),
        ("return60", "60日收益"),
        ("return120", "120日收益"),
        ("volatility20", "20日年化波动"),
        ("max_drawdown60", "60日最大回撤"),
    ):
        frame[target] = frame[source] * 100
    return frame[
        [
            "code", "name", "类别", "状态", "评分", "20日收益", "60日收益",
            "120日收益", "20日年化波动", "60日最大回撤", "rejection_reason",
        ]
    ].rename(
        columns={"code": "代码", "name": "名称", "rejection_reason": "淘汰原因"}
    )


try:
    db = _database(_runtime_setting("DATABASE_URL"))
except Exception as error:
    st.error(f"数据库连接失败：{error}")
    st.info("请检查 Streamlit Secrets 中的 DATABASE_URL；未配置时会使用本地SQLite。")
    st.stop()
if not st.session_state.get("paper_account_loaded"):
    st.session_state["paper_portfolio"] = db.load()
    st.session_state["paper_account_loaded"] = True
portfolio: PortfolioManager | None = st.session_state.get("paper_portfolio")


with st.sidebar:
    st.title("场内ETF趋势轮动")
    st.caption("项目扫描 · 单账户模拟 · 手动实盘参考")
    database_ok, database_message = db.healthcheck()
    if db.is_cloud_persistent and database_ok:
        st.success("账户存储：PostgreSQL 持久化")
    elif database_ok:
        st.caption("账户存储：本地 SQLite（云端重启可能丢失）")
    else:
        st.error(f"账户存储异常：{database_message}")
    if db.migration_warning:
        st.warning(db.migration_warning)

    planned_capital = st.number_input(
        "计划/初始资金（元）",
        min_value=1_000,
        max_value=100_000_000,
        value=int(portfolio.initial_capital if portfolio else 100_000),
        step=10_000,
        disabled=portfolio is not None,
    )

    if portfolio is None:
        if st.button("创建模拟账户", type="primary", width="stretch"):
            portfolio = PortfolioManager(initial_capital=float(planned_capital))
            _save_account(db, portfolio)
            st.session_state["paper_portfolio"] = portfolio
            _set_flash("success", "模拟账户已创建")
            st.rerun()
    else:
        st.caption(f"账户总资产：¥{portfolio.total_equity:,.2f}")

    st.divider()
    st.subheader("扫描参数")
    max_positions = st.slider("最多持有", 1, 6, 4)
    cash_reserve = st.slider("最低现金比例", 0.0, 0.50, 0.10, 0.05)
    max_position_weight = st.slider("单只仓位上限", 0.10, 0.50, 0.30, 0.05)

    with st.expander("流动性与去重参数"):
        min_avg_amount_wan = st.number_input(
            "20日平均成交额下限（万元）",
            min_value=100,
            max_value=100_000,
            value=3_000,
            step=500,
        )
        min_daily_amount_wan = st.number_input(
            "20日最低成交额下限（万元）",
            min_value=50,
            max_value=50_000,
            value=500,
            step=100,
        )
        correlation_threshold = st.slider(
            "60日相关性去重阈值", 0.70, 0.99, 0.90, 0.01
        )

    pool_mode = st.radio("候选池", ["默认多资产池", "自定义ETF代码"])
    if pool_mode == "默认多资产池":
        pool_key = tuple(entry["code"] for entry in DEFAULT_ETF_POOL)
        st.caption(f"当前包含 {len(pool_key)} 只代表性场内ETF")
    else:
        custom_text = st.text_area(
            "ETF代码",
            value="510300, 510500, 513100, 518880, 511010",
            help="可用逗号、空格或换行分隔。持仓代码应始终保留在候选池中。",
        )
        pool_key = _parse_codes(custom_text)
        st.caption(f"识别到 {len(pool_key)} 个有效代码")

    current_config = RotationConfig(
        max_positions=max_positions,
        cash_reserve=cash_reserve,
        max_position_weight=max_position_weight,
        min_avg_amount=min_avg_amount_wan * 10_000.0,
        min_daily_amount=min_daily_amount_wan * 10_000.0,
        correlation_threshold=correlation_threshold,
    )

    run_scan = st.button("扫描并生成调仓清单", type="primary", width="stretch")

    with st.expander("账户备份与重置"):
        if db.is_cloud_persistent:
            st.caption("账户已保存到PostgreSQL；JSON备份仍可用于迁移或手工归档。")
        else:
            st.caption("云端本地磁盘可能在重启后丢失；请定期保存JSON备份。")
        if portfolio is not None:
            backup = json.dumps(
                db.export_backup(portfolio), ensure_ascii=False, indent=2
            ).encode("utf-8")
            st.download_button(
                "保存账户备份",
                data=backup,
                file_name=f"etf-paper-account-{date.today().isoformat()}.json",
                mime="application/json",
                width="stretch",
            )
        uploaded_backup = st.file_uploader("恢复账户备份", type=["json"])
        if st.button("确认恢复备份", width="stretch"):
            if uploaded_backup is None:
                st.warning("请先选择JSON备份文件")
            else:
                try:
                    payload = json.loads(uploaded_backup.getvalue().decode("utf-8"))
                    portfolio = db.restore_backup(payload)
                    st.session_state["paper_portfolio"] = portfolio
                    _set_flash("success", "账户备份已恢复")
                    st.rerun()
                except (ValueError, RuntimeError, json.JSONDecodeError) as error:
                    st.error(str(error))

        confirm_reset = st.checkbox("我确认清空全部模拟账户数据")
        if st.button(
            "重置模拟账户",
            disabled=not confirm_reset or portfolio is None,
            width="stretch",
        ):
            if db.reset():
                st.session_state["paper_portfolio"] = None
                st.session_state.pop("rotation_scan_result", None)
                _set_flash("success", "模拟账户已清空")
                st.rerun()
            st.error("模拟账户重置失败")

    st.caption("仅用于策略研究和模拟验证，不会向券商自动下单。")


st.title("场内ETF轮动 · 扫描与模拟账户")
st.caption(
    "唯一信号源为本项目成功读取的完整日线。扫描失败、数据过期或覆盖率不足时自动冻结交易。"
)
_show_flash()


if run_scan:
    if not pool_key:
        st.error("候选池为空，请输入至少一个6位ETF代码。")
    else:
        with st.spinner("正在读取历史行情、计算目标组合和调仓差额……"):
            result = _cached_scan(
                pool_key,
                max_positions,
                cash_reserve,
                max_position_weight,
                min_avg_amount_wan * 10_000.0,
                min_daily_amount_wan * 10_000.0,
                correlation_threshold,
            )
            st.session_state["rotation_scan_result"] = result
            st.session_state["rotation_scan_config"] = current_config


result = st.session_state.get("rotation_scan_result")
scan_config: RotationConfig = st.session_state.get(
    "rotation_scan_config", RotationConfig()
)

plan: RebalancePlan | None = None
if result is not None and portfolio is not None:
    plan = build_rebalance_plan(portfolio, result, scan_config)
    try:
        _save_account(db, portfolio, result.as_of)
    except RuntimeError as error:
        st.error(str(error))


curve = db.get_equity_curve() if portfolio is not None else pd.DataFrame()
if portfolio is not None:
    max_drawdown = _account_max_drawdown(curve)
    win_rate = _win_rate(portfolio)
    metric_cols = st.columns(6)
    metric_cols[0].metric("模拟总资产", f"¥{portfolio.total_equity:,.2f}")
    metric_cols[1].metric("累计收益", f"{portfolio.total_return_pct:+.2%}")
    metric_cols[2].metric("最大回撤", f"{max_drawdown:.2%}")
    metric_cols[3].metric("可用现金", f"¥{portfolio.cash:,.2f}")
    metric_cols[4].metric("当前持仓", f"{len(portfolio.holdings)} 只")
    metric_cols[5].metric("卖出胜率", "—" if win_rate is None else f"{win_rate:.1%}")

    st.subheader("模拟账户净值")
    if len(curve) >= 8:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["equity"],
                mode="lines+markers",
                name="账户净值",
                line=dict(color=PRIMARY, width=2),
                marker=dict(size=5, color=PRIMARY),
                hovertemplate="%{x|%Y-%m-%d}<br>总资产 ¥%{y:,.2f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=portfolio.initial_capital,
            line_color="#64748b",
            line_dash="dash",
            annotation_text="初始资金",
        )
        apply_chart_theme(fig, height=320)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_xaxes(title="日期")
        fig.update_yaxes(title="总资产（元）")
        st.plotly_chart(fig, width="stretch")
        st.caption("按每日最后一次保存的持仓收盘价估值；虚线为初始资金。")
    else:
        st.info(f"已记录 {len(curve)} 个日快照；累计到8个交易日后展示净值趋势图。")
else:
    st.info("请先在左侧创建模拟账户。创建后即可用扫描结果生成可执行的模拟调仓清单。")


tab_rebalance, tab_account, tab_trades, tab_universe, tab_backtest = st.tabs(
    ["今日调仓", "模拟账户", "交易记录", "候选明细", "历史回测"]
)


with tab_rebalance:
    if result is None:
        st.info("请在左侧点击“扫描并生成调仓清单”。")
    else:
        successful_total = result.scanned_count + len(result.errors)
        coverage = result.scanned_count / successful_total if successful_total else 0.0
        cols = st.columns(5)
        cols[0].metric("行情日期", result.as_of.isoformat() if result.as_of else "—")
        cols[1].metric("扫描成功", f"{result.scanned_count} 只")
        cols[2].metric("扫描覆盖率", f"{coverage:.0%}")
        cols[3].metric("趋势合格", f"{result.eligible_count} 只")
        cols[4].metric("目标持仓", f"{len(result.targets)} 只")

        if portfolio is None:
            st.warning("当前尚未创建模拟账户，只展示目标组合，不生成持仓差额。")
        elif plan is not None:
            st.subheader("调仓清单")
            if plan.errors:
                for error in plan.errors.values():
                    st.warning(error)
            display_plan = _display_plan(plan)
            st.dataframe(
                display_plan,
                hide_index=True,
                width="stretch",
                column_config={
                    "参考收盘价": st.column_config.NumberColumn(format="%.4f"),
                    "目标仓位": st.column_config.NumberColumn(format="%.1f%%"),
                    "预计金额": st.column_config.NumberColumn(format="¥%.2f"),
                    "调整份额": st.column_config.NumberColumn(format="%+d"),
                },
            )
            st.caption(
                "正数为买入、负数为卖出。参考价格是最新完整日线收盘价，实际模拟成交会加入滑点。"
            )

            with st.expander("执行全部模拟调仓", expanded=plan.actionable_count > 0):
                execution_date = st.date_input("模拟成交日期", value=date.today())
                slippage_bps = st.number_input(
                    "单边滑点（基点）", min_value=0, max_value=100, value=10, step=5
                )
                st.caption("10个基点 = 0.1%；系统始终先卖后买。")
                if st.button(
                    f"确认模拟执行（{plan.actionable_count} 笔）",
                    type="primary",
                    disabled=plan.actionable_count == 0,
                ):
                    execution_plan = build_rebalance_plan(
                        portfolio,
                        result,
                        scan_config,
                        trade_date=execution_date.isoformat(),
                    )
                    execution = execute_rebalance_plan(
                        portfolio,
                        execution_plan,
                        trade_date=execution_date.isoformat(),
                        slippage_pct=slippage_bps / 10_000.0,
                    )
                    if execution.trades:
                        _save_account(db, portfolio, result.as_of)
                        st.session_state["paper_portfolio"] = portfolio
                        message = f"已完成 {len(execution.trades)} 笔模拟成交"
                        if execution.errors:
                            message += f"，{len(execution.errors)} 笔未成交"
                        _set_flash("success", message)
                        st.rerun()
                    if execution.errors:
                        for error in execution.errors.values():
                            st.error(error)
                    else:
                        st.info("当前没有需要执行的模拟订单")

        st.subheader("目标组合")
        if result.targets.empty:
            st.warning("当前没有ETF满足全部条件，策略目标组合为空。")
        else:
            sizing_capital = portfolio.total_equity if portfolio else float(planned_capital)
            st.dataframe(
                _display_targets(result.targets, sizing_capital),
                hide_index=True,
                width="stretch",
                column_config={
                    "目标仓位": st.column_config.NumberColumn(format="%.1f%%"),
                    "建议金额": st.column_config.NumberColumn(format="¥%.2f"),
                    "参考价格": st.column_config.NumberColumn(format="%.4f"),
                },
            )


with tab_account:
    if portfolio is None:
        st.info("请先创建模拟账户。")
    else:
        st.subheader("当前持仓")
        holdings = portfolio.get_holdings_table()
        if holdings:
            holdings_frame = pd.DataFrame(holdings).rename(
                columns={
                    "code": "代码", "name": "名称", "shares": "份额",
                    "avg_cost": "平均成本", "total_cost": "持仓成本",
                    "current_price": "当前估值价", "market_value": "市值",
                    "unrealized_pnl": "浮动盈亏", "unrealized_pnl_pct": "浮动收益率",
                    "entry_date": "建仓日期", "highest_price": "持仓最高价",
                }
            )
            st.dataframe(
                holdings_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "平均成本": st.column_config.NumberColumn(format="%.4f"),
                    "当前估值价": st.column_config.NumberColumn(format="%.4f"),
                    "持仓最高价": st.column_config.NumberColumn(format="%.4f"),
                    "持仓成本": st.column_config.NumberColumn(format="¥%.2f"),
                    "市值": st.column_config.NumberColumn(format="¥%.2f"),
                    "浮动盈亏": st.column_config.NumberColumn(format="¥%+.2f"),
                    "浮动收益率": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
        else:
            st.info("当前全部持有现金。")

        st.subheader("登记单笔模拟成交")
        st.caption("用于补录实际手工成交或修正模拟记录；输入价格视为最终成交价，不再叠加滑点。")
        with st.form("manual_trade_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)
            manual_code = col1.text_input("ETF代码", max_chars=6)
            manual_action = col2.selectbox("操作", ["买入", "卖出"])
            manual_date = col3.date_input("成交日期", value=date.today())
            col4, col5, col6 = st.columns(3)
            manual_name = col4.text_input("名称（可选）")
            manual_price = col5.number_input(
                "成交价格", min_value=0.0001, value=1.0000, step=0.0010, format="%.4f"
            )
            manual_shares = col6.number_input(
                "成交份额", min_value=100, value=100, step=100
            )
            manual_reason = st.text_input("备注", value="手工登记")
            submitted = st.form_submit_button("登记模拟成交", type="primary")
        if submitted:
            code = manual_code.strip()
            if len(code) != 6 or not code.isdigit():
                st.error("请输入有效的6位ETF代码")
            else:
                if manual_action == "买入":
                    trade = portfolio.buy(
                        code, float(manual_price), int(manual_shares),
                        name=manual_name.strip(), reason=manual_reason,
                        trade_date=manual_date.isoformat(),
                    )
                else:
                    trade = portfolio.sell(
                        code, float(manual_price), int(manual_shares),
                        name=manual_name.strip(), reason=manual_reason,
                        trade_date=manual_date.isoformat(),
                    )
                if trade is None:
                    st.error("未成交：请检查现金、持仓、T+1可卖份额和100份交易单位。")
                else:
                    _save_account(db, portfolio, result.as_of if result else None)
                    st.session_state["paper_portfolio"] = portfolio
                    _set_flash("success", f"已登记{manual_action} {trade.code} {trade.shares}份")
                    st.rerun()


with tab_trades:
    if portfolio is None:
        st.info("请先创建模拟账户。")
    else:
        st.subheader("模拟成交记录")
        trades = portfolio.get_trade_history(200)
        if trades:
            st.dataframe(pd.DataFrame(trades), hide_index=True, width="stretch")
        else:
            st.info("尚无成交记录。")
        st.caption(
            f"记录共 {portfolio.total_trades} 笔；账户存储位置为 `{db.db_path}`。"
        )


with tab_universe:
    if result is None:
        st.info("完成一次扫描后查看候选ETF排名和淘汰原因。")
    elif result.rankings.empty:
        st.error("没有获得足够的有效历史数据，未生成任何信号。")
    else:
        st.subheader("全部候选与淘汰原因")
        st.dataframe(
            _display_rankings(result.rankings),
            hide_index=True,
            width="stretch",
            column_config={
                "20日收益": st.column_config.NumberColumn(format="%.2f%%"),
                "60日收益": st.column_config.NumberColumn(format="%.2f%%"),
                "120日收益": st.column_config.NumberColumn(format="%.2f%%"),
                "20日年化波动": st.column_config.NumberColumn(format="%.2f%%"),
                "60日最大回撤": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        if result.errors:
            with st.expander(f"数据读取失败（{len(result.errors)}只）"):
                for code, error in result.errors.items():
                    st.write(f"- `{code}`：{error}")


with tab_backtest:
    st.subheader("历史回测与基准比较")
    st.caption(
        "回测仅使用项目能够读取的ETF日线；信号在收盘后生成，下一交易日开盘模拟成交，"
        "并计入佣金、100份交易单位、T+1和所选滑点。"
    )
    backtest_col1, backtest_col2, backtest_col3 = st.columns(3)
    backtest_start = backtest_col1.date_input(
        "回测开始",
        value=date.today() - timedelta(days=365),
        key="backtest_start",
    )
    backtest_end = backtest_col2.date_input(
        "回测结束",
        value=date.today(),
        key="backtest_end",
    )
    backtest_capital = backtest_col3.number_input(
        "回测初始资金（元）",
        min_value=10_000,
        max_value=100_000_000,
        value=int(planned_capital),
        step=10_000,
        key="backtest_capital",
    )

    backtest_col4, backtest_col5, backtest_col6 = st.columns(3)
    frequency_label = backtest_col4.selectbox(
        "信号频率",
        ["每日", "每周末", "每月末"],
        help="每日与当前扫描纪律一致；周/月频率用于比较换手影响。",
    )
    frequency_map = {"每日": "daily", "每周末": "weekly", "每月末": "monthly"}
    backtest_slippage_bps = backtest_col5.number_input(
        "单边滑点（基点）",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
        key="backtest_slippage",
    )
    default_map = {entry["code"]: entry["name"] for entry in DEFAULT_ETF_POOL}
    benchmark_codes = list(pool_key)
    benchmark_default = benchmark_codes.index("510300") if "510300" in benchmark_codes else 0
    benchmark_code = backtest_col6.selectbox(
        "比较基准",
        benchmark_codes or ["510300"],
        index=benchmark_default if benchmark_codes else 0,
        format_func=lambda code: f"{code} {default_map.get(code, '')}".strip(),
    )

    run_backtest = st.button(
        "运行历史回测",
        type="primary",
        width="stretch",
        disabled=not bool(pool_key),
    )
    st.caption("默认19只ETF、每日信号和一年区间通常需要数十秒；相同参数会使用一小时缓存。")
    if run_backtest:
        if backtest_start >= backtest_end:
            st.error("回测开始日期必须早于结束日期。")
        else:
            with st.spinner("正在下载历史行情并逐交易日回放策略，请稍候……"):
                try:
                    bundle = _cached_backtest(
                        pool_key,
                        backtest_start.isoformat(),
                        backtest_end.isoformat(),
                        float(backtest_capital),
                        backtest_slippage_bps / 10_000.0,
                        frequency_map[frequency_label],
                        benchmark_code,
                        max_positions,
                        cash_reserve,
                        max_position_weight,
                        min_avg_amount_wan * 10_000.0,
                        min_daily_amount_wan * 10_000.0,
                        correlation_threshold,
                    )
                    st.session_state["backtest_bundle"] = bundle
                    st.session_state.pop("backtest_parameter_sweep", None)
                except Exception as error:
                    st.error(f"历史回测失败：{error}")

    backtest_bundle = st.session_state.get("backtest_bundle")
    if backtest_bundle is None:
        st.info("设置区间和基准后点击“运行历史回测”。回测结果不会修改模拟账户。")
    else:
        backtest_data, backtest_result, backtest_config, backtest_settings = backtest_bundle
        render_backtest_result(backtest_result)
        st.divider()
        st.subheader("参数稳健性检查")
        st.caption("固定其他参数，仅比较最多持有2、3、4、5只ETF；运行时间约为单次回测的4倍。")
        if st.button(
            "运行持仓数量稳健性检查",
            disabled=backtest_data.coverage < 0.80,
        ):
            with st.spinner("正在依次回放4组持仓数量参数……"):
                sweep = run_parameter_sweep(
                    backtest_data.histories,
                    backtest_data.metadata,
                    backtest_config,
                    backtest_settings,
                    max_positions_values=(2, 3, 4, 5),
                )
                st.session_state["backtest_parameter_sweep"] = sweep
        sweep = st.session_state.get("backtest_parameter_sweep")
        if sweep is not None:
            render_parameter_sweep(sweep)


with st.expander("策略与模拟交易规则", expanded=False):
    st.markdown(
        """
        - **唯一信号源**：本项目成功读取的最新完整日线；默认19只ETF，也可自定义候选池。
        - **买入**：进入目标组合后，按总资产目标权重和100份交易单位计算。
        - **卖出**：动态止损、移动止盈、趋势、排名和时间退出；卖出先于买入。
        - **费用模型**：ETF不计股票印花税；默认佣金万三、最低5元；批量模拟默认单边滑点0.1%。
        - **交易纪律**：同日新买份额不可卖；持仓数据缺失时冻结该标的；数据质量门禁失败时冻结全部自动调仓。
        - **实盘边界**：页面不会连接券商。真实成交后可在“模拟账户”中手工登记，用于对照跟踪。
        """
    )

st.caption("本项目仅用于策略研究和模拟验证，不构成收益承诺或个性化投资建议。")
