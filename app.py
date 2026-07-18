"""Single-strategy dashboard for exchange-traded ETF trend rotation."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from src.engine.rotation_scanner import DEFAULT_ETF_POOL, scan_etf_pool
from src.strategy.etf_rotation import RotationConfig
from src.ui.terminal_theme import inject_css


st.set_page_config(
    page_title="场内ETF趋势轮动",
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


def _parse_codes(text: str) -> tuple[str, ...]:
    codes = re.findall(r"(?<!\d)\d{6}(?!\d)", text or "")
    return tuple(dict.fromkeys(codes))


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


with st.sidebar:
    st.title("场内ETF趋势轮动")
    st.caption("唯一策略 · 多资产动量 · 二级市场买卖")

    capital = st.number_input(
        "计划资金（元）",
        min_value=1_000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
    )
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
        st.caption(f"当前包含 {len(pool_key)} 只代表性ETF；聚宽版本使用历史全市场ETF池。")
    else:
        custom_text = st.text_area(
            "ETF代码",
            value="510300, 510500, 513100, 518880, 511010",
            help="可用逗号、空格或换行分隔。",
        )
        pool_key = _parse_codes(custom_text)
        st.caption(f"识别到 {len(pool_key)} 个有效代码")

    run_scan = st.button("开始扫描", type="primary", use_container_width=True)
    st.divider()
    st.caption("只提供研究与模拟信号，不承诺收益。实盘前必须完成回测和模拟盘验证。")


st.title("场内ETF多资产趋势轮动")
st.write(
    "使用前一交易日完整日线，从流动性合格的ETF中选择绝对趋势向上、"
    "20/60/120日动量领先且波动和回撤较低的标的。"
)

if run_scan:
    if not pool_key:
        st.error("候选池为空，请输入至少一个6位ETF代码。")
    else:
        with st.spinner("正在读取历史行情并计算目标组合……"):
            st.session_state["rotation_scan_result"] = _cached_scan(
                pool_key,
                max_positions,
                cash_reserve,
                max_position_weight,
                min_avg_amount_wan * 10_000.0,
                min_daily_amount_wan * 10_000.0,
                correlation_threshold,
            )
            st.session_state["rotation_scan_capital"] = capital


result = st.session_state.get("rotation_scan_result")
scan_capital = float(st.session_state.get("rotation_scan_capital", capital))

if result is None:
    st.info("请在左侧确认参数后点击“开始扫描”。")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成功读取", f"{result.scanned_count} 只")
    col2.metric("趋势合格", f"{result.eligible_count} 只")
    col3.metric("目标持仓", f"{len(result.targets)} 只")
    deployed = float(result.targets["target_weight"].sum()) if not result.targets.empty else 0.0
    col4.metric("目标总仓位", f"{deployed:.1%}")

    if result.as_of:
        st.caption(f"日线数据截至：{result.as_of.isoformat()}。展示金额按扫描时计划资金 ¥{scan_capital:,.0f} 计算。")

    st.subheader("目标组合")
    if result.targets.empty:
        st.warning("当前没有ETF同时满足流动性与绝对趋势条件，策略建议持有现金。")
    else:
        targets = result.targets.copy()
        targets["类别"] = targets["category"].map(CATEGORY_LABELS).fillna("其他")
        targets["评分"] = targets["score"].round(1)
        targets["目标仓位"] = targets["target_weight"] * 100
        targets["建议金额"] = targets["target_weight"] * scan_capital
        targets["参考价格"] = targets["close"].round(4)
        targets["建议份额"] = (
            np.floor(targets["建议金额"] / (targets["close"] * 100)) * 100
        ).fillna(0).astype(int)
        targets["实际估算金额"] = targets["建议份额"] * targets["close"]
        st.dataframe(
            targets[
                [
                    "code",
                    "name",
                    "类别",
                    "评分",
                    "目标仓位",
                    "参考价格",
                    "建议份额",
                    "实际估算金额",
                ]
            ].rename(columns={"code": "代码", "name": "名称"}),
            hide_index=True,
            use_container_width=True,
            column_config={
                "目标仓位": st.column_config.NumberColumn(format="%.1f%%"),
                "实际估算金额": st.column_config.NumberColumn(format="¥%.2f"),
            },
        )
        st.caption("建议份额按100份向下取整，未计入实际佣金、滑点和盘口冲击。")

    st.subheader("全部候选与淘汰原因")
    if not result.rankings.empty:
        ranking = result.rankings.copy()
        ranking["状态"] = ranking["eligible"].map({True: "合格", False: "淘汰"})
        ranking["类别"] = ranking["category"].map(CATEGORY_LABELS).fillna("其他")
        ranking["评分"] = ranking["score"].round(1)
        for source, target in (
            ("return20", "20日收益"),
            ("return60", "60日收益"),
            ("return120", "120日收益"),
            ("volatility20", "20日年化波动"),
            ("max_drawdown60", "60日最大回撤"),
        ):
            ranking[target] = ranking[source] * 100
        st.dataframe(
            ranking[
                [
                    "code",
                    "name",
                    "类别",
                    "状态",
                    "评分",
                    "20日收益",
                    "60日收益",
                    "120日收益",
                    "20日年化波动",
                    "60日最大回撤",
                    "rejection_reason",
                ]
            ].rename(
                columns={"code": "代码", "name": "名称", "rejection_reason": "淘汰原因"}
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "20日收益": st.column_config.NumberColumn(format="%.2f%%"),
                "60日收益": st.column_config.NumberColumn(format="%.2f%%"),
                "120日收益": st.column_config.NumberColumn(format="%.2f%%"),
                "20日年化波动": st.column_config.NumberColumn(format="%.2f%%"),
                "60日最大回撤": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    else:
        st.error("没有获得足够的有效历史数据，未生成任何买入信号。")

    if result.errors:
        with st.expander(f"数据读取失败（{len(result.errors)}只）"):
            for code, error in result.errors.items():
                st.write(f"- `{code}`：{error}")


with st.expander("买入、卖出和风险规则", expanded=False):
    st.markdown(
        """
        - **买入**：盘前排名后，10:05检查开盘偏离和30分钟VWAP，卖出完成后再买入。
        - **硬止损**：`2.5 × ATR20`，限制在3%～8%。
        - **移动止盈**：最高盈利达到2%后，回撤`2 × ATR20`，限制在2.5%～10%。
        - **趋势退出**：连续两日跌破MA20，或MA5下穿MA10。
        - **排名退出**：持有至少5日后，连续两日跌出前8名。
        - **时间止损**：持有10日仍未形成有效趋势且排名偏弱。
        - **交易纪律**：所有ETF统一禁止当日买入后卖出；无合格标的时保留现金。
        """
    )

st.caption("自动执行版本位于 joinquant/etf_rotation_strategy.py；本地页面只生成研究和模拟组合建议。")
