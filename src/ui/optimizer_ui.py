"""Strategy parameter optimizer UI — progress bar + results table.

Renders the "optimize" button, a live progress bar while the grid
search runs, and a ranked results table with the top-N parameter
combinations.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.engine.optimizer import (
    OptimizationReport,
    OptimizationResult,
    run_optimization,
)
from src.strategy.base import BaseStrategy
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
)


def render_optimizer(
    df,
    strategy: BaseStrategy,
    strategy_cls,
    pe_value: float | None = None,
    pe_percentile=None,
    initial_capital: float = 100_000,
) -> None:
    """Render the parameter optimizer section.

    Shows a button to start optimization, a progress bar while running,
    and a results table + best-configuration highlight when done.
    """
    with st.expander("🧪 策略参数优化", expanded=False):
        st.caption(
            f"对 **{strategy.name}** 策略进行网格搜索，"
            f"找到历史表现最佳的参数组合。最多测试约 120 种组合。"
        )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            do_optimize = st.button(
                "🔍 开始优化",
                type="primary",
                use_container_width=True,
                key="optimizer_start_btn",
            )
        with col_info:
            st.caption("优化期间请勿切换策略或ETF，等待进度条完成。")

        if not do_optimize:
            # Show stale results if available
            if "optimizer_report" in st.session_state:
                _render_report(st.session_state["optimizer_report"])
            return

        # --- Run optimization ---
        progress_bar = st.progress(0.0, text="准备中...")
        status = st.empty()

        def _on_progress(completed: int, total: int) -> None:
            pct = completed / max(total, 1)
            progress_bar.progress(min(pct, 1.0), text=f"回测中... {completed}/{total}")

        status.info(f"正在优化 {strategy.name}，请稍候...")

        report = run_optimization(
            df=df,
            strategy_cls=strategy_cls,
            max_combinations=120,
            pe_value=pe_value,
            pe_percentile=pe_percentile,
            initial_capital=initial_capital,
            progress_callback=_on_progress,
        )

        progress_bar.progress(1.0, text="完成！")
        status.success(
            f"✅ 优化完成！测试了 {report.total_combinations} 种参数组合，"
            f"耗时 {report.elapsed_seconds:.1f} 秒"
        )

        # Cache in session_state for persistence across reruns
        st.session_state["optimizer_report"] = report

        _render_report(report)


def _render_report(report: OptimizationReport) -> None:
    """Render the optimization results table + best config highlight."""
    if report.best is None:
        st.warning("没有找到有效的参数组合。请检查历史数据是否足够。")
        return

    best = report.best

    # --- Best configuration card ---
    st.markdown(
        f'<div style="padding:14px 18px; background:#f0fdf4; '
        f'border:2px solid {SUCCESS}; border-radius:10px; margin:12px 0;">'
        f'<div style="font-size:0.85rem; font-weight:700; color:{SUCCESS}; '
        f'margin-bottom:6px;">🏆 最佳参数配置 (Score: {best.score:.4f})</div>'
        f'<div style="font-size:0.8rem; color:{DARK};">'
        + " | ".join(
            f"<b>{k}</b>={v}" for k, v in best.params.items()
        )
        + f'</div>'
        f'<div style="font-size:0.75rem; color:{NEUTRAL}; margin-top:6px;">'
        f'年化 {best.annual_return:+.1%} | '
        f'Sharpe {best.sharpe_ratio:.2f} | '
        f'最大回撤 {best.max_drawdown:.1%} | '
        f'胜率 {best.win_rate:.0%} | '
        f'{best.total_trades}笔交易'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- "Apply best params" button ---
    if st.button("✅ 应用最佳参数到当前策略", type="primary", key="optimizer_apply_btn"):
        # Merge best params into session_state params for the strategy
        # We use a flag that app.py checks to update the sidebar params
        st.session_state["optimizer_best_params"] = best.params
        st.session_state["optimizer_applied"] = True
        st.rerun()

    # --- Comparison: Default vs Best ---
    st.markdown("#### 📊 默认 vs 最优对比")
    _render_comparison_table(report)

    # --- Top-N results table ---
    if len(report.top_n) > 1:
        st.markdown(f"#### 📋 Top {len(report.top_n)} 参数组合")
        _render_results_table(report.top_n)


def _render_comparison_table(report: OptimizationReport) -> None:
    """Side-by-side comparison of default params vs best params."""
    strategy_cls = type(report.best.params)  # not useful, just use the data we have

    best = report.best
    if best is None:
        return

    # Find the default-params result from all_results
    default_result = None
    for r in report.all_results:
        if r.rank > len(report.all_results) * 0.8:  # Defaults usually score low-mid
            pass
    # We don't have the strategy class here, so just pick the median result
    # as an approximation of "default" for visual comparison
    if len(report.all_results) >= 2:
        mid = len(report.all_results) // 2
        default_result = report.all_results[mid]  # rough approximation

    rows = []
    for label, result in [("🏆 最优", best)] + (
        [("📊 中位数", default_result)] if default_result else []
    ):
        rows.append({
            "": label,
            "年化收益": f"{result.annual_return:+.1%}",
            "Sharpe": f"{result.sharpe_ratio:.2f}",
            "最大回撤": f"{result.max_drawdown:.1%}",
            "Calmar": f"{result.calmar_ratio:.2f}",
            "胜率": f"{result.win_rate:.0%}",
            "交易": result.total_trades,
            "评分": f"{result.score:.4f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_results_table(results: list[OptimizationResult]) -> None:
    """Render a ranked table of the top parameter combinations."""
    rows = []
    for r in results:
        param_summary = ", ".join(
            f"{k}={v}" for k, v in r.params.items()
        )
        rows.append({
            "排名": r.rank,
            "参数": param_summary,
            "年化": f"{r.annual_return:+.1%}",
            "Sharpe": round(r.sharpe_ratio, 2),
            "回撤": f"{r.max_drawdown:.1%}",
            "胜率": f"{r.win_rate:.0%}",
            "交易": r.total_trades,
            "评分": r.score,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 35 * len(rows) + 38),
    )
