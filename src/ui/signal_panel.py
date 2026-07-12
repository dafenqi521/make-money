"""Multi-factor daily signal panel — PE + MA + Grid → actionable advice.

Independently evaluates three core factors (PE valuation, MA trend, grid
position) and produces a weighted composite score with specific action
steps.  Unlike strategy-specific ``LiveSignal``, this gives a bird's-eye
view of what ALL indicators are saying right now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
)

if TYPE_CHECKING:
    from src.data.pe_history import PEPercentile
    from src.data.macro_pulse import MacroPulse


# =========================================================================
# Data containers
# =========================================================================


@dataclass
class FactorSignal:
    """One factor's independent assessment."""

    name: str           # "PE估值" / "均线趋势" / "网格位置"
    signal: str         # "bullish" / "bearish" / "neutral" / "no_data"
    score: float        # 0.0 (bearish) → 1.0 (bullish), 0.5 = neutral
    label: str          # "低估区" / "多头排列" / "低位"
    detail: str         # One-sentence explanation
    icon: str           # "🟢" / "🔴" / "🟡" / "⚪"
    color: str          # CSS color
    weight: float       # 0.0–1.0, factor importance in composite


@dataclass
class DailySignal:
    """Multi-factor daily operation recommendation."""

    factors: list[FactorSignal] = field(default_factory=list)
    composite_score: float = 0.50
    composite_action: str = "hold"
    action_label: str = "持有观望"
    action_icon: str = "🔵"
    action_color: str = PRIMARY
    summary: str = ""
    steps: list[str] = field(default_factory=list)
    current_price: float | None = None
    pe_value: float | None = None


# =========================================================================
# Signal computation
# =========================================================================


def compute_pe_factor(
    pe_percentile: "PEPercentile | None",
    pe_value: float | None,
) -> FactorSignal:
    """Evaluate PE valuation independently.

    Prefers historical percentile when available; falls back to static
    PE snapshot thresholds.
    """
    # --- Historical percentile mode (preferred) ---
    if pe_percentile is not None and pe_percentile.pe_percentile is not None:
        pct = pe_percentile.pe_percentile
        pe_val = pe_percentile.current_pe

        if pct < 10:
            return FactorSignal(
                name="PE估值", signal="bullish", score=0.90,
                label=f"极度低估（分位{pct:.0f}%）",
                detail=f"PE={pe_val:.1f}，处于历史最低{pct:.0f}%区间，是极佳的建仓时机",
                icon="🟢", color=SUCCESS, weight=0.40,
            )
        elif pct < 30:
            return FactorSignal(
                name="PE估值", signal="bullish", score=0.75,
                label=f"低估区（分位{pct:.0f}%）",
                detail=f"PE={pe_val:.1f}，处于历史低估值区间，适合分批买入",
                icon="🟢", color=SUCCESS, weight=0.40,
            )
        elif pct < 50:
            return FactorSignal(
                name="PE估值", signal="neutral", score=0.55,
                label=f"合理偏低（分位{pct:.0f}%）",
                detail=f"PE={pe_val:.1f}，略低于历史中位数，可正常定投",
                icon="🟡", color=WARNING, weight=0.40,
            )
        elif pct < 70:
            return FactorSignal(
                name="PE估值", signal="neutral", score=0.45,
                label=f"合理偏高（分位{pct:.0f}%）",
                detail=f"PE={pe_val:.1f}，略高于历史中位数，建议减少买入",
                icon="🟡", color=WARNING, weight=0.40,
            )
        elif pct < 90:
            return FactorSignal(
                name="PE估值", signal="bearish", score=0.25,
                label=f"高估区（分位{pct:.0f}%）",
                detail=f"PE={pe_val:.1f}，处于历史高估值区间，应考虑分批止盈",
                icon="🔴", color=DANGER, weight=0.40,
            )
        else:
            return FactorSignal(
                name="PE估值", signal="bearish", score=0.10,
                label=f"极度高估（分位{pct:.0f}%）",
                detail=f"PE={pe_val:.1f}，处于历史最高{pct:.0f}%区间，强烈建议减仓",
                icon="🔴", color=DANGER, weight=0.40,
            )

    # --- Static PE threshold mode (fallback) ---
    if pe_value is not None and pe_value > 0:
        if pe_value < 12:
            return FactorSignal(
                name="PE估值", signal="bullish", score=0.75,
                label=f"低估（PE={pe_value:.1f}）",
                detail=f"PE(TTM)={pe_value:.1f}，绝对值处于低估区间",
                icon="🟢", color=SUCCESS, weight=0.40,
            )
        elif pe_value < 18:
            return FactorSignal(
                name="PE估值", signal="neutral", score=0.55,
                label=f"合理偏低（PE={pe_value:.1f}）",
                detail=f"PE(TTM)={pe_value:.1f}，估值合理偏低",
                icon="🟡", color=WARNING, weight=0.40,
            )
        elif pe_value < 25:
            return FactorSignal(
                name="PE估值", signal="neutral", score=0.45,
                label=f"合理偏高（PE={pe_value:.1f}）",
                detail=f"PE(TTM)={pe_value:.1f}，估值合理偏高",
                icon="🟡", color=WARNING, weight=0.40,
            )
        elif pe_value < 35:
            return FactorSignal(
                name="PE估值", signal="bearish", score=0.25,
                label=f"高估（PE={pe_value:.1f}）",
                detail=f"PE(TTM)={pe_value:.1f}，估值偏高，注意风险",
                icon="🔴", color=DANGER, weight=0.40,
            )
        else:
            return FactorSignal(
                name="PE估值", signal="bearish", score=0.10,
                label=f"极度高估（PE={pe_value:.1f}）",
                detail=f"PE(TTM)={pe_value:.1f}，估值严重偏高",
                icon="🔴", color=DANGER, weight=0.40,
            )

    # --- No PE data ---
    return FactorSignal(
        name="PE估值", signal="no_data", score=0.50,
        label="无PE数据",
        detail="该ETF暂无可用的PE(TTM)数据，PE因子不参与评分",
        icon="⚪", color=NEUTRAL, weight=0.40,
    )


def compute_ma_factor(df: pd.DataFrame) -> FactorSignal:
    """Evaluate moving-average trend independently.

    Checks for golden/death cross events and the overall MA alignment
    (bullish vs bearish).
    """
    # --- No data ---
    if df is None or len(df) < 2:
        return FactorSignal(
            name="均线趋势", signal="no_data", score=0.50,
            label="数据不足",
            detail="历史K线数据不足，无法计算均线趋势",
            icon="⚪", color=NEUTRAL, weight=0.35,
        )

    # --- Check MA columns ---
    has_ma5 = "ma5" in df.columns
    has_ma20 = "ma20" in df.columns
    has_ma10 = "ma10" in df.columns

    if not has_ma5 or not has_ma20:
        return FactorSignal(
            name="均线趋势", signal="no_data", score=0.50,
            label="缺少MA数据",
            detail="缺少MA5或MA20均线列，均线因子不参与评分",
            icon="⚪", color=NEUTRAL, weight=0.35,
        )

    df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
    last = df_sorted.iloc[-1]
    prev = df_sorted.iloc[-2] if len(df_sorted) >= 2 else None

    ma5_now = last["ma5"]
    ma20_now = last["ma20"]
    ma10_now = last["ma10"] if has_ma10 else None

    if pd.isna(ma5_now) or pd.isna(ma20_now):
        return FactorSignal(
            name="均线趋势", signal="no_data", score=0.50,
            label="MA值缺失",
            detail="最新K线的MA值为空，均线因子不参与评分",
            icon="⚪", color=NEUTRAL, weight=0.35,
        )

    # --- Detect cross events ---
    golden_cross = False
    death_cross = False
    if prev is not None:
        ma5_prev = prev["ma5"]
        ma20_prev = prev["ma20"]
        if pd.notna(ma5_prev) and pd.notna(ma20_prev):
            if ma5_prev <= ma20_prev and ma5_now > ma20_now:
                golden_cross = True
            elif ma5_prev >= ma20_prev and ma5_now < ma20_now:
                death_cross = True

    # --- Determine alignment ---
    if golden_cross:
        score = 0.85
        label = "金叉买入"
        detail = f"MA5({ma5_now:.3f}) 上穿 MA20({ma20_now:.3f})，短期趋势转强，典型买入信号"
        icon = "🟢"
        color = SUCCESS
        sig = "bullish"
    elif death_cross:
        score = 0.15
        label = "死叉卖出"
        detail = f"MA5({ma5_now:.3f}) 下穿 MA20({ma20_now:.3f})，短期趋势转弱，典型卖出信号"
        icon = "🔴"
        color = DANGER
        sig = "bearish"
    elif ma10_now is not None and pd.notna(ma10_now) and ma5_now > ma10_now > ma20_now:
        score = 0.70
        label = "多头排列"
        detail = f"MA5({ma5_now:.3f}) > MA10({ma10_now:.3f}) > MA20({ma20_now:.3f})，均线多头排列，趋势向好"
        icon = "🟢"
        color = SUCCESS
        sig = "bullish"
    elif ma10_now is not None and pd.notna(ma10_now) and ma5_now < ma10_now < ma20_now:
        score = 0.25
        label = "空头排列"
        detail = f"MA5({ma5_now:.3f}) < MA10({ma10_now:.3f}) < MA20({ma20_now:.3f})，均线空头排列，趋势偏弱"
        icon = "🔴"
        color = DANGER
        sig = "bearish"
    elif ma5_now > ma20_now:
        score = 0.60
        label = "偏多"
        detail = f"MA5({ma5_now:.3f}) > MA20({ma20_now:.3f})，短周期均线在上方，偏多但不稳固"
        icon = "🟡"
        color = WARNING
        sig = "neutral"
    else:
        score = 0.35
        label = "偏空"
        detail = f"MA5({ma5_now:.3f}) < MA20({ma20_now:.3f})，短周期均线在下方，偏空宜观望"
        icon = "🟡"
        color = WARNING
        sig = "neutral"

    return FactorSignal(
        name="均线趋势", signal=sig, score=score,
        label=label, detail=detail,
        icon=icon, color=color, weight=0.35,
    )


def compute_grid_factor(
    df: pd.DataFrame,
    current_price: float | None,
) -> FactorSignal:
    """Evaluate price position within historical range independently.

    Uses closing prices to establish the historical band, then assesses
    where the current price sits (percentile within band).
    """
    # --- No data ---
    if df is None or len(df) < 20:
        return FactorSignal(
            name="网格位置", signal="no_data", score=0.50,
            label="数据不足",
            detail="需要至少20个交易日数据才能评估价格区间位置",
            icon="⚪", color=NEUTRAL, weight=0.25,
        )

    df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)

    price_high = float(df_sorted["close"].max())
    price_low = float(df_sorted["close"].min())

    if price_high <= price_low:
        return FactorSignal(
            name="网格位置", signal="no_data", score=0.50,
            label="区间无效",
            detail="历史最高价≤最低价，无法计算价格区间",
            icon="⚪", color=NEUTRAL, weight=0.25,
        )

    # Current price: prefer real-time, fall back to last close
    if current_price is None:
        current_price = float(df_sorted.iloc[-1]["close"])

    if current_price is None or current_price <= 0:
        return FactorSignal(
            name="网格位置", signal="no_data", score=0.50,
            label="无当前价格",
            detail="无法获取当前价格，网格因子不参与评分",
            icon="⚪", color=NEUTRAL, weight=0.25,
        )

    # --- Compute position percentile ---
    band_range = price_high - price_low
    position_pct = (current_price - price_low) / band_range * 100
    position_pct = max(0, min(100, position_pct))  # clamp

    # --- Score by zone ---
    if position_pct < 20:
        score = 0.80
        label = f"低位（区间{position_pct:.0f}%）"
        detail = (
            f"当前价¥{current_price:.3f}处于历史价格区间的底部{position_pct:.0f}%位置，"
            f"历史最高¥{price_high:.3f}，最低¥{price_low:.3f}"
        )
        icon = "🟢"
        color = SUCCESS
        sig = "bullish"
    elif position_pct < 40:
        score = 0.60
        label = f"偏低（区间{position_pct:.0f}%）"
        detail = (
            f"当前价¥{current_price:.3f}处于历史价格区间的中低位置({position_pct:.0f}%)，"
            f"历史最高¥{price_high:.3f}"
        )
        icon = "🟡"
        color = SUCCESS
        sig = "neutral"
    elif position_pct < 60:
        score = 0.50
        label = f"中位（区间{position_pct:.0f}%）"
        detail = (
            f"当前价¥{current_price:.3f}处于历史价格区间的中间位置({position_pct:.0f}%)，"
            f"方向不明确"
        )
        icon = "🟡"
        color = WARNING
        sig = "neutral"
    elif position_pct < 80:
        score = 0.35
        label = f"偏高（区间{position_pct:.0f}%）"
        detail = (
            f"当前价¥{current_price:.3f}处于历史价格区间的中高位置({position_pct:.0f}%)，"
            f"历史最低¥{price_low:.3f}"
        )
        icon = "🔴"
        color = DANGER
        sig = "neutral"
    else:
        score = 0.20
        label = f"高位（区间{position_pct:.0f}%）"
        detail = (
            f"当前价¥{current_price:.3f}接近历史最高¥{price_high:.3f}，"
            f"处于价格区间顶部{position_pct:.0f}%位置"
        )
        icon = "🔴"
        color = DANGER
        sig = "bearish"

    return FactorSignal(
        name="网格位置", signal=sig, score=score,
        label=label, detail=detail,
        icon=icon, color=color, weight=0.25,
    )


def compute_daily_signal(
    df: pd.DataFrame,
    info: dict,
    pe_percentile: "PEPercentile | None" = None,
    macro_pulse: "MacroPulse | None" = None,
) -> DailySignal:
    """Compute the full multi-factor daily operation recommendation.

    This is the main entry point.  It computes PE, MA, and Grid factor
    signals independently, then combines them into a weighted composite
    with specific action steps.
    """
    pe_value = info.get("pe_ttm") or info.get("pe_static")
    current_price = info.get("current_price")

    # --- Compute each factor ---
    pe_factor = compute_pe_factor(pe_percentile, pe_value)
    ma_factor = compute_ma_factor(df)
    grid_factor = compute_grid_factor(df, current_price)

    factors = [pe_factor, ma_factor, grid_factor]

    # --- Weighted composite (only include factors with data) ---
    total_weight = 0.0
    weighted_score = 0.0
    for f in factors:
        if f.signal != "no_data":
            total_weight += f.weight
            weighted_score += f.score * f.weight

    if total_weight > 0:
        # Renormalize weights
        composite_score = weighted_score / total_weight
    else:
        composite_score = 0.50

    # --- Determine action from composite ---
    if composite_score >= 0.70:
        action = "buy"
        action_label = "强烈建议买入"
        action_icon = "🟢"
        action_color = SUCCESS
    elif composite_score >= 0.60:
        action = "accumulate"
        action_label = "建议增持"
        action_icon = "🟢"
        action_color = SUCCESS
    elif composite_score >= 0.45:
        action = "hold"
        action_label = "持有观望"
        action_icon = "🔵"
        action_color = PRIMARY
    elif composite_score >= 0.35:
        action = "reduce"
        action_label = "建议减仓"
        action_icon = "🟡"
        action_color = WARNING
    else:
        action = "sell"
        action_label = "建议卖出"
        action_icon = "🔴"
        action_color = DANGER

    # --- Build summary ---
    bullish_count = sum(1 for f in factors if f.signal == "bullish")
    bearish_count = sum(1 for f in factors if f.signal == "bearish")
    neutral_count = sum(1 for f in factors if f.signal == "neutral")

    summary_parts = []
    for f in factors:
        if f.signal != "no_data":
            summary_parts.append(f"{f.name}:{f.label}")
    summary = (
        f"综合评分 {composite_score:.2f}/1.00 | "
        + " | ".join(summary_parts)
        + f" | 看多:{bullish_count} 看空:{bearish_count} 中性:{neutral_count}"
    )

    # --- Build action steps ---
    steps = _build_action_steps(
        action, factors, current_price, pe_value, macro_pulse,
    )

    return DailySignal(
        factors=factors,
        composite_score=round(composite_score, 3),
        composite_action=action,
        action_label=action_label,
        action_icon=action_icon,
        action_color=action_color,
        summary=summary,
        steps=steps,
        current_price=current_price,
        pe_value=pe_value,
    )


def _build_action_steps(
    action: str,
    factors: list[FactorSignal],
    current_price: float | None,
    pe_value: float | None,
    macro_pulse: "MacroPulse | None",
) -> list[str]:
    """Generate specific, actionable steps based on the composite signal."""
    steps: list[str] = []

    price_str = f"¥{current_price:.3f}" if current_price else "当前价格"

    if action == "buy":
        steps.append(f"✅ 多因子共振看多，{price_str}是良好的入场价位")
        # Find the most bullish factor
        pe_factor = factors[0]
        if pe_factor.signal == "bullish":
            if pe_value:
                steps.append(f"📊 PE估值因子极度看多（PE={pe_value:.1f}），可加大仓位至正常的1.5-2倍")
            else:
                steps.append("📊 PE估值因子看多，估值处于历史低位区间")
        ma_factor = factors[1] if len(factors) > 1 else None
        if ma_factor and ma_factor.signal == "bullish":
            steps.append("📈 均线趋势确认多头，可跟随趋势加仓")
        grid_factor = factors[2] if len(factors) > 2 else None
        if grid_factor and grid_factor.signal == "bullish":
            steps.append("📐 价格处于网格低位，适合建立底仓")
        steps.append("💡 建议：分批建仓，每跌2-3%加一份，控制单次仓位不超过总资金的20%")

    elif action == "accumulate":
        steps.append(f"✅ 多数因子偏多，{price_str}适合适度增持")
        steps.append("💡 建议：按正常定投节奏买入，仓位控制在50-70%以内")

    elif action == "hold":
        steps.append(f"⏸️ 多空因子分歧，{price_str}建议保持现有仓位不动")
        # Check for mixed signals
        bullish = [f for f in factors if f.signal == "bullish"]
        bearish = [f for f in factors if f.signal == "bearish"]
        if bullish and bearish:
            bull_names = "、".join(f.name for f in bullish)
            bear_names = "、".join(f.name for f in bearish)
            steps.append(f"⚖️ {bull_names}看多 vs {bear_names}看空，信号矛盾，观望为主")
        steps.append("💡 建议：不买不卖，等待更明确的信号出现再操作")

    elif action == "reduce":
        steps.append(f"⚠️ 多数因子偏空，{price_str}建议考虑逐步减仓")
        steps.append("💡 建议：分2-3次减仓，每次减1/3，锁定部分利润")

    elif action == "sell":
        steps.append(f"🚨 多因子共振看空，{price_str}建议减仓或清仓")
        pe_factor = factors[0]
        if pe_factor.signal == "bearish":
            steps.append("📊 PE估值过高，历史分位处于危险区间")
        ma_factor = factors[1] if len(factors) > 1 else None
        if ma_factor and ma_factor.signal == "bearish":
            steps.append("📈 均线趋势转空，死叉/空头排列确认下跌趋势")
        steps.append("💡 建议：立即减仓50%以上，剩余部分设置止损线")

    # --- Macro overlay ---
    if macro_pulse is not None and macro_pulse.total_signals > 0:
        if macro_pulse.risk_level in ("high", "extreme"):
            steps.append(
                f"🌡️ 宏观情绪偏弱（指数{macro_pulse.overall_sentiment:.2f}），"
                f"建议降低仓位上限，增加现金比例"
            )
        elif macro_pulse.risk_level == "elevated":
            steps.append(
                f"🌡️ 宏观情绪中性偏弱，可正常操作但避免追高"
            )

    return steps


# =========================================================================
# UI rendering
# =========================================================================


def render_signal_panel(ds: DailySignal) -> None:
    """Render the multi-factor daily signal panel.

    Shows:
    1. A prominent composite action badge
    2. Three factor checklist cards (PE / MA / Grid)
    3. Composite score bar
    4. Specific action steps
    """
    # ── Section header ──────────────────────────────────────────────
    st.markdown("### 📋 今日操作建议")
    st.caption("多因子综合评估（PE估值 + 均线趋势 + 网格位置）")

    # ── Top: Composite action badge ─────────────────────────────────
    action_bg = _lighten(ds.action_color, 0.92)

    st.markdown(
        f'<div style="padding:16px 20px; background:{action_bg}; '
        f'border:2px solid {ds.action_color}; border-radius:10px; '
        f'margin:8px 0 12px 0; display:flex; align-items:center; gap:14px;">'
        f'<span style="font-size:2.2rem;">{ds.action_icon}</span>'
        f'<div>'
        f'<div style="font-size:1.15rem; font-weight:700; color:{ds.action_color}; '
        f'font-family:{FONT};">{ds.action_label}</div>'
        f'<div style="font-size:0.85rem; color:{DARK};">综合评分 '
        f'<b style="font-size:1.1rem;">{ds.composite_score:.2f}</b> / 1.00</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Middle: Three factor columns ─────────────────────────────────
    col1, col2, col3 = st.columns(3)
    factor_cols = [col1, col2, col3]

    for i, f in enumerate(ds.factors):
        if i >= len(factor_cols):
            break
        with factor_cols[i]:
            _render_factor_card(f)

    # ── Composite score bar ──────────────────────────────────────────
    score_pct = int(ds.composite_score * 100)
    # Gradient: red → yellow → green
    if score_pct >= 70:
        bar_color = SUCCESS
    elif score_pct >= 45:
        bar_color = PRIMARY
    elif score_pct >= 35:
        bar_color = WARNING
    else:
        bar_color = DANGER

    st.markdown(
        f'<div style="margin:8px 0 4px 0;">'
        f'<span style="font-size:0.75rem; color:{NEUTRAL};">多因子综合评分</span></div>'
        f'<div style="position:relative; height:10px; background:#e2e8f0; '
        f'border-radius:5px; margin-bottom:4px;">'
        f'<div style="height:10px; width:{score_pct}%; '
        f'background:{bar_color}; border-radius:5px; '
        f'transition:width 0.5s;"></div></div>'
        f'<div style="display:flex; justify-content:space-between; '
        f'font-size:0.65rem; color:{NEUTRAL};">'
        f'<span>强烈卖出</span><span>卖出</span><span>减仓</span>'
        f'<span>持有</span><span>增持</span><span>买入</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Summary ──────────────────────────────────────────────────────
    st.caption(ds.summary)

    # ── Action steps ─────────────────────────────────────────────────
    if ds.steps:
        st.markdown("**操作步骤**")
        for step in ds.steps:
            st.markdown(
                f'<div style="padding:6px 12px; margin:3px 0; '
                f'background:{BG_CARD}; border-left:3px solid {ds.action_color}; '
                f'border-radius:4px; font-size:0.85rem; color:{DARK};">'
                f'{step}</div>',
                unsafe_allow_html=True,
            )


def render_signal_panel_mini(ds: DailySignal) -> None:
    """Render a compact one-line signal summary for sidebars or narrow spaces."""
    st.markdown(
        f'<div style="padding:8px 12px; background:{BG_CARD}; '
        f'border:1px solid {BORDER}; border-radius:8px; '
        f'display:flex; align-items:center; gap:8px; margin:4px 0;">'
        f'<span style="font-size:1.1rem;">{ds.action_icon}</span>'
        f'<span style="font-size:0.85rem; font-weight:600; color:{ds.action_color};">'
        f'{ds.action_label}</span>'
        f'<span style="font-size:0.75rem; color:{NEUTRAL};">'
        f'综合 {ds.composite_score:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =========================================================================
# Internal helpers
# =========================================================================


def _render_factor_card(f: FactorSignal) -> None:
    """Render a single factor assessment card."""
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:6px; '
            f'margin-bottom:4px;">'
            f'<span style="font-size:1rem;">{f.icon}</span>'
            f'<span style="font-size:0.8rem; font-weight:600; color:{DARK};">'
            f'{f.name}</span>'
            f'<span style="display:inline-block; padding:1px 8px; '
            f'background:{f.color}15; color:{f.color}; '
            f'border-radius:10px; font-size:0.7rem; font-weight:600;">'
            f'{f.label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Score bar
        score_pct = int(f.score * 100)
        bar_color = (
            SUCCESS if f.score >= 0.60 else
            WARNING if f.score >= 0.40 else
            DANGER
        )
        st.markdown(
            f'<div style="height:4px; background:#e2e8f0; '
            f'border-radius:2px; margin:4px 0;">'
            f'<div style="height:4px; width:{score_pct}%; '
            f'background:{bar_color}; border-radius:2px;"></div></div>'
            f'<div style="font-size:0.65rem; color:{NEUTRAL}; '
            f'text-align:right;">评分 {f.score:.2f}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f.detail)


def _lighten(hex_color: str, factor: float = 0.9) -> str:
    """Mix a hex color with white to produce a lighter tint.

    Args:
        hex_color: e.g. "#16a34a"
        factor: 0.0 = white, 1.0 = original color. Default 0.9 = very light.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#f8fafc"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * (1.0 - factor))
    g = int(g + (255 - g) * (1.0 - factor))
    b = int(b + (255 - b) * (1.0 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"
