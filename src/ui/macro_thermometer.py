"""Macro sentiment thermometer UI.

Renders a compact macro sentiment panel driven by prediction-market
probabilities from Polymarket.  Follows the existing card+column UI
patterns (same color constants, same border-radius, same font stacks).
"""

from __future__ import annotations

import streamlit as st

from src.data.macro_pulse import MacroPulse
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
)

# ---------------------------------------------------------------------------
# Module label → icon lookup
# ---------------------------------------------------------------------------

_MODULE_ICONS: dict[str, str] = {
    "monetary": "💰",
    "macro": "📊",
    "geopolitics": "🌍",
    "commodities": "🛢️",
    "ai_tech": "🤖",
}

# Risk label → display text
_RISK_LABELS: dict[str, str] = {
    "extreme": "极端恐惧",
    "high": "高度恐惧",
    "elevated": "偏高风险",
    "low": "情绪正常",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_macro_thermometer(pulse: MacroPulse) -> None:
    """Render the full macro sentiment panel.

    Should be placed in the main content area, typically inside an
    ``st.expander`` or as a standalone section.
    """
    if pulse.total_signals == 0:
        st.caption("暂无宏观情绪数据")
        return

    # ── Header row: overall gauge + risk badge ──
    _render_overall_gauge(pulse)

    # ── Module overview cards ──
    _render_module_cards(pulse)

    # ── Top signals detail ──
    _render_top_signals(pulse)


def render_mini_indicator(pulse: MacroPulse | None) -> None:
    """Render a single-line sentiment indicator for the sidebar.

    Extremely compact — just a colored dot + risk label + sentiment score.
    """
    if pulse is None or pulse.total_signals == 0:
        st.caption("🌡️ 宏观温度计: 暂无数据")
        return

    risk = pulse.risk_level
    color = pulse.risk_color
    label = _RISK_LABELS.get(risk, risk)
    score = pulse.overall_sentiment

    emoji_map = {"extreme": "🔴", "high": "🟠", "elevated": "🟡", "low": "🟢"}
    emoji = emoji_map.get(risk, "⚪")

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:6px; '
        f'padding:6px 10px; background:{BG_CARD}; border:1px solid {BORDER}; '
        f'border-radius:8px; font-size:0.75rem;">'
        f'<span>{emoji}</span>'
        f'<span style="color:{DARK}; font-weight:600;">{label}</span>'
        f'<span style="color:{color}; font-family:{FONT_MONO}; '
        f'font-weight:700;">{score:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_overall_gauge(pulse: MacroPulse) -> None:
    """Top section: sentiment bar + risk-level badge."""
    score = pulse.overall_sentiment
    risk = pulse.risk_level
    color = pulse.risk_color
    label = _RISK_LABELS.get(risk, risk)

    # Single row: bar left, badge right
    col_bar, col_badge = st.columns([4, 1])

    with col_bar:
        st.caption(f"🌡️ 宏观情绪温度计 · 更新于 {pulse.refreshed_at}")
        # Gradient-style progress bar — use columns to simulate color zones
        bars_cols = st.columns(100)
        # Green zone (60-100)
        green_end = max(0, min(100, int((score - 0.6) / 0.4 * 100))) if score > 0.6 else 0
        # Yellow zone (45-60)
        yellow_end = max(0, min(100, int((score - 0.45) / 0.15 * 100))) if score > 0.45 else 0
        # Orange zone (30-45)
        orange_end = max(0, min(100, int((score - 0.3) / 0.15 * 100))) if score > 0.3 else 0
        # Full bar: red base, overlay from left
        st.progress(min(score, 1.0))

        # Color-coded label
        st.caption(
            f"综合情绪指数: **{score:.3f}**  "
            f"覆盖 {pulse.module_count} 个模块 · {pulse.total_signals} 个信号"
        )

    with col_badge:
        st.markdown(
            f'<div style="padding:8px 12px; background:{color}15; '
            f'border:2px solid {color}; border-radius:8px; text-align:center; '
            f'margin-top:4px;">'
            f'<div style="font-size:0.65rem; color:{color};">风险等级</div>'
            f'<div style="font-size:1rem; font-weight:700; color:{color}; '
            f'font-family:{FONT};">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_module_cards(pulse: MacroPulse) -> None:
    """Row of module cards, 5 per row max, showing avg sentiment per module."""
    if not pulse.modules_detail:
        return

    detail = pulse.modules_detail
    # Sort modules: those with extreme scores first
    sorted_modules = sorted(
        detail.items(),
        key=lambda kv: abs(kv[1]["avg_sentiment"] - 0.5),
        reverse=True,
    )

    n = min(len(sorted_modules), 5)
    if n == 0:
        return
    cols = st.columns(n)

    for i, (mod_key, mod_info) in enumerate(sorted_modules[:n]):
        with cols[i]:
            avg = mod_info["avg_sentiment"]  # Already sentiment-normalised
            n_signals = mod_info["n_signals"]
            icon = _MODULE_ICONS.get(mod_key, "📌")
            label = mod_info.get("label", mod_key)

            # Color: green if >0.6, orange if 0.45-0.6, red if <0.45
            if avg >= 0.6:
                mod_color = SUCCESS
            elif avg >= 0.45:
                mod_color = WARNING
            else:
                mod_color = DANGER

            st.markdown(
                f'<div style="padding:10px; background:{BG_CARD}; '
                f'border:1px solid {BORDER}; border-radius:8px; '
                f'text-align:center;">'
                f'<div style="font-size:0.65rem; color:{NEUTRAL};">'
                f'{icon} {label}</div>'
                f'<div style="font-size:1.1rem; font-weight:700; '
                f'color:{mod_color}; font-family:{FONT_MONO};">'
                f'{avg:.2f}</div>'
                f'<div style="font-size:0.6rem; color:{NEUTRAL};">'
                f'{n_signals} 个信号</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_top_signals(pulse: MacroPulse) -> None:
    """Show top signals grouped by module, with probability bars."""
    if not pulse.signals:
        return

    # Group by module
    by_module: dict[str, list] = {}
    for s in pulse.signals:
        by_module.setdefault(s.module_label, []).append(s)

    for mod_label, mod_signals in by_module.items():
        # Show top 3 per module
        top = sorted(mod_signals, key=lambda s: s.volume_total, reverse=True)[:3]
        if not top:
            continue

        st.caption(f"**{mod_label}**")

        for s in top:
            prob_pct = s.probability * 100
            # Color: green if signal is bullish (>60%), orange if mixed, red if bearish (<40%)
            if prob_pct > 60:
                bar_color = SUCCESS
            elif prob_pct > 40:
                bar_color = WARNING
            else:
                bar_color = DANGER

            col_q, col_bar = st.columns([3, 1])
            with col_q:
                st.caption(
                    f"{s.question[:80]}{'…' if len(s.question) > 80 else ''}"
                )
            with col_bar:
                # Mini progress bar + percentage
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:4px;">'
                    f'<div style="flex:1; height:6px; background:{BORDER}; '
                    f'border-radius:3px;">'
                    f'<div style="width:{prob_pct}%; height:6px; '
                    f'background:{bar_color}; border-radius:3px;"></div>'
                    f'</div>'
                    f'<span style="font-size:0.7rem; font-weight:600; '
                    f'color:{bar_color}; min-width:40px; text-align:right;">'
                    f'{prob_pct:.0f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
