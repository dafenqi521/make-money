"""Strategy Dashboard UI — live signal cards, dashboard card grid, strategy header.

Renders the strategy-centric main area: a prominent live-signal recommendation,
a grid of strategy-specific info cards, and a compact strategy header bar.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.strategy.signals import LiveSignal, DashboardCard
from src.strategy.base import BaseStrategy
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
)


# ---------------------------------------------------------------------------
# Strategy header bar
# ---------------------------------------------------------------------------

def render_strategy_header(strategy: BaseStrategy, info: dict, signal: LiveSignal) -> None:
    """Compact bar showing strategy name, current zone, and quick status."""
    zone = signal.current_zone or ""
    action = signal.action

    if action in ("buy",):
        zone_color = SUCCESS
        action_label = "买入信号"
    elif action in ("sell",):
        zone_color = DANGER
        action_label = "卖出信号"
    elif action in ("wait_for_drop", "wait_for_rise"):
        zone_color = WARNING
        action_label = "等待触发"
    else:
        zone_color = PRIMARY
        action_label = "持有观望"

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:12px; '
        f'padding:10px 16px; background:{BG_CARD}; border:1px solid {BORDER}; '
        f'border-radius:8px; margin:8px 0 16px 0;">'
        f'<span style="font-weight:700; font-size:1rem; color:{DARK}; '
        f'font-family:{FONT};">{strategy.name}</span>'
        f'<span style="font-size:0.75rem; color:{NEUTRAL};">|</span>'
        f'<span style="display:inline-block; padding:2px 10px; '
        f'background:{zone_color}15; color:{zone_color}; '
        f'border-radius:12px; font-size:0.8rem; font-weight:600;">'
        f'{zone}</span>'
        f'<span style="font-size:0.8rem; color:{zone_color}; font-weight:600;">'
        f'{action_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Live signal card — the most prominent element on the page
# ---------------------------------------------------------------------------

def render_live_signal(signal: LiveSignal) -> None:
    """Render a large, color-coded live trading recommendation card.

    Green  = buy / accumulate
    Red    = sell / reduce
    Blue   = hold / wait
    Yellow = waiting for trigger
    """
    action = signal.action

    # ── Color scheme ──
    if action in ("buy",):
        bg = "#f0fdf4"
        border = SUCCESS
        icon = "🟢"
        action_text = "建议买入"
        action_color = SUCCESS
    elif action in ("sell",):
        bg = "#fef2f2"
        border = DANGER
        icon = "🔴"
        action_text = "建议卖出"
        action_color = DANGER
    elif action in ("wait_for_drop", "wait_for_rise", "wait_for_strength"):
        bg = "#fffbeb"
        border = WARNING
        icon = "🟡"
        action_text = "等待触发"
        action_color = WARNING
    else:
        bg = "#eff6ff"
        border = PRIMARY
        icon = "🔵"
        action_text = "继续持有"
        action_color = PRIMARY

    # ── Build HTML ──
    price_str = f"¥{signal.current_price:.3f}" if signal.current_price else "—"
    trigger_str = (
        f"¥{signal.next_trigger_price:.3f}"
        if signal.next_trigger_price
        else ""
    )

    # Portions bar (if applicable)
    portions_html = ""
    if signal.portions_used is not None and signal.portions_total is not None:
        pct = signal.portions_used / max(signal.portions_total, 1) * 100
        portions_html = (
            f'<div style="margin-top:10px;">'
            f'<span style="font-size:0.75rem; color:{NEUTRAL};">'
            f'已用份额 {signal.portions_used}/{signal.portions_total}</span>'
            f'<div style="height:6px; background:#e2e8f0; border-radius:3px; '
            f'margin-top:4px; width:200px;">'
            f'<div style="height:6px; background:{action_color}; border-radius:3px; '
            f'width:{pct}%;"></div></div></div>'
        )

    # Suggested amount
    amount_html = ""
    if signal.suggested_amount > 0:
        amount_html = (
            f'<span style="font-size:0.9rem; color:{DARK};">'
            f'建议金额 <b>¥{signal.suggested_amount:,.0f}</b> '
            f'(约 {signal.suggested_shares} 股)</span>'
        )

    # Price range for execution
    range_html = ""
    if signal.suggested_price_low and signal.suggested_price_high:
        range_html = (
            f'<span style="font-size:0.85rem; color:{NEUTRAL};">'
            f'建议价格区间: <b style="color:{DARK};">'
            f'¥{signal.suggested_price_low:.3f} ~ ¥{signal.suggested_price_high:.3f}</b>'
            f'（当前价 ¥{signal.current_price:.3f}）</span>'
        )

    st.markdown(
        f'<div style="padding:20px 24px; background:{bg}; '
        f'border:2px solid {border}; border-radius:12px; margin:16px 0;">'
        # Top row: icon + action + price
        f'<div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">'
        f'<span style="font-size:2rem;">{icon}</span>'
        f'<span style="font-size:1.3rem; font-weight:700; color:{action_color}; '
        f'font-family:{FONT};">{action_text}</span>'
        f'<span style="font-size:1.5rem; font-weight:700; color:{DARK}; '
        f'font-family:{FONT_MONO};">{price_str}</span>'
        f'</div>'
        # Trigger description
        f'<div style="margin:4px 0 8px 0;">'
        f'<span style="font-size:0.95rem; color:{DARK};">'
        f'{signal.trigger_description}</span>'
        f'</div>'
        # Suggested amount
        f'<div style="margin-bottom:4px;">{amount_html}</div>'
        # Price range
        f'<div style="margin-bottom:4px;">{range_html}</div>'
        # Next trigger
        f'<div style="margin-bottom:4px;">'
        + (f'<span style="font-size:0.85rem; color:{NEUTRAL};">'
           f'下次触发价: <b style="color:{DARK};">{trigger_str}</b></span>'
           if trigger_str else "")
        + f'</div>'
        # Reason
        f'<div style="margin-top:8px;">'
        f'<span style="font-size:0.8rem; color:{NEUTRAL};">{signal.reason}</span>'
        f'</div>'
        # Portions bar
        f'{portions_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dashboard card grid
# ---------------------------------------------------------------------------

def render_dashboard_cards(cards: list[DashboardCard]) -> None:
    """Render strategy-specific info cards in a responsive column grid.

    Card types and their rendering:
      - "metric":   st.metric-style (title + big value + subtitle)
      - "trigger":  price trigger with direction arrow
      - "progress": labeled progress bar
      - "info":     key-value pairs in a compact card
      - "warning":  yellow warning box
    """
    if not cards:
        return

    # Sort by priority
    cards = sorted(cards, key=lambda c: c.priority)

    # Use up to 4 columns
    n = min(len(cards), 4)
    cols = st.columns(n)

    for i, card in enumerate(cards):
        with cols[i % n]:
            _render_one_card(card)


def _render_one_card(card: DashboardCard) -> None:
    """Render a single dashboard card based on its type."""
    content = card.content
    card_type = card.card_type

    with st.container(border=True):
        st.caption(card.title)

        if card_type == "metric":
            value = content.get("value", "—")
            subtitle = content.get("subtitle", "")
            color = content.get("color")
            st.markdown(
                f'<span style="font-size:1.3rem; font-weight:700; '
                + (f'color:{color};' if color else f'color:{DARK};')
                + f'font-family:{FONT_MONO};">{value}</span>',
                unsafe_allow_html=True,
            )
            if subtitle:
                st.caption(subtitle)

        elif card_type == "trigger":
            current = content.get("current_price")
            trigger = content.get("trigger_price")
            direction = content.get("direction", "down")  # "up" or "down"
            drop_needed = content.get("drop_needed_pct")
            rise_needed = content.get("rise_needed_pct")

            arrow = "↓" if direction == "down" else "↑"
            arrow_color = SUCCESS if direction == "down" else DANGER

            if trigger is not None:
                st.markdown(
                    f'<span style="font-size:1.3rem; font-weight:700; '
                    f'color:{DARK}; font-family:{FONT_MONO};">'
                    f'¥{trigger:.3f}</span>',
                    unsafe_allow_html=True,
                )
                if current is not None:
                    diff_pct = abs((trigger - current) / current * 100) if current > 0 else 0
                    st.caption(
                        f'{arrow} 距当前 {diff_pct:.1f}%'
                        f'{"  (已触发)" if diff_pct < 0.5 else ""}'
                    )
            else:
                st.caption("暂无触发价")

        elif card_type == "progress":
            value_pct = float(content.get("value_pct", 0))
            label = content.get("label", "")
            sub = content.get("subtitle", "")
            color = content.get("color", PRIMARY)

            st.progress(min(value_pct / 100, 1.0))
            if label:
                st.caption(label)
            if sub:
                st.caption(sub)

        elif card_type == "warning":
            st.warning(content.get("message", ""))

        elif card_type == "info":
            # Support plan_html for trade timeline cards
            plan_html = content.get("plan_html")
            if plan_html:
                st.markdown(plan_html, unsafe_allow_html=True)

            # Support price levels (key price ladder)
            levels = content.get("levels", [])
            for lv in levels:
                if isinstance(lv, dict):
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:2px 0;">'
                        f'<span style="font-size:0.75rem; color:{NEUTRAL};">{lv.get("label","")}</span>'
                        f'<span style="font-size:0.8rem; font-weight:600; '
                        f'color:{DARK};">{lv.get("price","")} '
                        f'<small style="color:{NEUTRAL};">{lv.get("pct","")}</small></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            items = content.get("items", [])
            for item in items:
                if isinstance(item, dict):
                    k = item.get("label", "")
                    v = item.get("value", "—")
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:2px 0;">'
                        f'<span style="font-size:0.75rem; color:{NEUTRAL};">{k}</span>'
                        f'<span style="font-size:0.8rem; font-weight:600; '
                        f'color:{DARK};">{v}</span></div>',
                        unsafe_allow_html=True,
                    )

        elif card_type == "pe_percentile":
            # PE历史分位卡片 — percentile bar + key stats
            current_pe = content.get("current_pe")
            pe_pct = content.get("pe_percentile")
            pe_mean = content.get("pe_mean")
            pe_plus = content.get("pe_plus_1std")
            pe_minus = content.get("pe_minus_1std")
            zone_label = content.get("zone_label", "")
            zone_color = content.get("zone_color", PRIMARY)
            data_points = content.get("data_points", 0)
            index_name = content.get("index_name", "")

            # Current PE display
            pe_display = f"{current_pe:.2f}" if current_pe is not None else "—"
            st.markdown(
                f'<span style="font-size:1.3rem; font-weight:700; '
                f'color:{DARK}; font-family:{FONT_MONO};">PE(TTM) {pe_display}</span>',
                unsafe_allow_html=True,
            )

            # Percentile bar
            if pe_pct is not None:
                st.caption(f"历史分位 {pe_pct:.1f}%")
                # Color the progress bar by zone
                st.progress(min(pe_pct / 100, 1.0))

            # Zone badge
            st.markdown(
                f'<span style="display:inline-block; padding:2px 8px; '
                f'background:{zone_color}15; color:{zone_color}; '
                f'border-radius:8px; font-size:0.75rem; font-weight:600;">'
                f'{zone_label}</span>',
                unsafe_allow_html=True,
            )

            # Stats
            if pe_mean is not None:
                st.caption(f"均值 {pe_mean:.1f} | +1σ {pe_plus:.1f}" if pe_plus else f"均值 {pe_mean:.1f}")
            if pe_minus is not None:
                st.caption(f"-1σ {pe_minus:.1f}")
            if data_points > 0:
                st.caption(f"数据: {data_points}个交易日")

        elif card_type == "macro_pulse":
            # 宏观情绪卡片 — sentiment bar + module summary + risk badge
            overall = content.get("overall_sentiment", 0.5)
            risk_level = content.get("risk_level", "low")
            risk_color = content.get("risk_color", PRIMARY)
            risk_label = content.get("risk_label", "")
            modules = content.get("modules", [])
            warning = content.get("warning", "")

            # Sentiment score
            st.markdown(
                f'<span style="font-size:1.3rem; font-weight:700; '
                f'color:{risk_color}; font-family:{FONT_MONO};">'
                f'🌡️ {overall:.2f}</span>',
                unsafe_allow_html=True,
            )
            # Risk badge
            if risk_label:
                st.markdown(
                    f'<span style="display:inline-block; padding:2px 8px; '
                    f'background:{risk_color}15; color:{risk_color}; '
                    f'border-radius:8px; font-size:0.75rem; font-weight:600;">'
                    f'{risk_label}</span>',
                    unsafe_allow_html=True,
                )
            # Progress bar
            st.progress(min(max(overall, 0.0), 1.0))

            # Module summary (compact)
            for m in modules[:3]:
                st.caption(f"{m.get('icon','')} {m['label']}: {m['avg']:.2f}")

            # Warning message
            if warning:
                st.warning(warning)

        else:
            # Fallback: just show content as text
            st.write(str(content))
