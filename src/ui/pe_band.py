"""PE Band chart & PE percentile overview for the dashboard.

Renders a Plotly PE Band chart (PE over time with ±1σ bands) and a
compact percentile overview row of metric cards.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from src.data.pe_history import PEPercentile, get_pe_band_data
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
)


# ---------------------------------------------------------------------------
# PE percentile overview — compact row of metric cards
# ---------------------------------------------------------------------------

def render_pe_percentile_overview(pp: PEPercentile) -> None:
    """Render a 5-column overview of PE percentile data above the live signal.

    Cards: 当前PE | PE历史分位 | 历史均值 | 估值区间 | 5年范围
    """
    current_pe = pp.current_pe
    percentile = pp.pe_percentile
    zone_label = pp.zone_label
    zone_color = pp.zone_color

    c1, c2, c3, c4, c5 = st.columns(5)

    # 1. Current PE
    with c1:
        pe_str = f"{current_pe:.2f}" if current_pe is not None else "—"
        st.markdown(
            f'<div style="padding:10px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">当前PE(TTM)</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{DARK}; '
            f'font-family:{FONT_MONO};">{pe_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 2. PE Percentile
    with c2:
        pct_str = f"{percentile:.1f}%" if percentile is not None else "—"
        # Color based on zone
        pct_color = zone_color if zone_color else PRIMARY
        st.markdown(
            f'<div style="padding:10px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">PE历史分位</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{pct_color}; '
            f'font-family:{FONT_MONO};">{pct_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 3. Historical mean
    with c3:
        mean_str = f"{pp.pe_mean:.2f}" if pp.pe_mean is not None else "—"
        median_str = f"{pp.pe_median:.2f}" if pp.pe_median is not None else "—"
        st.markdown(
            f'<div style="padding:10px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">历史均值 / 中位数</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{DARK}; '
            f'font-family:{FONT_MONO};">{mean_str}</div>'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">中位数 {median_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 4. Valuation zone
    with c4:
        st.markdown(
            f'<div style="padding:10px; background:{zone_color}15; '
            f'border:2px solid {zone_color}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.65rem; color:{zone_color};">估值区间</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{zone_color}; '
            f'font-family:{FONT};">{zone_label}</div>'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">'
            f'{pp.index_name} | {pp.data_points}个交易日'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 5. 5-year range
    with c5:
        min5_str = f"{pp.pe_min_5yr:.2f}" if pp.pe_min_5yr is not None else "—"
        max5_str = f"{pp.pe_max_5yr:.2f}" if pp.pe_max_5yr is not None else "—"
        st.markdown(
            f'<div style="padding:10px; background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">近5年PE范围</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{DARK}; '
            f'font-family:{FONT_MONO};">{min5_str} — {max5_str}</div>'
            f'<div style="font-size:0.65rem; color:{NEUTRAL};">{pp.date_range}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PE Band chart
# ---------------------------------------------------------------------------


def render_pe_band(etf_code: str, pp: PEPercentile | None = None) -> None:
    """Render a Plotly PE Band chart: PE(TTM) over time with ±1σ bands.

    Args:
        etf_code: ETF code (used to load PE history).
        pp: Optional pre-computed PEPercentile for annotation.
    """
    df = get_pe_band_data(etf_code)
    if df is None or df.empty:
        return

    if "pe_ttm" not in df.columns:
        return

    pe_col = "pe_ttm"
    df = df.dropna(subset=[pe_col]).copy()
    if len(df) < 50:
        return

    fig = go.Figure()

    # ── PE(TTM) line ──
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[pe_col],
        mode="lines",
        name="PE(TTM)",
        line=dict(color="#3b82f6", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>PE(TTM): %{y:.2f}<extra></extra>",
    ))

    # ── Mean line ──
    if "pe_ttm_mean" in df.columns and df["pe_ttm_mean"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["pe_ttm_mean"],
            mode="lines",
            name="历史均值",
            line=dict(color="#6b7280", width=1, dash="dash"),
            hovertemplate="均值: %{y:.2f}<extra></extra>",
        ))

    # ── +1σ band ──
    if "pe_ttm_plus_1std" in df.columns and df["pe_ttm_plus_1std"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["pe_ttm_plus_1std"],
            mode="lines",
            name="+1σ (高估线)",
            line=dict(color="#ef4444", width=1, dash="dot"),
            hovertemplate="+1σ: %{y:.2f}<extra></extra>",
        ))

    # ── -1σ band (computed from mean) ──
    if "pe_ttm_mean" in df.columns and df["pe_ttm_mean"].notna().any():
        pe_mean = df["pe_ttm_mean"].mean()
        pe_std = df[pe_col].std()
        minus_1std = pe_mean - pe_std
        fig.add_hline(
            y=minus_1std,
            line=dict(color="#22c55e", width=1, dash="dot"),
            annotation_text="-1σ (低估线)",
            annotation_position="bottom right",
        )

    # ── Fill between mean and ±1σ ──
    if "pe_ttm_mean" in df.columns and "pe_ttm_plus_1std" in df.columns:
        # Green fill below mean
        pe_mean_vals = df["pe_ttm_mean"].values
        pe_minus_1std_vals = pe_mean_vals - df[pe_col].std()
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=list(pe_mean_vals) + list(pe_minus_1std_vals[::-1]),
            fill="toself",
            fillcolor="rgba(34,197,94,0.08)",
            line=dict(width=0),
            name="低估区",
            showlegend=True,
            hoverinfo="skip",
        ))
        # Red fill above mean
        if df["pe_ttm_plus_1std"].notna().any():
            pe_plus_vals = df["pe_ttm_plus_1std"].values
            fig.add_trace(go.Scatter(
                x=pd.concat([df["date"], df["date"][::-1]]),
                y=list(pe_plus_vals) + list(pe_mean_vals[::-1]),
                fill="toself",
                fillcolor="rgba(239,68,68,0.08)",
                line=dict(width=0),
                name="高估区",
                showlegend=True,
                hoverinfo="skip",
            ))

    # ── Current PE marker ──
    if pp is not None and pp.current_pe is not None and len(df) > 0:
        latest_date = df["date"].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[latest_date],
            y=[pp.current_pe],
            mode="markers",
            name=f"当前PE: {pp.current_pe:.2f}",
            marker=dict(
                color=pp.zone_color if pp.zone_color else "#3b82f6",
                size=10,
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            hovertemplate=(
                f"当前PE: {pp.current_pe:.2f}<br>"
                f"分位: {pp.pe_percentile:.1f}%<br>"
                f"区间: {pp.zone_label}<extra></extra>"
            ),
        ))

    # ── Layout ──
    fig.update_layout(
        title=f"PE Band · {pp.index_name if pp else etf_code}",
        xaxis_title="日期",
        yaxis_title="PE(TTM)",
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)
