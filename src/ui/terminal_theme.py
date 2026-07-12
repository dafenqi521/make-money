"""Clean professional light theme for Streamlit — Chinese financial dashboard.

No dark backgrounds.  Fluid, modern, card-based aesthetic.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# =========================================================================
# Palette
# =========================================================================

PRIMARY = "#2563eb"  # Blue-600 — buttons, links, active
SUCCESS = "#16a34a"  # Green-600 — price up, buy, profit
DANGER = "#dc2626"  # Red-600 — price down, sell, loss
WARNING = "#f59e0b"  # Amber-500 — alerts
INFO = "#0891b2"  # Cyan-700 — info
NEUTRAL = "#64748b"  # Slate-500 — secondary text
DARK = "#1e293b"  # Slate-800 — primary text

BG_PAGE = "#ffffff"
BG_CARD = "#f8fafc"
BORDER = "#e2e8f0"

UP_COLOR = SUCCESS
DOWN_COLOR = DANGER

CHART_COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#dc2626", "#0891b2"]
MA_COLORS = {"ma5": "#f59e0b", "ma10": "#0891b2", "ma20": "#7c3aed"}

FONT = "'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"
FONT_MONO = "'SF Mono', 'Cascadia Code', 'Consolas', monospace"


# =========================================================================
# CSS injection
# =========================================================================

_CSS_INJECTED = False


def inject_css() -> None:
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    _CSS_INJECTED = True

    st.markdown(f"""
    <style>
    /* ── Page background ── */
    .stApp, .main, [data-testid="stAppViewContainer"] {{
        background: {BG_PAGE};
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: {BG_CARD};
        border-right: 1px solid {BORDER};
    }}

    /* ── Inputs ── */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        font-family: {FONT} !important;
    }}
    input:focus, [data-baseweb="input"]:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 8px !important;
        font-family: {FONT} !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.2);
    }}

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {{
        font-family: {FONT} !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: {FONT} !important;
        color: {NEUTRAL} !important;
        font-size: 0.75rem !important;
    }}

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    [data-testid="stDataFrame"] th {{
        background: {BG_CARD} !important;
        color: {DARK} !important;
        font-family: {FONT} !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
    }}
    [data-testid="stDataFrame"] td {{
        font-family: {FONT} !important;
        font-size: 0.8rem !important;
    }}

    /* ── Tabs — fluid switching ── */
    [data-testid="stTabs"] {{
        margin-top: 4px;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        font-family: {FONT} !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        border: none !important;
        color: {NEUTRAL} !important;
        background: transparent !important;
        transition: color 0.2s, border-color 0.2s !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        color: {DARK} !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
        color: {PRIMARY} !important;
        border-bottom: 2px solid {PRIMARY} !important;
        background: transparent !important;
    }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    /* ── Alerts ── */
    [data-testid="stInfo"]    {{ border-radius: 8px !important; }}
    [data-testid="stWarning"] {{ border-radius: 8px !important; }}
    [data-testid="stError"]   {{ border-radius: 8px !important; }}
    [data-testid="stSuccess"] {{ border-radius: 8px !important; }}

    /* ── Divider ── */
    hr {{ border-color: {BORDER} !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

    /* ── Smooth page transitions ── */
    .main .block-container {{
        animation: fadeIn 0.25s ease-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """, unsafe_allow_html=True)


# =========================================================================
# Plotly helper
# =========================================================================

def apply_chart_theme(fig: go.Figure, height: int = 500) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG_PAGE,
        plot_bgcolor=BG_CARD,
        font=dict(family=FONT, color=DARK, size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(family=FONT, size=10, color=NEUTRAL),
        ),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=NEUTRAL)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=NEUTRAL)


# =========================================================================
# Helper
# =========================================================================

import pandas as pd


def _styler_apply(styled: "pd.io.formats.style.Styler", func, subset: list[str]) -> "pd.io.formats.style.Styler":
    """Apply func to Styler columns, using map (pandas>=2.1) or applymap (2.0)."""
    if hasattr(styled, "map"):
        return styled.map(func, subset=subset)
    return styled.applymap(func, subset=subset)


def section_header(label: str) -> None:
    st.markdown(
        f'<p style="font-family:{FONT}; font-size:0.75rem; color:{NEUTRAL}; '
        f'font-weight:600; margin:20px 0 4px 0; '
        f'text-transform:uppercase; letter-spacing:0.5px;">{label}</p>',
        unsafe_allow_html=True,
    )
