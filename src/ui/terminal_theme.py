"""Retro-Futuristic Terminal Design System for Streamlit.

Anchor: Retro-Futuristic — CRT scanlines, phosphor-green data, neon accents.
Tokens locked: Space Mono + VT323, #0A0014 surface, #00FF41/#FF006E/#FFB000 accents.

Injects global CSS that restyles Streamlit's entire chrome into a synthwave
trading terminal.  Plotly charts get a matching dark template.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# =========================================================================
# Token Palette — Retro-Futuristic
# =========================================================================

SURFACE = "#0A0014"  # Deep navy-black CRT off-state
PANEL = "#10031F"  # Slightly lighter panel bg
CARD = "#1A0A2E"  # Card / widget bg
BORDER = "#2D1B4E"  # Glow-border base
TEXT = "#E0D0F0"  # Muted lavender — readable on dark
TEXT_DIM = "#8060A0"  # Secondary text

# Accents
GREEN = "#00FF41"  # Phosphor green — price UP, buy, profit
MAGENTA = "#FF006E"  # Neon magenta — price DOWN, sell, loss
AMBER = "#FFB000"  # Amber — warnings, alerts
CYAN = "#00FFFF"  # Cyan — nav active, links, emphasis
VIOLET = "#A855F7"  # Violet — chart line accent

# Semantic
UP_COLOR = GREEN
DOWN_COLOR = MAGENTA

# Chart palette (4 distinct for overlays)
CHART_COLORS = [GREEN, CYAN, AMBER, VIOLET]

# MA line colors
MA_COLORS = {"ma5": AMBER, "ma10": CYAN, "ma20": VIOLET}


# =========================================================================
# Typography
# =========================================================================

FONT_MONO = "'Space Mono', 'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace"
FONT_HEADER = "'VT323', 'Courier New', monospace"
FONT_SYSTEM = "system-ui, -apple-system, sans-serif"


# =========================================================================
# Master CSS Injection
# =========================================================================

_CSS_INJECTED = False


def inject_terminal_css() -> None:
    """Inject the full Retro-Futuristic CSS override once per session.

    Restyles every Streamlit chrome element: sidebar, buttons, inputs,
    dataframes, expanders, metrics — everything.
    """
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    _CSS_INJECTED = True

    css = f"""
    <style>
    /* ============================================================
       GLOBAL RESET -- CRT Surface
       ============================================================ */
    body, .stApp, .main, [data-testid="stAppViewContainer"] {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        font-family: {FONT_MONO} !important;
    }}

    /* CRT Scanline Overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            rgba(0, 255, 65, 0.015) 0px,
            rgba(0, 255, 65, 0.015) 1px,
            transparent 1px,
            transparent 3px
        );
        pointer-events: none;
        z-index: 9999;
    }}

    /* ============================================================
       SIDEBAR -- Terminal Nav Panel
       ============================================================ */
    [data-testid="stSidebar"] {{
        background: {PANEL} !important;
        border-right: 1px solid {BORDER} !important;
        box-shadow: 0 0 30px rgba(0, 255, 65, 0.03) !important;
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {{
        color: {TEXT} !important;
        font-family: {FONT_MONO} !important;
    }}

    /* ============================================================
       INPUTS -- Phosphor-Glow Edges
       ============================================================ */
    input, textarea, .stTextInput input, .stSelectbox select,
    [data-baseweb="input"], [data-baseweb="select"] {{
        background: {CARD} !important;
        color: {GREEN} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 0 !important;
        font-family: {FONT_MONO} !important;
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.08) !important;
    }}
    input:focus, textarea:focus, [data-baseweb="input"]:focus {{
        border-color: {GREEN} !important;
        box-shadow: 0 0 16px rgba(0, 255, 65, 0.18) !important;
        outline: none !important;
    }}

    /* ============================================================
       BUTTONS -- Phosphor Activate
       ============================================================ */
    .stButton > button, [data-testid="baseButton-primary"] {{
        background: transparent !important;
        color: {GREEN} !important;
        border: 1px solid {GREEN} !important;
        border-radius: 0 !important;
        font-family: {FONT_MONO} !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 0 12px rgba(0, 255, 65, 0.15) !important;
        transition: all 0.15s !important;
    }}
    .stButton > button:hover, [data-testid="baseButton-primary"]:hover {{
        background: rgba(0, 255, 65, 0.08) !important;
        box-shadow: 0 0 24px rgba(0, 255, 65, 0.30) !important;
    }}

    /* ============================================================
       METRICS -- Large Phosphor Digits
       ============================================================ */
    [data-testid="stMetricValue"] {{
        font-family: {FONT_MONO} !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: {GREEN} !important;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.20) !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: {FONT_MONO} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: {FONT_MONO} !important;
        color: {TEXT_DIM} !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}

    /* ============================================================
       DATAFRAMES -- Terminal Table
       ============================================================ */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 0 !important;
    }}
    [data-testid="stDataFrame"] th {{
        background: {CARD} !important;
        color: {CYAN} !important;
        font-family: {FONT_MONO} !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-bottom: 1px solid {BORDER} !important;
    }}
    [data-testid="stDataFrame"] td {{
        background: {PANEL} !important;
        color: {TEXT} !important;
        font-family: {FONT_MONO} !important;
        font-size: 0.75rem !important;
        border-bottom: 1px solid rgba(45, 27, 78, 0.3) !important;
    }}

    /* ============================================================
       TABS -- Terminal Nav
       ============================================================ */
    [data-testid="stTabs"] button {{
        background: {PANEL} !important;
        color: {TEXT_DIM} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 0 !important;
        font-family: {FONT_MONO} !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        background: {CARD} !important;
        color: {CYAN} !important;
        border-bottom: 2px solid {CYAN} !important;
    }}

    /* ============================================================
       EXPANDERS
       ============================================================ */
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 0 !important;
        background: {PANEL} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {TEXT} !important;
        font-family: {FONT_MONO} !important;
    }}

    /* ============================================================
       RADIO / CHECKBOX -- Phosphor Dots
       ============================================================ */
    .stRadio label, .stCheckbox label {{
        font-family: {FONT_MONO} !important;
        color: {TEXT} !important;
    }}
    .stRadio [data-testid="stMarkdownContainer"] p {{
        font-family: {FONT_MONO} !important;
    }}

    /* ============================================================
       SELECTBOX
       ============================================================ */
    .stSelectbox label {{
        font-family: {FONT_MONO} !important;
        color: {TEXT_DIM} !important;
    }}

    /* ============================================================
       DIVIDER -- Glow Line
       ============================================================ */
    hr, [data-testid="stDivider"] {{
        border-color: {BORDER} !important;
        box-shadow: 0 0 6px rgba(0, 255, 65, 0.06) !important;
    }}

    /* ============================================================
       ALERTS -- Terminal Status Banners
       ============================================================ */
    [data-testid="stInfo"] {{
        background: {CARD} !important;
        border-left: 3px solid {CYAN} !important;
        color: {TEXT} !important;
        font-family: {FONT_MONO} !important;
    }}
    [data-testid="stWarning"] {{
        background: rgba(255, 176, 0, 0.06) !important;
        border-left: 3px solid {AMBER} !important;
        color: {AMBER} !important;
        font-family: {FONT_MONO} !important;
    }}
    [data-testid="stError"] {{
        background: rgba(255, 0, 110, 0.06) !important;
        border-left: 3px solid {MAGENTA} !important;
        color: {MAGENTA} !important;
        font-family: {FONT_MONO} !important;
    }}
    [data-testid="stSuccess"] {{
        background: rgba(0, 255, 65, 0.06) !important;
        border-left: 3px solid {GREEN} !important;
        color: {TEXT} !important;
        font-family: {FONT_MONO} !important;
    }}

    /* ============================================================
       SCROLLBAR -- Thin Phosphor
       ============================================================ */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {PANEL}; }}
    ::-webkit-scrollbar-thumb {{
        background: {BORDER};
        border-radius: 0;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {GREEN}; }}

    /* ============================================================
       TITLE & HEADERS
       ============================================================ */
    h1, h2, h3 {{
        font-family: {FONT_MONO} !important;
        color: {TEXT} !important;
        letter-spacing: 0 !important;
    }}
    h1 {{ font-size: 1.3rem !important; text-transform: uppercase !important; }}
    h2 {{ font-size: 1.1rem !important; }}
    h3 {{ font-size: 0.95rem !important; color: {TEXT_DIM} !important; }}

    /* ============================================================
       CAPTION / SMALL TEXT
       ============================================================ */
    .stCaption, caption, small {{
        font-family: {FONT_MONO} !important;
        color: {TEXT_DIM} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================================================================
# HTML Component Generators
# =========================================================================

def ticker_header(symbols: list[dict]) -> None:
    """Render a horizontal scrolling ticker tape of ETF prices.

    Args:
        symbols: [{"code": "510300", "name": "沪深300ETF", "price": 4.907, "change_pct": -2.79}, ...]
    """
    items = "".join(
        f'<span style="margin:0 24px;">'
        f'<span style="color:{TEXT_DIM};">{s["code"]}</span> '
        f'<span style="color:{TEXT};">{s.get("name","")}</span> '
        f'<span style="color:{GREEN if s.get("change_pct",0)>=0 else MAGENTA};">'
        f'{s.get("price","—")} '
        f'({"+" if s.get("change_pct",0)>=0 else ""}{s.get("change_pct",0):.2f}%)'
        f'</span></span>'
        for s in symbols
    )
    # Duplicate for seamless loop
    tape = items + items

    html = f"""
    <div style="
        width:100%; overflow:hidden; background:{PANEL};
        border-bottom:1px solid {BORDER};
        padding:6px 0; position:relative;
    ">
        <div style="
            display:inline-block; white-space:nowrap;
            animation: ticker-scroll 40s linear infinite;
            font-family:{FONT_MONO}; font-size:0.7rem;
            letter-spacing:0.5px;
        ">{tape}</div>
    </div>
    <style>
        @keyframes ticker-scroll {{
            0%   {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_header(label: str) -> None:
    """Render a terminal-style section header."""
    st.markdown(
        f'<p style="font-family:{FONT_MONO}; font-size:0.65rem; '
        f'color:{TEXT_DIM}; text-transform:uppercase; letter-spacing:2px; '
        f'margin:16px 0 4px 0; border-bottom:1px solid {BORDER}; '
        f'padding-bottom:4px;">{label}</p>',
        unsafe_allow_html=True,
    )


def price_display(price: float, change_pct: float | None) -> None:
    """Big phosphor price with glow."""
    color = GREEN if (change_pct or 0) >= 0 else MAGENTA
    sign = "+" if (change_pct or 0) >= 0 else ""
    pct_str = f"{sign}{change_pct:.2f}%" if change_pct is not None else "—"

    st.markdown(
        f'<div style="font-family:{FONT_MONO};">'
        f'<span style="font-size:2.4rem; font-weight:700; color:{color}; '
        f'text-shadow:0 0 20px {color}44;">{price:.3f}</span>'
        f'<span style="font-size:1.1rem; color:{color}; margin-left:12px; '
        f'text-shadow:0 0 10px {color}33;">{pct_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, kind: str = "info") -> None:
    """Render a small phosphor status indicator."""
    colors = {
        "info": (CYAN, f"rgba(0,255,255,0.08)"),
        "success": (GREEN, f"rgba(0,255,65,0.08)"),
        "warning": (AMBER, f"rgba(255,176,0,0.08)"),
        "danger": (MAGENTA, f"rgba(255,0,110,0.08)"),
    }
    c, bg = colors.get(kind, colors["info"])
    st.markdown(
        f'<span style="display:inline-block; font-family:{FONT_MONO}; '
        f'font-size:0.65rem; color:{c}; background:{bg}; '
        f'border:1px solid {c}44; padding:2px 8px; '
        f'text-transform:uppercase; letter-spacing:1px;">{text}</span>',
        unsafe_allow_html=True,
    )


# =========================================================================
# Plotly Dark Chart Theme
# =========================================================================

def apply_terminal_chart(fig: go.Figure, height: int = 500) -> None:
    """Apply Retro-Futuristic dark theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_MONO, color=TEXT, size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(family=FONT_MONO, size=10, color=TEXT_DIM),
        ),
    )
    fig.update_xaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        color=TEXT_DIM,
    )
    fig.update_yaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        color=TEXT_DIM,
    )
