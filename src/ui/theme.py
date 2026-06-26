"""Unified design system — colors, chart helpers, CSS, reusable UI components.

Single source of truth for all visual constants.  Every hardcoded hex
color and duplicated ``fig.update_layout()`` call in the project should
be replaced by a reference to this module.
"""

from __future__ import annotations

import streamlit as st

# =========================================================================
# Color palette
# =========================================================================

PRIMARY = "#1a56db"  # Deep blue — trust, finance, buttons
SUCCESS = "#059669"  # Emerald green — price up, buy, profit
DANGER = "#dc2626"  # Red — price down, sell, loss, drawdown
WARNING = "#f59e0b"  # Amber — PE warnings, risk alerts
NEUTRAL = "#64748b"  # Slate gray — secondary text, captions
DARK = "#1e293b"  # Slate-800 — primary text
BG_CARD = "#ffffff"  # White — card backgrounds
BG_PAGE = "#f8f9fb"  # Cool gray — page background
BORDER = "#e2e8f0"  # Slate-200 — subtle borders

# Semantic aliases (used by dashboard & strategy charts)
UP_COLOR = SUCCESS
DOWN_COLOR = DANGER
UP_BG = "#ecfdf5"  # Emerald-50 — buy row highlight
DOWN_BG = "#fef2f2"  # Red-50 — sell row highlight

# Chart color sequence (high contrast, colorblind-friendly)
CHART_COLORS = [
    "#1a56db",  # blue
    "#059669",  # green
    "#f59e0b",  # amber
    "#7c3aed",  # violet
    "#dc2626",  # red
    "#0891b2",  # cyan
]

# MA line colors (consistent across all charts)
MA_COLORS = {
    "ma5": "#f59e0b",  # amber
    "ma10": "#0891b2",  # cyan
    "ma20": "#7c3aed",  # violet
}


# =========================================================================
# Plotly helpers
# =========================================================================

def chart_layout(
    height: int = 500,
    showlegend: bool = True,
    margin: dict | None = None,
) -> dict:
    """Return a standard plotly layout dict for ETF charts.

    Usage::

        fig = make_subplots(...)
        fig.update_layout(**chart_layout(height=600))
    """
    if margin is None:
        margin = dict(l=0, r=0, t=30, b=0)

    return dict(
        height=height,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=DARK),
        xaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
        ),
        yaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
        ),
    )


def apply_chart_theme(fig) -> None:
    """Apply the standard theme to an already-created Plotly figure.

    Sets transparent backgrounds, standard font, and grid colors on all
    axes.  Call this *after* building traces.

    Usage::

        fig = go.Figure()
        fig.add_trace(...)
        apply_chart_theme(fig)
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=DARK),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)


def make_equity_colors(n: int = 4) -> list[str]:
    """Return *n* distinct colors for overlaid equity curves."""
    return CHART_COLORS[:n]


# =========================================================================
# CSS injection
# =========================================================================

_CSS_INJECTED = False


def inject_css() -> None:
    """Inject custom CSS once per session.

    Idempotent — safe to call in every page render.
    """
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    _CSS_INJECTED = True

    css = f"""
    <style>
    /* ── Metric cards ── */
    .theme-card {{
        border-radius: 10px;
        padding: 16px 20px;
        background: {BG_CARD};
        border: 1px solid {BORDER};
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }}
    .theme-card-highlight {{
        border-left: 4px solid {PRIMARY};
        background: #f0f5ff;
    }}

    /* ── Metric value emphasis ── */
    [data-testid="stMetricValue"] {{
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }}

    /* ── Sidebar polish ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 100%);
        border-right: 1px solid {BORDER};
    }}

    /* ── Section headers ── */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {DARK};
        margin-top: 8px;
        margin-bottom: 4px;
    }}
    .section-subtitle {{
        font-size: 0.8rem;
        color: {NEUTRAL};
        margin-bottom: 12px;
    }}

    /* ── Bid/Ask row indicators ── */
    .bid-indicator {{
        border-left: 3px solid {SUCCESS};
        padding-left: 6px;
    }}
    .ask-indicator {{
        border-left: 3px solid {DANGER};
        padding-left: 6px;
    }}

    /* ── Dataframe polish ── */
    [data-testid="stDataFrame"] {{
        border-radius: 8px;
        border: 1px solid {BORDER};
    }}

    /* ── Expander polish ── */
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================================================================
# Reusable UI components
# =========================================================================

def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a clean section header with optional subtitle.

    Replaces the old ``st.subheader(f"📊 {name}")`` pattern.
    """
    st.markdown(
        f'<p class="section-header">{title}</p>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<p class="section-subtitle">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    help: str | None = None,
    highlight: bool = False,
) -> None:
    """Render a single metric inside a themed card.

    Wraps ``st.metric`` with a border+shadow card for visual grouping.
    Set ``highlight=True`` to add a blue left-border accent.
    """
    card_class = "theme-card theme-card-highlight" if highlight else "theme-card"
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.metric(label=label, value=value, delta=delta, help=help)
    st.markdown("</div>", unsafe_allow_html=True)


def metric_row(metrics: list[dict]) -> None:
    """Render a row of 4 metric cards.

    Args:
        metrics: List of dicts with keys matching ``st.metric`` kwargs:
            label, value, delta, help (all optional except label & value).
            Must have exactly 4 items (or fewer — remaining cols are empty).
    """
    cols = st.columns(4)
    for i, m in enumerate(metrics[:4]):
        with cols[i]:
            st.metric(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                delta=m.get("delta"),
                help=m.get("help"),
            )


def info_banner(message: str, kind: str = "info") -> None:
    """Render a styled banner (info / warning / error / success)."""
    colors = {
        "info": (PRIMARY, "#f0f5ff"),
        "warning": (WARNING, "#fffbeb"),
        "error": (DANGER, "#fef2f2"),
        "success": (SUCCESS, "#ecfdf5"),
    }
    border, bg = colors.get(kind, colors["info"])
    st.markdown(
        f'<div style="border-left:4px solid {border}; background:{bg}; '
        f'padding:12px 16px; border-radius:6px; margin:8px 0; '
        f'font-size:0.9rem;">{message}</div>',
        unsafe_allow_html=True,
    )
