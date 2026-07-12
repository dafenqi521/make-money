"""Strategy-aware ETF screener — score & rank ETFs by strategy fit.

Each strategy has its own scoring rubric. The screener batch-fetches
fundamental data for all available ETFs, scores them, and returns a
ranked DataFrame for display.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from src.data.fetcher import get_available_etfs, fetch_multi_etf_info
from src.strategy.base import BaseStrategy
from src.ui.terminal_theme import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, DARK,
    BG_CARD, BORDER, FONT, FONT_MONO,
    _styler_apply,
)

# ---------------------------------------------------------------------------
# Scoring logic per strategy
# ---------------------------------------------------------------------------

# Broad-market index keywords (宽基指数) — preferred for 定投 strategies
_BROAD_MARKET_KEYWORDS = [
    "沪深300", "中证500", "中证1000", "上证50", "上证180",
    "创业板", "科创50", "科创100", "深证100", "中证100",
    "中证红利", "中证A50", "A50", "MSCI", "标普500", "纳指",
    "恒生", "H股", "中国互联", "中概",
]

# Wide ETF classification — ETFs with "ETF" in name are usually index trackers
_ETF_INDICATORS = ["ETF", "指数", "指数基金"]


def _has_keywords(name: str, keywords: list[str]) -> int:
    """Count how many keyword groups match the ETF name."""
    if not isinstance(name, str):
        return 0
    return sum(1 for kw in keywords if kw in name)


def _score_for_four_percent_dca(info: dict) -> float:
    """Score an ETF for 雷牛牛 4%定投法.

    Criteria (total 100):
      - PE available + in buy zone:  30 pts
      - PE available + reasonable:   15 pts
      - Broad-market index (宽基):   15 pts
      - Turnover > 2% (active):      15 pts
      - Market cap > 50亿 (stable):  10 pts
      - Amplitude 2-5% (sweet spot): 10 pts
      - Price 1-10元 (accessible):    10 pts
      - ETF/index in name:            5 pts
      - PB < 2 (value):               5 pts
    """
    score = 0.0
    name = info.get("name", "")

    # 1. PE valuation (most important — the "1" in 1+1+4)
    pe = info.get("pe_ttm") or info.get("pe_static")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 30  # Buy zone!
        elif pe < 20:
            score += 20  # Reasonable
        elif pe < 30:
            score += 10  # Acceptable
        # else: PE too high, no points
    # (PE=None: no penalty, just no bonus — common for ETFs)

    # 2. Broad-market index (宽基指数优先 — the other "1" in 1+1+4)
    broad_hits = _has_keywords(name, _BROAD_MARKET_KEYWORDS)
    score += min(broad_hits * 8, 15)

    # 3. Liquidity — turnover rate
    turnover = info.get("turnover_pct")
    if turnover is not None:
        if turnover > 5:
            score += 15
        elif turnover > 2:
            score += 12
        elif turnover > 1:
            score += 8
        elif turnover > 0.3:
            score += 4

    # 4. Market cap — stability
    mcap = info.get("mcap_yi")
    if mcap is not None:
        if mcap > 500:
            score += 10
        elif mcap > 100:
            score += 8
        elif mcap > 50:
            score += 5
        elif mcap > 10:
            score += 3

    # 5. Amplitude — sweet spot for 4% triggers
    amp = info.get("amplitude")
    if amp is not None:
        if 2 <= amp <= 5:
            score += 10
        elif 1 <= amp < 2 or 5 < amp <= 8:
            score += 6
        elif amp > 0:
            score += 3

    # 6. Price accessibility
    price = info.get("current_price")
    if price is not None:
        if 1 <= price <= 10:
            score += 10
        elif 0.5 <= price < 1 or 10 < price <= 20:
            score += 6
        elif price > 0:
            score += 3

    # 7. ETF/index indicator
    if _has_keywords(name, _ETF_INDICATORS) > 0:
        score += 5

    # 8. PB — value check
    pb = info.get("pb")
    if pb is not None and pb > 0:
        if pb < 1.5:
            score += 5
        elif pb < 3:
            score += 3

    return min(score, 100)


def _score_for_value_averaging(info: dict) -> float:
    """Similar to 4% DCA but weights PE even higher."""
    score = _score_for_four_percent_dca(info)
    # Value Averaging cares most about PE zone
    pe = info.get("pe_ttm") or info.get("pe_static")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 10  # Extra boost for low PE
        elif pe > 30:
            score -= 20  # Penalty for high PE
    return max(score, 0)


def _score_for_trend_following(info: dict) -> float:
    """Score for trend following — prioritizes liquidity & momentum.

    Criteria:
      - Turnover > 3% (liquidity):   25 pts
      - Volume ratio > 1 (active):   20 pts
      - Amplitude > 3% (trendy):     20 pts
      - Market cap > 100亿 (stable): 15 pts
      - Broad-market index:          10 pts
      - Price 1-20元:                10 pts
    """
    score = 0.0
    name = info.get("name", "")

    turnover = info.get("turnover_pct")
    if turnover is not None:
        if turnover > 5:
            score += 25
        elif turnover > 3:
            score += 20
        elif turnover > 1:
            score += 12
        elif turnover > 0.3:
            score += 6

    vol_ratio = info.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio > 2:
            score += 20
        elif vol_ratio > 1:
            score += 15
        elif vol_ratio > 0.5:
            score += 8

    amp = info.get("amplitude")
    if amp is not None:
        if amp > 5:
            score += 20
        elif amp > 3:
            score += 15
        elif amp > 1:
            score += 8

    mcap = info.get("mcap_yi")
    if mcap is not None:
        if mcap > 100:
            score += 15
        elif mcap > 50:
            score += 10
        elif mcap > 10:
            score += 5

    score += min(_has_keywords(name, _BROAD_MARKET_KEYWORDS) * 5, 10)

    price = info.get("current_price")
    if price is not None and 1 <= price <= 20:
        score += 10
    elif price is not None and price > 0:
        score += 5

    return min(score, 100)


def _score_for_grid_trading(info: dict) -> float:
    """Score for grid trading — likes range-bound, liquid ETFs.

    Criteria:
      - Turnover > 2% (liquidity):  25 pts
      - Amplitude 3-8% (gridable):  25 pts
      - Market cap > 50亿:          15 pts
      - Moderate price (1-20元):    15 pts
      - Volume ratio near 1:        10 pts
      - Broad-market index:         10 pts
    """
    score = 0.0
    name = info.get("name", "")

    turnover = info.get("turnover_pct")
    if turnover is not None:
        if turnover > 3:
            score += 25
        elif turnover > 2:
            score += 20
        elif turnover > 1:
            score += 12
        elif turnover > 0.3:
            score += 6

    amp = info.get("amplitude")
    if amp is not None:
        if 3 <= amp <= 8:
            score += 25  # Sweet spot for grid
        elif 2 <= amp < 3 or 8 < amp <= 10:
            score += 18
        elif 1 <= amp < 2:
            score += 10
        elif amp > 10:
            score += 5  # Too volatile

    mcap = info.get("mcap_yi")
    if mcap is not None:
        if mcap > 100:
            score += 15
        elif mcap > 50:
            score += 10
        elif mcap > 10:
            score += 5

    price = info.get("current_price")
    if price is not None:
        if 1 <= price <= 10:
            score += 15
        elif 10 < price <= 20:
            score += 10
        elif price > 0:
            score += 5

    vol_ratio = info.get("vol_ratio")
    if vol_ratio is not None:
        if 0.8 <= vol_ratio <= 1.2:
            score += 10
        elif 0.5 <= vol_ratio <= 1.5:
            score += 5

    score += min(_has_keywords(name, _BROAD_MARKET_KEYWORDS) * 5, 10)

    return min(score, 100)


def _score_for_hybrid(info: dict) -> float:
    """Average of DCA and Grid scores since Hybrid combines both."""
    dca = _score_for_value_averaging(info)
    grid = _score_for_grid_trading(info)
    return (dca * 0.6 + grid * 0.4)  # Weighted toward DCA


# Strategy → scorer mapping
_SCORERS = {
    "4%定投法": _score_for_four_percent_dca,
    "估值定投": _score_for_value_averaging,
    "趋势跟随": _score_for_trend_following,
    "网格交易": _score_for_grid_trading,
    "网格+定投": _score_for_hybrid,
}


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------

def screen_etfs(
    strategy_name: str,
    top_n: int = 20,
    batch_size: int = 80,
    progress_bar=None,
    status_text=None,
) -> pd.DataFrame:
    """Screen all available ETFs and rank by strategy fit.

    Args:
        strategy_name: Chinese strategy name (must be in _SCORERS).
        top_n: Return top N results.
        batch_size: Codes per Tencent API call (max ~200 before URL too long).
        progress_bar: Optional streamlit progress bar widget.
        status_text: Optional streamlit text element for status updates.

    Returns:
        DataFrame with columns: 代码, 名称, 评分, 最新价, PE(TTM), PB,
        总市值(亿), 换手率, 振幅, 涨跌幅, 成交量.
        Sorted by 评分 descending. Empty DataFrame on failure.
    """
    scorer = _SCORERS.get(strategy_name, _score_for_four_percent_dca)

    # 1. Get ETF list
    if status_text is not None:
        status_text.text("正在获取ETF列表...")
    try:
        etf_list = get_available_etfs()
    except Exception:
        return pd.DataFrame()

    if etf_list.empty:
        return pd.DataFrame()

    # Normalize columns — AKShare may return different column names
    code_col = "代码" if "代码" in etf_list.columns else etf_list.columns[0]
    name_col = "名称" if "名称" in etf_list.columns else None
    codes = etf_list[code_col].dropna().tolist()

    if not codes:
        return pd.DataFrame()

    # 2. Batch-fetch info
    all_info: dict[str, dict] = {}
    total_batches = (len(codes) + batch_size - 1) // batch_size

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        batch_num = i // batch_size + 1

        if status_text is not None:
            status_text.text(f"正在获取行情数据... ({batch_num}/{total_batches})")
        if progress_bar is not None:
            progress_bar.progress(min(batch_num / total_batches, 1.0))

        try:
            batch_info = fetch_multi_etf_info(batch)
            all_info.update(batch_info)
        except Exception:
            continue

        # Brief pause between batches to avoid rate limiting
        if batch_num < total_batches:
            time.sleep(0.3)

    if not all_info:
        return pd.DataFrame()

    # 3. Score each ETF
    if status_text is not None:
        status_text.text("正在计算策略适配评分...")

    rows = []
    for code, info in all_info.items():
        name = info.get("name", "")
        score = scorer(info)

        rows.append({
            "代码": code,
            "名称": name,
            "评分": round(score, 0),
            "最新价": info.get("current_price"),
            "PE(TTM)": info.get("pe_ttm"),
            "PB": info.get("pb"),
            "总市值(亿)": info.get("mcap_yi"),
            "换手率": (
                round(info["turnover_pct"], 2)
                if info.get("turnover_pct") is not None
                else None
            ),
            "振幅": (
                round(info["amplitude"], 2)
                if info.get("amplitude") is not None
                else None
            ),
            "涨跌幅": (
                round(info["change_pct"], 2)
                if info.get("change_pct") is not None
                else None
            ),
            "成交量": info.get("volume"),
        })

    if not rows:
        return pd.DataFrame()

    # 4. Build & sort DataFrame
    result = pd.DataFrame(rows)
    result = result.sort_values("评分", ascending=False).reset_index(drop=True)

    # Add rank column
    result.insert(0, "排名", range(1, len(result) + 1))

    if status_text is not None:
        status_text.text(f"筛选完成！共评估 {len(rows)} 只ETF")

    if progress_bar is not None:
        progress_bar.progress(1.0)

    return result.head(top_n)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_etf_screener(strategy_name: str) -> str | None:
    """Render the ETF screener section and return the selected ETF code.

    Args:
        strategy_name: Current strategy name (used for scoring).

    Returns:
        Selected ETF code string, or None if no selection made.
    """
    st.subheader("🔍 策略匹配 ETF 筛选")

    with st.expander("关于筛选逻辑", expanded=False):
        st.markdown(
            f"**「{strategy_name}」策略的筛选标准：**\n\n"
            f"- **4%定投法 / 估值定投**: 优先低PE、宽基指数、高流动性、价格适中\n"
            f"- **趋势跟随**: 优先高换手率、高振幅、趋势明朗的ETF\n"
            f"- **网格交易**: 优先区间震荡型、流动性好、振幅适中的ETF\n"
            f"- **网格+定投**: 综合定投和网格两者的偏好\n\n"
            f"评分满分100分，基于实时行情数据（价格、PE、市值、换手率、振幅等）自动计算。\n"
            f"⚠️ PE数据来自腾讯财经，许多ETF不返回PE值（非Bug，数据源特性）。",
            unsafe_allow_html=False,
        )

    # Screener button
    need_screen = st.button(
        "🔍 开始筛选",
        type="primary",
        use_container_width=True,
        key="etf_screener_btn",
    )

    # Initialize session state for screening results
    if "screener_results" not in st.session_state:
        st.session_state["screener_results"] = None
        st.session_state["screener_strategy"] = ""

    selected_code: str | None = None

    if need_screen:
        progress = st.progress(0.0)
        status = st.empty()

        with st.spinner(""):
            result_df = screen_etfs(
                strategy_name,
                top_n=30,
                progress_bar=progress,
                status_text=status,
            )

        st.session_state["screener_results"] = result_df
        st.session_state["screener_strategy"] = strategy_name

    # Show cached results if available for this strategy
    cached = st.session_state.get("screener_results")
    cached_strategy = st.session_state.get("screener_strategy", "")

    if cached is not None and not cached.empty:
        if cached_strategy != strategy_name:
            st.caption(f"⚠️ 显示的是「{cached_strategy}」的筛选结果，点击按钮重新筛选")
        else:
            st.caption(f"共评估 {len(cached)} 只ETF，按「{strategy_name}」策略适配度排序")

        # Render the table with formatting
        _render_screener_table(cached)

        # Quick-select from top results
        top_codes = ["— 手动输入 —"] + cached["代码"].tolist()[:10]
        selected = st.selectbox(
            "快速选择 Top ETF",
            top_codes,
            key="screener_quick_select",
        )
        if selected and selected != "— 手动输入 —":
            selected_code = selected
            st.success(f"已选择 `{selected}` — 可在上方查看详情")

    elif need_screen:
        st.warning("未获取到任何ETF数据，请检查网络连接。")

    return selected_code


def _render_screener_table(df: pd.DataFrame) -> None:
    """Render the ranked ETF table with conditional formatting."""
    if df.empty:
        return

    display_df = df.copy()

    # Chinese column names for display
    display_cols = []
    for col in ["排名", "代码", "名称", "评分", "最新价", "PE(TTM)", "PB",
                "总市值(亿)", "换手率", "振幅", "涨跌幅"]:
        if col in display_df.columns:
            display_cols.append(col)
    display_df = display_df[display_cols]

    # Format numbers
    for col in ["最新价"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) and x is not None else "—"
            )
    for col in ["PE(TTM)", "PB"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and x is not None and x > 0
                else ("亏损" if pd.notna(x) and x is not None and x <= 0 else "—")
            )
    for col in ["总市值(亿)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) and x is not None else "—"
            )
    for col in ["换手率", "振幅", "涨跌幅"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—"
            )

    # Score color gradient via Styler
    if "评分" in display_df.columns:
        def _score_color(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if v >= 50:
                return f"color: {SUCCESS}; font-weight: 700;"
            elif v >= 35:
                return f"color: {WARNING}; font-weight: 600;"
            else:
                return f"color: {NEUTRAL};"
        styled = _styler_apply(display_df.style, _score_color, ["评分"])
    else:
        styled = display_df.style

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 36 + 38, 600),
    )
