"""消息格式化 — 将信号/简报数据渲染为 Markdown 推送内容."""

from __future__ import annotations

from typing import Optional

# 尝试导入 DailySignal 类型（用于类型标注，CLI 中也能工作）
try:
    from src.ui.signal_panel import DailySignal
except ImportError:
    DailySignal = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = "> ⚠️ 本消息由 AI 投资决策系统自动生成，仅供参考，不构成投资建议"

ACTION_EMOJI = {
    "buy": "🟢",
    "accumulate": "🔵",
    "hold": "⚪",
    "reduce": "🟠",
    "sell": "🔴",
}

FACTOR_EMOJI = {
    "PE估值": "📊",
    "均线趋势": "📈",
    "网格位置": "📐",
}


# ---------------------------------------------------------------------------
# Signal alert message
# ---------------------------------------------------------------------------


def build_signal_message(
    code: str,
    name: str,
    signal,
    info: dict,
    pe_percentile=None,
) -> tuple:
    """构建交易信号推送消息.

    Args:
        code: ETF 代码，如 "510300"
        name: ETF 名称，如 "沪深300ETF"
        signal: DailySignal 对象（来自 signal_panel.compute_daily_signal）
        info: fetch_etf_info() 返回的实时行情 dict
        pe_percentile: 可选 PEPercentile 对象

    Returns:
        (title: str, content_markdown: str)
    """
    emoji = ACTION_EMOJI.get(signal.composite_action, "⚪")

    # --- Title ---
    title = f"{emoji} {code} {name} — {signal.action_label}"

    # --- Content ---
    lines = [
        f"## {emoji} 操作建议：{signal.action_label}",
        "",
        f"**综合评分：{signal.composite_score:.2f} / 1.00**",
        "",
    ]

    # Factor details
    if signal.factors:
        lines.append("### 因子详情")
        lines.append("")
        for f in signal.factors:
            fe = FACTOR_EMOJI.get(f.name, "•")
            lines.append(
                f"- {f.icon} **{f.name}**：{f.label} — {f.detail}"
            )
        lines.append("")

    # Key prices
    lines.append("### 关键数据")
    lines.append("")
    current_price = signal.current_price or info.get("current_price", "N/A")
    if isinstance(current_price, (int, float)):
        lines.append(f"- 当前价：¥{current_price:.3f}")
    pe_value = signal.pe_value or info.get("pe_ttm") or info.get("pe_static")
    if pe_value and isinstance(pe_value, (int, float)):
        lines.append(f"- PE(TTM)：{pe_value:.1f}")
    if pe_percentile is not None:
        try:
            pp = pe_percentile.pe_percentile
            if pp is not None:
                lines.append(f"- PE 历史分位：{pp:.0%}")
        except (AttributeError, TypeError):
            pass
    lines.append("")

    # Action steps
    if signal.steps:
        lines.append("### 操作步骤")
        lines.append("")
        for i, step in enumerate(signal.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    # Summary
    if signal.summary:
        lines.append(f"> {signal.summary}")
        lines.append("")

    lines.append("---")
    lines.append(DISCLAIMER)

    return title, "\n".join(lines)


# ---------------------------------------------------------------------------
# Daily summary message
# ---------------------------------------------------------------------------


def build_summary_message(
    etfs_data: list,
    time_label: str = "",
) -> tuple:
    """构建定时简报推送消息.

    Args:
        etfs_data: ETF 数据列表，每项为 dict:
            {
                "code": "510300",
                "name": "沪深300ETF",
                "price": 3.850,
                "change_pct": 1.2,
                "signal": DailySignal | None,
                "pe_value": 11.2,
            }
        time_label: 时段标签，如 "午间"、"收盘前"

    Returns:
        (title: str, content_markdown: str)
    """
    from datetime import datetime

    now = datetime.now()
    label = time_label or ("午间" if now.hour < 14 else "收盘前")

    # --- Title ---
    title = f"📊 {label}ETF简报 {now.strftime('%m-%d %H:%M')}"

    # --- Content ---
    lines = [
        f"## 📊 {label}ETF简报",
        f"**{now.strftime('%Y年%m月%d日 %H:%M')}**",
        "",
    ]

    for etf in etfs_data:
        code = etf.get("code", "N/A")
        name = etf.get("name", f"ETF {code}")
        price = etf.get("price")
        change_pct = etf.get("change_pct", 0) or 0
        signal = etf.get("signal")
        pe_value = etf.get("pe_value")

        direction = "🔺" if change_pct > 0 else ("🔻" if change_pct < 0 else "➖")
        price_str = f"¥{price:.3f}" if isinstance(price, (int, float)) else "N/A"

        lines.append(f"### {direction} {code} {name}")
        lines.append("")
        lines.append(f"- **当前价**：{price_str}（{change_pct:+.2f}%）")

        if signal is not None:
            lines.append(
                f"- **信号**：{signal.action_icon} {signal.action_label}"
                f"（{signal.composite_score:.2f}）"
            )

        if pe_value and isinstance(pe_value, (int, float)):
            lines.append(f"- **PE(TTM)**：{pe_value:.1f}")

        lines.append("")

    lines.append("---")
    lines.append(DISCLAIMER)

    return title, "\n".join(lines)
