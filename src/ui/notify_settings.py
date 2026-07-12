"""通知设置 UI — 侧边栏 PushPlus 配置面板."""

from __future__ import annotations

import streamlit as st

from src.notify import load_config, save_config, NotificationSender
from src.ui.terminal_theme import SUCCESS, DANGER, NEUTRAL


def render_notify_settings() -> None:
    """在侧边栏渲染通知设置面板.

    Call this from st.sidebar context.
    """
    with st.expander("🔔 通知设置", expanded=False):
        st.caption("配置微信推送，在交易信号触发时主动通知你。")

        config = load_config()

        # --- Token ---
        token = st.text_input(
            "PushPlus Token",
            value=config.get("pushplus_token", ""),
            type="password",
            placeholder="在 pushplus.plus 获取 32 位 token",
            key="notify_token_input",
            help="微信关注公众号「pushplus推送加」→ 登录 pushplus.plus → 个人中心获取 token",
        )

        # --- Enabled ---
        enabled = st.toggle(
            "启用微信推送",
            value=config.get("enabled", False),
            key="notify_enabled_toggle",
            help="开启后，信号触发时自动推送微信消息",
        )

        # --- Actions ---
        notify_actions = st.multiselect(
            "推送信号类型",
            options=["buy", "sell", "accumulate", "reduce"],
            default=config.get("notify_on_actions", ["buy", "sell"]),
            format_func=lambda a: {
                "buy": "🟢 强烈买入",
                "accumulate": "🔵 建议增持",
                "reduce": "🟠 建议减仓",
                "sell": "🔴 建议卖出",
            }.get(a, a),
            key="notify_actions_select",
            help="选择哪些信号类型触发推送。持有观望不会推送（仅出现在定时简报中）。",
        )

        # --- ETF codes ---
        etf_codes_str = st.text_input(
            "监控 ETF 代码",
            value=",".join(config.get("etf_codes", ["510300"])),
            placeholder="510300,510050",
            key="notify_etf_codes",
            help="多个代码用逗号分隔，定时简报会汇总所有 ETF",
        )

        # --- Status hint ---
        last = config.get("last_notified", {})
        if last:
            last_info = []
            for code, entry in last.items():
                last_info.append(
                    f"{code}: {entry.get('action','?')} ({entry.get('date','?')})"
                )
            st.caption(f"📨 上次推送：{' | '.join(last_info)}")
        else:
            st.caption("📭 尚未发送过推送")

        # --- Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存设置", use_container_width=True, key="notify_save_btn"):
                # Parse ETF codes
                codes = [c.strip() for c in etf_codes_str.split(",") if c.strip()]
                new_config = {
                    "pushplus_token": token.strip(),
                    "enabled": enabled,
                    "notify_on_actions": notify_actions,
                    "etf_codes": codes,
                    "last_notified": config.get("last_notified", {}),
                }
                save_config(new_config)
                st.toast("✅ 通知设置已保存", icon="🔔")

        with col2:
            if st.button("📤 测试推送", use_container_width=True, key="notify_test_btn"):
                if not token.strip():
                    st.error("请先填写 PushPlus Token")
                else:
                    # Temporarily use current token even if not saved
                    sender = NotificationSender()
                    sender.channel.token = token.strip()
                    ok = sender.send_test()
                    if ok:
                        st.toast("✅ 测试消息发送成功！请检查微信", icon="✅")
                    else:
                        st.error("发送失败，请检查 Token 和网络")
