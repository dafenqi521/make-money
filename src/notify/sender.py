"""通知分发器 — 去重 + 多渠道分发.

NotificationSender 是整个通知模块的门面：调用方只需创建 sender 实例，
传入信号或简报数据，其余（去重、格式化、渠道分发）全部内部处理。
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from src.notify.channels import PushPlusChannel
from src.notify.config import load_config, update_last_notified
from src.notify.message_builder import build_signal_message, build_summary_message


class NotificationSender:
    """通知分发器：负责去重判断 + 消息发送.

    用法:
        sender = NotificationSender()
        sender.send_test()                              # 测试 PushPlus 通道
        sender.send_signal_alert("510300", "沪深300ETF", signal, info)  # 信号推送
        sender.send_daily_summary([etf1, etf2])         # 定时简报
    """

    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else load_config()
        token = self.config.get("pushplus_token", "")
        self.channel = PushPlusChannel(token)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def should_notify(self, code: str, action: str, score: float) -> bool:
        """判断是否应该发送信号推送.

        规则:
          - enabled=False → 不发送
          - action 不在 notify_on_actions 白名单 → 不发送
          - buy / sell → 始终发送（重要信号）
          - accumulate / reduce → 同一天同一 action 不重复
          - hold → 信号触发时不发送（只在简报中出现）

        Args:
            code: ETF 代码
            action: 信号动作 (buy/sell/accumulate/reduce/hold)
            score: 综合评分（只用于记录，不影响判断）

        Returns:
            True 如果应该发送
        """
        if not self.config.get("enabled", False):
            return False

        allowed = self.config.get("notify_on_actions", [])
        if action not in allowed:
            return False

        # Always notify for buy/sell
        if action in ("buy", "sell"):
            return True

        # Dedup accumulate/reduce: same action on same day → skip
        last = self.config.get("last_notified", {}).get(code, {})
        today = date.today().isoformat()

        if last.get("date") != today:
            return True  # New day, allow

        if last.get("action") != action:
            return True  # Different action, allow (e.g. reduce→accumulate)

        return False  # Same action, same day → skip

    # ------------------------------------------------------------------
    # Send methods
    # ------------------------------------------------------------------

    def send_signal_alert(
        self,
        code: str,
        name: str,
        signal,
        info: dict,
        pe_percentile=None,
    ) -> bool:
        """发送交易信号推送（带去重检查）.

        Returns:
            True 如果消息已发送，False 如果被去重拦截或发送失败
        """
        action = getattr(signal, "composite_action", "hold")
        score = getattr(signal, "composite_score", 0.5)

        if not self.should_notify(code, action, score):
            return False

        title, content = build_signal_message(
            code, name, signal, info, pe_percentile=pe_percentile,
        )

        success = self.channel.send(title, content)

        if success:
            update_last_notified(code, action, score)

        return success

    def send_daily_summary(self, etfs_data: list, time_label: str = "") -> bool:
        """发送定时简报（不去重，每次调用都发送）.

        Returns:
            True 如果发送成功
        """
        if not self.config.get("enabled", False):
            return False

        title, content = build_summary_message(etfs_data, time_label=time_label)
        return self.channel.send(title, content)

    def send_test(self) -> bool:
        """发送测试消息，用于验证 token 是否有效.

        Returns:
            True 如果测试消息发送成功
        """
        from datetime import datetime

        title = "✅ PushPlus 测试消息"
        content = (
            "## ✅ 推送通道正常\n\n"
            f"您的 ETF 投资决策系统已成功接入 PushPlus 微信推送。\n\n"
            f"- 发送时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- 推送渠道：PushPlus → 微信\n\n"
            "之后您将在此收到交易信号提醒和定时简报。\n\n"
            "---\n"
            "> 如需关闭推送，请在系统侧边栏「通知设置」中关闭。"
        )

        return self.channel.send(title, content)
