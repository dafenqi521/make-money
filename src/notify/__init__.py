"""通知推送模块 — 微信消息推送 + 定时简报.

提供 PushPlus 微信推送渠道封装、配置管理、消息格式化和通知分发器。
可从 Streamlit UI 或独立 CLI 脚本调用，零 Streamlit 依赖。

用法:
    from src.notify import NotificationSender, load_config, save_config

    sender = NotificationSender()
    sender.send_test()                        # 测试通道
    sender.send_signal_alert(code, name, signal, info)  # 信号推送
    sender.send_daily_summary(etfs_data)      # 定时简报
"""

from src.notify.channels import PushPlusChannel
from src.notify.config import load_config, save_config, get_token, update_last_notified
from src.notify.sender import NotificationSender
from src.notify.message_builder import build_signal_message, build_summary_message

__all__ = [
    "PushPlusChannel",
    "NotificationSender",
    "load_config",
    "save_config",
    "get_token",
    "update_last_notified",
    "build_signal_message",
    "build_summary_message",
]
