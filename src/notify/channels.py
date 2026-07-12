"""推送渠道封装.

目前支持 PushPlus（pushplus.plus），后续可扩展 Server酱、WxPusher 等。
"""

from __future__ import annotations

import requests

PUSHPLUS_API = "https://www.pushplus.plus/send"
TIMEOUT = 10  # seconds


class PushPlusChannel:
    """PushPlus 微信推送渠道.

    免费额度 200 条/天，支持 Markdown，API 极简。

    注册:
        1. 微信关注公众号「pushplus推送加」
        2. 登录 https://www.pushplus.plus/ 获取 token（32位 hex）
        3. 将 token 填入配置文件或 Streamlit 设置面板
    """

    def __init__(self, token: str):
        self.token = token.strip() if token else ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, title: str, content: str, template: str = "markdown") -> bool:
        """发送推送消息.

        Args:
            title: 消息标题（微信通知卡片上显示的标题）
            content: 消息正文（Markdown 格式）
            template: 消息模板类型，默认 "markdown"

        Returns:
            True 如果发送成功，False 如果失败（不抛异常）
        """
        if not self.token:
            return False

        payload = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template,
        }

        try:
            resp = requests.post(PUSHPLUS_API, json=payload, timeout=TIMEOUT)
            data = resp.json()
            # PushPlus returns {"code": 200, "msg": "success", ...}
            return data.get("code") == 200
        except (requests.RequestException, ValueError):
            return False
