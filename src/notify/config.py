"""通知配置管理 — JSON 文件持久化.

配置文件位置: src/notify/config.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# 配置文件与当前模块同目录
_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _CONFIG_DIR / "config.json"

_DEFAULT_CONFIG: dict = {
    "pushplus_token": "",
    "enabled": False,
    "notify_on_actions": ["buy", "sell", "accumulate", "reduce"],
    "etf_codes": ["510300"],
    "last_notified": {},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """读取通知配置.

    如果配置文件不存在，返回默认配置并自动创建文件。
    """
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Merge with defaults to handle new fields added in future versions
            config = dict(_DEFAULT_CONFIG)
            config.update(loaded)
            return config
    except (json.JSONDecodeError, IOError):
        pass

    # File missing or corrupt — write defaults
    save_config(_DEFAULT_CONFIG)
    return dict(_DEFAULT_CONFIG)


def save_config(config: dict) -> None:
    """保存通知配置到 JSON 文件."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_token() -> str:
    """快捷方法: 返回配置中的 pushplus_token."""
    cfg = load_config()
    return cfg.get("pushplus_token", "").strip()


def update_last_notified(code: str, action: str, score: float) -> None:
    """更新 last_notified 记录（用于去重）."""
    from datetime import date
    cfg = load_config()
    cfg.setdefault("last_notified", {})
    cfg["last_notified"][code] = {
        "date": date.today().isoformat(),
        "action": action,
        "score": round(score, 4),
    }
    save_config(cfg)
