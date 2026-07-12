"""Tests for notification module — channels, config, dedup, messages, sender."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.notify.channels import PushPlusChannel
from src.notify.config import (
    load_config, save_config,
    get_token, update_last_notified,
    _CONFIG_PATH, _DEFAULT_CONFIG,
)
from src.notify.message_builder import (
    build_signal_message,
    build_summary_message,
    DISCLAIMER,
)
from src.notify.sender import NotificationSender


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_signal():
    """Create a mock DailySignal-like object for testing."""

    class MockFactor:
        def __init__(self, name, icon, label, detail, signal="bullish", score=0.8):
            self.name = name
            self.icon = icon
            self.label = label
            self.detail = detail
            self.signal = signal
            self.score = score
            self.weight = 0.33
            self.color = "#16a34a"

    class MockSignal:
        composite_action = "buy"
        composite_score = 0.78
        action_label = "强烈建议买入"
        action_icon = "🟢"
        action_color = "#16a34a"
        factors = [
            MockFactor("PE估值", "🟢", "低估区", "PE 处于历史低位"),
            MockFactor("均线趋势", "🟢", "金叉信号", "MA5 上穿 MA20"),
            MockFactor("网格位置", "🟡", "中位区", "价格处于网格中位"),
        ]
        current_price = 3.850
        pe_value = 11.2
        summary = "多因子综合评分较高，建议买入"
        steps = [
            "以当前价 ¥3.850 限价买入 1500 股",
            "设置止损位 ¥3.650（-5.2%）",
            "目标价位 ¥4.200（+9.1%）",
        ]

    return MockSignal()


@pytest.fixture
def sample_info():
    return {
        "name": "沪深300ETF",
        "current_price": 3.850,
        "pe_ttm": 11.2,
        "pe_static": 12.5,
    }


@pytest.fixture
def test_config():
    """Temporary config for testing."""
    return {
        "pushplus_token": "test_token_32chars",
        "enabled": True,
        "notify_on_actions": ["buy", "sell", "accumulate", "reduce"],
        "etf_codes": ["510300"],
        "last_notified": {},
    }


# ===========================================================================
# PushPlusChannel
# ===========================================================================


class TestPushPlusChannel:
    """PushPlus 渠道封装."""

    def test_init_with_token(self):
        ch = PushPlusChannel("abc123")
        assert ch.token == "abc123"

    def test_init_strips_whitespace(self):
        ch = PushPlusChannel("  abc123  ")
        assert ch.token == "abc123"

    def test_send_with_empty_token_returns_false(self):
        ch = PushPlusChannel("")
        ok = ch.send("test", "body")
        assert ok is False

    @patch("src.notify.channels.requests.post")
    def test_send_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200, "msg": "success"}
        mock_post.return_value = mock_resp

        ch = PushPlusChannel("valid_token")
        ok = ch.send("Title", "## Content", template="markdown")
        assert ok is True
        mock_post.assert_called_once()

    @patch("src.notify.channels.requests.post")
    def test_send_api_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 500, "msg": "error"}
        mock_post.return_value = mock_resp

        ch = PushPlusChannel("token")
        ok = ch.send("Title", "Content")
        assert ok is False

    @patch("src.notify.channels.requests.post")
    def test_send_network_error_returns_false(self, mock_post):
        import requests as rq
        mock_post.side_effect = rq.ConnectionError("Connection refused")

        ch = PushPlusChannel("token")
        ok = ch.send("Title", "Content")
        assert ok is False  # Never raises

    @patch("src.notify.channels.requests.post")
    def test_send_posts_to_correct_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200}
        mock_post.return_value = mock_resp

        ch = PushPlusChannel("tok")
        ch.send("T", "C", template="html")
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://www.pushplus.plus/send"
        payload = call_args[1]["json"]
        assert payload["token"] == "tok"
        assert payload["title"] == "T"
        assert payload["content"] == "C"
        assert payload["template"] == "html"


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    """配置管理."""

    def test_default_config_structure(self):
        cfg = load_config()
        assert "pushplus_token" in cfg
        assert "enabled" in cfg
        assert "notify_on_actions" in cfg
        assert "etf_codes" in cfg
        assert "last_notified" in cfg

    def test_save_and_load_roundtrip(self, test_config):
        save_config(test_config)
        loaded = load_config()
        assert loaded["pushplus_token"] == "test_token_32chars"
        assert loaded["enabled"] is True
        assert "buy" in loaded["notify_on_actions"]

    def test_load_creates_file_if_missing(self):
        # Delete config file if it exists
        if _CONFIG_PATH.exists():
            backup = _CONFIG_PATH.read_text(encoding="utf-8")
            _CONFIG_PATH.unlink()
            try:
                cfg = load_config()
                assert _CONFIG_PATH.exists()
                assert cfg == _DEFAULT_CONFIG
            finally:
                # Restore
                _CONFIG_PATH.write_text(backup, encoding="utf-8")
        else:
            cfg = load_config()
            assert _CONFIG_PATH.exists()

    def test_get_token(self, test_config):
        save_config(test_config)
        token = get_token()
        assert token == "test_token_32chars"

    def test_get_token_returns_empty_when_not_set(self):
        cfg = _DEFAULT_CONFIG.copy()
        cfg["pushplus_token"] = ""
        save_config(cfg)
        assert get_token() == ""

    def test_update_last_notified(self, test_config):
        test_config["last_notified"] = {}
        save_config(test_config)
        update_last_notified("510300", "buy", 0.85)
        cfg = load_config()
        entry = cfg["last_notified"].get("510300", {})
        assert entry["action"] == "buy"
        assert entry["date"] == date.today().isoformat()
        assert entry["score"] == 0.85

    def test_merge_missing_keys_on_load(self, test_config):
        """New fields in _DEFAULT_CONFIG should appear in loaded config."""
        # Save a partial config
        partial = {"pushplus_token": "xyz"}
        save_config(partial)
        loaded = load_config()
        assert loaded["pushplus_token"] == "xyz"
        # Missing keys filled from defaults
        assert "enabled" in loaded
        assert "last_notified" in loaded


# ===========================================================================
# Message Builder
# ===========================================================================


class TestBuildSignalMessage:
    """信号消息格式化."""

    def test_returns_tuple(self, mock_signal, sample_info):
        title, content = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert isinstance(title, str)
        assert isinstance(content, str)

    def test_title_contains_code_and_action(self, mock_signal, sample_info):
        title, _ = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert "510300" in title
        assert "沪深300ETF" in title
        assert "强烈建议买入" in title

    def test_content_includes_factors(self, mock_signal, sample_info):
        _, content = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert "PE估值" in content
        assert "均线趋势" in content
        assert "网格位置" in content

    def test_content_includes_price(self, mock_signal, sample_info):
        _, content = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert "3.850" in content

    def test_content_includes_pe(self, mock_signal, sample_info):
        _, content = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert "11.2" in content

    def test_content_includes_steps(self, mock_signal, sample_info):
        _, content = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert "限价买入" in content

    def test_content_includes_disclaimer(self, mock_signal, sample_info):
        _, content = build_signal_message("510300", "沪深300ETF", mock_signal, sample_info)
        assert "仅供参考" in content

    def test_signal_without_factors(self, sample_info):
        """Signal with no factors should still work."""

        class MinimalSignal:
            composite_action = "hold"
            composite_score = 0.50
            action_label = "持有观望"
            action_icon = "⚪"
            action_color = "#888888"
            factors = []
            current_price = 3.85
            pe_value = None
            summary = ""
            steps = []

        title, content = build_signal_message("510300", "Test", MinimalSignal(), sample_info)
        assert "持有观望" in title
        assert "3.850" in content

    def test_with_pe_percentile(self, mock_signal, sample_info):
        """Content should include PE percentile when available."""

        class MockPE:
            pe_percentile = 0.12
            index_name = "沪深300"

        _, content = build_signal_message(
            "510300", "沪深300ETF", mock_signal, sample_info,
            pe_percentile=MockPE(),
        )
        assert "12%" in content


class TestBuildSummaryMessage:
    """简报消息格式化."""

    def test_returns_tuple(self):
        etfs = [{"code": "510300", "name": "沪深300ETF", "price": 3.85,
                  "change_pct": 1.2, "signal": None, "pe_value": 11.2}]
        title, content = build_summary_message(etfs, "午间")
        assert isinstance(title, str)
        assert isinstance(content, str)

    def test_title_contains_time_label(self):
        etfs = [{"code": "510300", "name": "T", "price": 3.85,
                  "change_pct": 0.5, "signal": None, "pe_value": 10.0}]
        title, _ = build_summary_message(etfs, "午间")
        assert "午间" in title
        assert "简报" in title

    def test_content_includes_etf_info(self):
        etfs = [{"code": "510300", "name": "沪深300ETF", "price": 4.20,
                  "change_pct": -0.5, "signal": None, "pe_value": 12.0}]
        _, content = build_summary_message(etfs, "")
        assert "510300" in content
        assert "沪深300ETF" in content
        assert "4.20" in content
        assert "-0.50" in content

    def test_multiple_etfs(self):
        etfs = [
            {"code": "510300", "name": "HS300", "price": 3.85,
             "change_pct": 1.2, "signal": None, "pe_value": 11.0},
            {"code": "510050", "name": "SZ50", "price": 2.70,
             "change_pct": -0.3, "signal": None, "pe_value": 9.5},
        ]
        _, content = build_summary_message(etfs, "收盘前")
        assert "HS300" in content
        assert "SZ50" in content

    def test_positive_change_shows_up_arrow(self):
        etfs = [{"code": "510300", "name": "T", "price": 3.85,
                  "change_pct": 2.5, "signal": None, "pe_value": 10.0}]
        _, content = build_summary_message(etfs, "")
        assert "🔺" in content

    def test_negative_change_shows_down_arrow(self):
        etfs = [{"code": "510300", "name": "T", "price": 3.85,
                  "change_pct": -1.5, "signal": None, "pe_value": 10.0}]
        _, content = build_summary_message(etfs, "")
        assert "🔻" in content

    def test_disclaimer_included(self):
        etfs = [{"code": "510300", "name": "T", "price": 3.85,
                  "change_pct": 0, "signal": None, "pe_value": 10.0}]
        _, content = build_summary_message(etfs, "")
        assert "仅供参考" in content

    def test_summary_with_signal(self, mock_signal):
        etfs = [{"code": "510300", "name": "T", "price": 3.85,
                  "change_pct": 0.5, "signal": mock_signal, "pe_value": 10.0}]
        _, content = build_summary_message(etfs, "")
        assert "强烈建议买入" in content


# ===========================================================================
# NotificationSender
# ===========================================================================


class TestNotificationSender:
    """通知分发器."""

    def test_init_loads_config(self):
        sender = NotificationSender()
        assert sender.config is not None
        assert "enabled" in sender.config

    def test_should_notify_disabled(self, test_config):
        test_config["enabled"] = False
        sender = NotificationSender(config=test_config)
        assert sender.should_notify("510300", "buy", 0.85) is False

    def test_should_notify_action_not_in_whitelist(self, test_config):
        test_config["notify_on_actions"] = ["buy"]
        sender = NotificationSender(config=test_config)
        assert sender.should_notify("510300", "sell", 0.2) is False

    def test_should_notify_buy_always(self, test_config):
        test_config["last_notified"]["510300"] = {
            "date": date.today().isoformat(),
            "action": "buy",
            "score": 0.85,
        }
        sender = NotificationSender(config=test_config)
        # buy is always sent even if already notified today
        assert sender.should_notify("510300", "buy", 0.90) is True

    def test_should_notify_sell_always(self, test_config):
        test_config["last_notified"]["510300"] = {
            "date": date.today().isoformat(),
            "action": "sell",
            "score": 0.20,
        }
        sender = NotificationSender(config=test_config)
        assert sender.should_notify("510300", "sell", 0.15) is True

    def test_should_notify_accumulate_dedup_same_day(self, test_config):
        test_config["last_notified"]["510300"] = {
            "date": date.today().isoformat(),
            "action": "accumulate",
            "score": 0.65,
        }
        sender = NotificationSender(config=test_config)
        # Same action, same day → skip
        assert sender.should_notify("510300", "accumulate", 0.68) is False

    def test_should_notify_accumulate_new_day(self, test_config):
        test_config["last_notified"]["510300"] = {
            "date": "2020-01-01",  # Old date
            "action": "accumulate",
            "score": 0.65,
        }
        sender = NotificationSender(config=test_config)
        assert sender.should_notify("510300", "accumulate", 0.68) is True

    def test_should_notify_different_action_same_day(self, test_config):
        test_config["last_notified"]["510300"] = {
            "date": date.today().isoformat(),
            "action": "accumulate",
            "score": 0.65,
        }
        sender = NotificationSender(config=test_config)
        # Different action → allow
        assert sender.should_notify("510300", "reduce", 0.38) is True

    def test_should_notify_hold_skipped(self, test_config):
        test_config["notify_on_actions"] = ["buy", "sell", "accumulate", "reduce", "hold"]
        sender = NotificationSender(config=test_config)
        # hold is not in default whitelist but we added it here
        assert sender.should_notify("510300", "hold", 0.50) is True

    def test_send_signal_alert_bypassed_by_dedup(self, mock_signal, sample_info, test_config):
        """Signal alert returns False when dedup says skip."""
        test_config["enabled"] = False
        sender = NotificationSender(config=test_config)
        ok = sender.send_signal_alert("510300", "沪深300ETF", mock_signal, sample_info)
        assert ok is False

    @patch("src.notify.channels.requests.post")
    def test_send_test_success(self, mock_post, test_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200}
        mock_post.return_value = mock_resp

        sender = NotificationSender(config=test_config)
        ok = sender.send_test()
        assert ok is True

    @patch("src.notify.channels.requests.post")
    def test_send_test_failure(self, mock_post, test_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 400}
        mock_post.return_value = mock_resp

        sender = NotificationSender(config=test_config)
        ok = sender.send_test()
        assert ok is False

    @patch("src.notify.channels.requests.post")
    def test_send_daily_summary_disabled(self, mock_post, test_config):
        test_config["enabled"] = False
        sender = NotificationSender(config=test_config)
        ok = sender.send_daily_summary([{"code": "510300"}])
        assert ok is False
        mock_post.assert_not_called()

    @patch("src.notify.channels.requests.post")
    def test_send_daily_summary_enabled(self, mock_post, test_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200}
        mock_post.return_value = mock_resp

        test_config["enabled"] = True
        sender = NotificationSender(config=test_config)
        etfs = [{"code": "510300", "name": "HS300", "price": 3.85,
                  "change_pct": 1.2, "signal": None, "pe_value": 11.0}]
        ok = sender.send_daily_summary(etfs)
        assert ok is True
        mock_post.assert_called_once()

    @patch("src.notify.channels.requests.post")
    def test_send_signal_alert_updates_last_notified(self, mock_post, mock_signal,
                                                      sample_info, test_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200}
        mock_post.return_value = mock_resp

        test_config["enabled"] = True
        test_config["last_notified"] = {}
        save_config(test_config)  # persist so update_last_notified picks it up
        sender = NotificationSender(config=test_config)
        ok = sender.send_signal_alert("510300", "沪深300ETF", mock_signal, sample_info)
        assert ok is True
        # update_last_notified writes to disk — reload to verify
        updated = load_config()
        entry = updated["last_notified"].get("510300", {})
        assert entry["action"] == "buy"
