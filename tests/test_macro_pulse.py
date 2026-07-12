"""Tests for macro pulse data fetcher, taxonomy classifier, and sentiment aggregation."""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.macro_pulse import (
    MacroSignal,
    MacroPulse,
    classify_module,
    compute_macro_pulse,
    _is_negative_module,
    _MODULE_MAP,
    MODULE_DEFS,
    _NEGATIVE_EVENT_KW,
    _KEYWORD_MODULE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_signals() -> list:
    """A realistic mix of macro signals across modules."""
    return [
        MacroSignal(
            question="Will the Fed cut rates in July 2026?",
            probability=0.68,
            raw_probability=0.68,
            volume_24h=50000,
            volume_total=2000000,
            change_24h=0.02,
            source="polymarket",
            module="monetary",
            module_label="货币政策",
            is_negative=False,
        ),
        MacroSignal(
            question="US recession in 2026?",
            probability=0.65,  # will be flipped: 1-0.35=0.65
            raw_probability=0.35,
            volume_24h=30000,
            volume_total=1500000,
            change_24h=-0.05,
            source="polymarket",
            module="macro",
            module_label="宏观经济",
            is_negative=True,
        ),
        MacroSignal(
            question="Will China invade Taiwan before 2027?",
            probability=0.88,  # will be flipped: 1-0.12=0.88
            raw_probability=0.12,
            volume_24h=100000,
            volume_total=5000000,
            change_24h=0.01,
            source="polymarket",
            module="geopolitics",
            module_label="地缘政治",
            is_negative=True,
        ),
        MacroSignal(
            question="Gold above $3000 by end of 2026?",
            probability=0.55,
            raw_probability=0.55,
            volume_24h=20000,
            volume_total=800000,
            change_24h=-0.01,
            source="polymarket",
            module="commodities",
            module_label="大宗商品",
            is_negative=False,
        ),
        MacroSignal(
            question="GPT-6 released before 2027?",
            probability=0.42,
            raw_probability=0.42,
            volume_24h=15000,
            volume_total=600000,
            change_24h=0.03,
            source="polymarket",
            module="ai_tech",
            module_label="AI/科技",
            is_negative=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Taxonomy Tests
# ---------------------------------------------------------------------------


class TestTaxonomy:
    """Tests for keyword-based module classification."""

    def test_all_modules_have_keywords(self):
        """Each module definition has at least 3 keywords."""
        for md in MODULE_DEFS:
            assert len(md["keywords"]) >= 3, f"{md['key']} needs more keywords"
            assert md["key"] in _MODULE_MAP
            assert _MODULE_MAP[md["key"]]["label"] == md["label"]

    def test_all_modules_in_map(self):
        """All 5 macro modules exist."""
        for key in ("monetary", "macro", "geopolitics", "commodities", "ai_tech"):
            assert key in _MODULE_MAP

    def test_keyword_module_flat_map(self):
        """_KEYWORD_MODULE has entries for all keywords."""
        assert len(_KEYWORD_MODULE) > 50  # reasonable lower bound

    def test_negative_event_keywords(self):
        """Negative event set contains common bad-event terms."""
        for kw in ("war", "recession", "crash", "sanction", "pandemic"):
            assert kw in _NEGATIVE_EVENT_KW

    # ── Classification accuracy ──

    def test_fed_rates(self):
        result = classify_module("Will the Fed cut rates in July?")
        assert result is not None
        assert result[0] == "monetary"
        assert result[2] is False  # not a negative event

    def test_recession_is_negative(self):
        result = classify_module("US recession in 2026?")
        assert result is not None
        assert result[0] == "macro"
        assert result[2] is True  # recession is negative

    def test_geopolitical_conflict(self):
        result = classify_module("Will China invade Taiwan before 2027?")
        assert result is not None
        assert result[0] == "geopolitics"
        assert result[2] is True  # invasion is negative

    def test_commodities(self):
        result = classify_module("Gold above $3000 by December 2026?")
        assert result is not None
        assert result[0] == "commodities"

    def test_ai_tech(self):
        result = classify_module("GPT-6 released before 2027?")
        assert result is not None
        assert result[0] == "ai_tech"

    def test_no_match_returns_none(self):
        """Non-macro questions return None."""
        assert classify_module("Will Rihanna release a new album?") is None
        assert classify_module("What will happen before GTA VI?") is None

    # ── Word-boundary edge cases ──

    def test_word_boundary_ai_not_in_spain(self):
        """'ai' in 'Spain' should NOT match."""
        result = classify_module("Will Spain win the FIFA World Cup?")
        assert result is None

    def test_word_boundary_war_not_in_warnock(self):
        """'war' in 'Warnock' should NOT match."""
        result = classify_module("Will Raphael Warnock win the nomination?")
        # "nomination" might match "election" keywords, but not war
        assert result is None or result[0] != "geopolitics"

    def test_war_matches_actual_war(self):
        """'war' should match in actual war contexts."""
        result = classify_module("Will there be a war over Taiwan?")
        assert result is not None
        assert result[0] == "geopolitics"
        assert result[2] is True

    def test_bitcoin_matches_ai_tech(self):
        result = classify_module("Will bitcoin hit $1 million before 2030?")
        assert result is not None
        assert result[0] == "ai_tech"

    def test_multi_keyword_tariff_china(self):
        result = classify_module("Will Trump impose tariffs on China?")
        assert result is not None
        assert result[0] == "geopolitics"


# ---------------------------------------------------------------------------
# Sentiment Computation Tests
# ---------------------------------------------------------------------------


class TestSentimentComputation:
    """Tests for compute_macro_pulse and sentiment normalisation."""

    def test_compute_with_signals(self, sample_signals):
        pulse = compute_macro_pulse(sample_signals)
        assert pulse is not None
        assert pulse.total_signals == 5
        assert pulse.module_count >= 3
        assert 0.0 <= pulse.overall_sentiment <= 1.0
        assert pulse.risk_level in ("low", "elevated", "high", "extreme")
        assert pulse.risk_color in ("#16a34a", "#f59e0b", "#f97316", "#ef4444")

    def test_empty_signals_returns_none(self):
        assert compute_macro_pulse([]) is None

    def test_module_scores_summary(self, sample_signals):
        pulse = compute_macro_pulse(sample_signals)
        assert pulse is not None
        for mod_key in pulse.module_scores:
            assert mod_key in _MODULE_MAP
            assert 0.0 <= pulse.module_scores[mod_key] <= 1.0

    def test_modules_detail_has_required_fields(self, sample_signals):
        pulse = compute_macro_pulse(sample_signals)
        assert pulse is not None
        for mod_key, detail in pulse.modules_detail.items():
            assert "label" in detail
            assert "avg_sentiment" in detail
            assert "top_question" in detail
            assert "n_signals" in detail

    def test_negative_module_direction(self):
        """Macro and geopolitics modules invert probability."""
        assert _is_negative_module("macro") is True
        assert _is_negative_module("geopolitics") is True
        assert _is_negative_module("monetary") is False
        assert _is_negative_module("commodities") is False
        assert _is_negative_module("ai_tech") is False

    def test_sentiment_normalisation_negative_events(self):
        """Negative events get their probability flipped (1-p)."""
        signals = [
            MacroSignal(
                question="Test recession",
                probability=0.3,  # raw prob
                raw_probability=0.3,
                volume_24h=1000, volume_total=10000,
                change_24h=0, source="polymarket",
                module="macro", module_label="macro", is_negative=True,
            ),
            MacroSignal(
                question="Test rate cut",
                probability=0.8,
                raw_probability=0.8,
                volume_24h=2000, volume_total=20000,
                change_24h=0, source="polymarket",
                module="monetary", module_label="monetary", is_negative=False,
            ),
        ]
        pulse = compute_macro_pulse(signals)
        assert pulse is not None
        # recession (negative): 1 - 0.30 = 0.70 sentiment
        # rate cut (positive): 0.80 sentiment
        assert pulse.module_scores.get("macro", 0) > 0.5  # flipped to optimistic
        assert pulse.module_scores.get("monetary", 0) > 0.5

    def test_sentiment_on_boundary_values(self):
        """Extreme probabilities produce correct sentiment extremes."""
        signals = [
            MacroSignal(
                question="Certain crash",
                probability=1.0, raw_probability=1.0,
                volume_24h=1000, volume_total=10000,
                change_24h=0, source="polymarket",
                module="macro", module_label="macro", is_negative=True,
            ),
        ]
        pulse = compute_macro_pulse(signals)
        assert pulse is not None
        # 100% crash probability → sentiment = 1 - 1.0 = 0.0 (extreme fear)
        assert pulse.overall_sentiment < 0.01
        assert pulse.risk_level == "extreme"

    def test_to_dict_serializable(self, sample_signals):
        pulse = compute_macro_pulse(sample_signals)
        assert pulse is not None
        d = pulse.to_dict()
        assert isinstance(d, dict)
        assert "overall_sentiment" in d
        assert "risk_level" in d
        assert "top_signals" in d
        json.dumps(d)  # should not raise


# ---------------------------------------------------------------------------
# Risk Level Boundary Tests
# ---------------------------------------------------------------------------


class TestRiskLevels:
    """Boundary tests for risk level classification."""

    def _make_pulse(self, overall: float) -> MacroPulse:
        return compute_macro_pulse([
            MacroSignal(
                question="Test",
                probability=overall, raw_probability=overall,
                volume_24h=1000, volume_total=10000,
                change_24h=0, source="polymarket",
                module="monetary", module_label="monetary",
                is_negative=False,
            ),
        ])

    def test_extreme_below_30(self):
        p = self._make_pulse(0.15)
        assert p.risk_level == "extreme"
        assert p.risk_color == "#ef4444"

    def test_high_at_35(self):
        p = self._make_pulse(0.35)
        assert p.risk_level == "high"
        assert p.risk_color == "#f97316"

    def test_elevated_at_50(self):
        p = self._make_pulse(0.50)
        assert p.risk_level == "elevated"
        assert p.risk_color == "#f59e0b"

    def test_low_at_70(self):
        p = self._make_pulse(0.70)
        assert p.risk_level == "low"
        assert p.risk_color == "#16a34a"

    def test_boundary_30_is_high(self):
        p = self._make_pulse(0.30)
        assert p.risk_level == "high"

    def test_boundary_45_is_elevated(self):
        p = self._make_pulse(0.45)
        assert p.risk_level == "elevated"

    def test_boundary_60_is_low(self):
        p = self._make_pulse(0.60)
        assert p.risk_level == "low"


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for _format_macro_warning and _build_macro_card."""

    def test_format_warning_high_risk(self):
        from src.data.macro_pulse import MacroPulse
        pulse = MacroPulse(
            overall_sentiment=0.25,
            risk_level="high",
            risk_color="#f97316",
            total_signals=5,
            module_count=3,
            refreshed_at="2026-07-12 15:30 CST",
        )
        # We need to import the function
        from src.strategy.four_percent_dca import _format_macro_warning
        warning = _format_macro_warning(pulse)
        assert "宏观情绪" in warning
        assert "0.25" in warning

    def test_format_warning_low_risk_empty(self):
        from src.data.macro_pulse import MacroPulse
        pulse = MacroPulse(
            overall_sentiment=0.75,
            risk_level="low",
            risk_color="#16a34a",
            total_signals=5,
            module_count=3,
            refreshed_at="2026-07-12 15:30 CST",
        )
        from src.strategy.four_percent_dca import _format_macro_warning
        assert _format_macro_warning(pulse) == ""

    def test_format_warning_none_returns_empty(self):
        from src.strategy.four_percent_dca import _format_macro_warning
        assert _format_macro_warning(None) == ""

    def test_build_macro_card_returns_list(self):
        from src.data.macro_pulse import MacroPulse
        pulse = MacroPulse(
            overall_sentiment=0.35,
            risk_level="high",
            risk_color="#f97316",
            total_signals=5,
            module_count=3,
            refreshed_at="2026-07-12 15:30 CST",
            modules_detail={
                "monetary": {
                    "label": "货币政策", "avg_sentiment": 0.35,
                    "top_question": "Fed cut?", "top_probability": 0.35,
                    "top_raw_probability": 0.35, "n_signals": 2,
                },
            },
        )
        from src.strategy.four_percent_dca import _build_macro_card
        result = _build_macro_card(pulse, "test_strat")
        assert isinstance(result, list)
        assert len(result) == 1
        card = result[0]
        assert card.card_type == "macro_pulse"
        assert card.content["overall_sentiment"] == 0.35

    def test_build_macro_card_none_returns_empty(self):
        from src.strategy.four_percent_dca import _build_macro_card
        assert _build_macro_card(None, "test") == []
