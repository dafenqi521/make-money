"""Macro sentiment pulse — prediction-market probability aggregator.

Fetches high-volume prediction markets from Polymarket (and optionally Kalshi),
classifies them into macro modules via keyword matching, computes a composite
sentiment score, and caches results to a local JSON snapshot.

Usage::

    from src.data.macro_pulse import get_macro_pulse

    pulse = get_macro_pulse()                # from cache, < 100 ms
    pulse = get_macro_pulse(force_refresh=True)  # live refresh, 2-5 s
    if pulse:
        print(f"Sentiment: {pulse.overall_sentiment:.2f}, Risk: {pulse.risk_level}")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "macro_cache"
_CACHE_FILE = _CACHE_DIR / "macro_pulse.json"
_CACHE_TTL_SECONDS = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MacroSignal:
    """A single prediction market, classified into a macro module."""

    question: str
    probability: float          # 0.0–1.0, sentiment-normalised
    raw_probability: float      # original Yes price before sentiment flip
    volume_24h: float
    volume_total: float
    change_24h: float           # probability delta over 24 h
    source: str                 # "polymarket" | "kalshi"
    module: str                 # taxonomy key
    module_label: str           # Chinese display label
    end_date: str = ""          # ISO date string
    is_negative: bool = False   # True when the event is "bad" → prob was flipped


@dataclass
class MacroPulse:
    """Aggregated macro sentiment snapshot."""

    signals: list[MacroSignal] = field(default_factory=list)
    module_scores: dict[str, float] = field(default_factory=dict)
    overall_sentiment: float = 0.5
    risk_level: str = "low"
    risk_color: str = "#16a34a"
    refreshed_at: str = ""
    module_count: int = 0
    total_signals: int = 0

    # Mapping of module key → {label, avg_sentiment, top_signal, n_signals}
    modules_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for session_state / JSON cache."""
        return {
            "overall_sentiment": round(self.overall_sentiment, 3),
            "risk_level": self.risk_level,
            "risk_color": self.risk_color,
            "refreshed_at": self.refreshed_at,
            "module_count": self.module_count,
            "total_signals": self.total_signals,
            "module_scores": {
                k: round(v, 3) for k, v in self.module_scores.items()
            },
            "modules_detail": self.modules_detail,
            "top_signals": [
                {
                    "question": s.question,
                    "probability": s.probability,
                    "module": s.module_label,
                    "volume_24h": s.volume_24h,
                    "change_24h": s.change_24h,
                    "source": s.source,
                }
                for s in self.signals[:15]
            ],
        }


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

# Each module: (key, Chinese label, sentiment direction)
# sentiment_direction: +1 = higher prob → more optimistic; -1 = invert (higher prob → more fearful)
MODULE_DEFS: list[dict] = [
    {
        "key": "monetary",
        "label": "货币政策",
        "sentiment_direction": +1,  # rate cuts are good → higher prob = better
        "keywords": [
            "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
            "inflation", "cpi", "pce", "core inflation", "ecb", "european central bank",
            "pboc", "people's bank of china", "boe", "bank of england",
            "boj", "bank of japan", "fomc", "monetary", "hawkish", "dovish",
            "quantitative easing", "qe", "tightening", "taper",
            "yield curve", "treasury yield", "bond yield",
        ],
    },
    {
        "key": "macro",
        "label": "宏观经济",
        "sentiment_direction": -1,  # recession talk is bad → higher prob = worse
        "keywords": [
            "recession", "gdp", "gross domestic product", "unemployment",
            "nfp", "non-farm", "nonfarm", "payroll", "jobless", "employment",
            "economic growth", "contraction", "depression", "soft landing",
            "hard landing", "no landing", "stagflation", "consumer confidence",
            "pmi", "manufacturing", "industrial production", "retail sales",
            "housing", "home price", "mortgage", "default", "credit",
            "s&p 500", "s&p500", "nasdaq", "dow jones", "stock market",
            "bear market", "bull market", "correction", "crash",
            "vix", "volatility",
        ],
    },
    {
        "key": "geopolitics",
        "label": "地缘政治",
        "sentiment_direction": -1,  # conflict is bad → higher prob = worse
        "keywords": [
            "war", "invasion", "conflict", "military", "nuclear",
            "tariff", "trade war", "sanction", "embargo", "blockade",
            "ukraine", "russia", "putin", "zelensky",
            "china", "taiwan", "xi jinping", "beijing", "ccp",
            "iran", "israel", "middle east", "north korea", "kim",
            "nato", "geopolitic", "regime", "coup", "overthrow",
            "terroris", "insurgency", "secession",
        ],
    },
    {
        "key": "commodities",
        "label": "大宗商品",
        "sentiment_direction": +1,  # stable/low commodity prices are good for most economies
        "keywords": [
            "oil", "crude", "brent", "wti", "gasoline", "natural gas",
            "gold", "silver", "copper", "platinum", "palladium",
            "commodity", "iron ore", "steel", "aluminum", "lithium",
            "agriculture", "wheat", "corn", "soybean", "grain",
            "opec", "energy price",
        ],
    },
    {
        "key": "ai_tech",
        "label": "AI/科技",
        "sentiment_direction": +1,  # AI progress is good → higher prob = better
        "keywords": [
            "ai", "artificial intelligence", "gpt", "openai", "chatgpt",
            "anthropic", "claude", "gemini", "llama", "deepmind",
            "agi", "artificial general intelligence", "singularity",
            "crypto", "bitcoin", "ethereum", "btc", "eth", "blockchain",
            "cryptocurren", "defi", "nft",
            "semiconductor", "chip", "nvidia", "tsmc", "intel",
            "quantum comput", "robot", "autonomous",
            "tech regulation", "antitrust", "big tech",
        ],
    },
]

# Build module lookup: key → {label, sentiment_direction}
_MODULE_MAP: dict[str, dict] = {}
for _md in MODULE_DEFS:
    _MODULE_MAP[_md["key"]] = {
        "label": _md["label"],
        "sentiment_direction": _md["sentiment_direction"],
    }

# Keyword → module key lookup (flat map for fast matching)
_KEYWORD_MODULE: list[tuple[str, str]] = []
for _md in MODULE_DEFS:
    for kw in _md["keywords"]:
        _KEYWORD_MODULE.append((kw, _md["key"]))

# Negative-event keywords — if matched, sentiment_direction is inverted
# (independent of the module's default direction)
_NEGATIVE_EVENT_KW: set[str] = {
    "war", "invasion", "invade", "conflict", "nuclear", "missile", "attack",
    "recession", "depression", "crash", "collapse", "default", "bankrupt",
    "terroris", "assassination", "coup", "overthrow",
    "tariff", "trade war", "sanction", "embargo", "blockade",
    "pandemic", "outbreak", "epidemic",
    "hack", "cyberattack", "data breach",
    "impeach", "government shutdown",
}


# ---------------------------------------------------------------------------
# Keyword classifier
# ---------------------------------------------------------------------------


def classify_module(question: str) -> tuple[str, str, bool] | None:
    """Classify a market question into a macro module.

    Args:
        question: The market question text (English).

    Returns:
        (module_key, module_label, is_negative) or None if no module matches.
        *is_negative* means the event itself is bad news — used to flip
        probability during sentiment computation.
    """
    q_lower = question.lower()

    # Score each module by keyword hits (word-boundary matching)
    scores: dict[str, int] = {}
    for kw, mod_key in _KEYWORD_MODULE:
        # Use word-boundary regex to avoid false matches (e.g. "ai" in "Spain")
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        if pattern.search(q_lower):
            scores[mod_key] = scores.get(mod_key, 0) + 1

    if not scores:
        return None

    # Pick the module with the most keyword hits
    best_module = max(scores, key=lambda k: scores[k])
    info = _MODULE_MAP[best_module]

    # Check for negative-event keywords (word-boundary)
    is_negative = any(
        re.search(r"\b" + re.escape(nk) + r"\b", q_lower, re.IGNORECASE)
        for nk in _NEGATIVE_EVENT_KW
    )

    return (best_module, info["label"], is_negative)


# ---------------------------------------------------------------------------
# Polymarket fetcher
# ---------------------------------------------------------------------------


def _http_get_json(url: str, params: dict[str, str], timeout: int = 20) -> list | None:
    """Make an HTTP GET request and return parsed JSON list.

    Uses PowerShell's Invoke-RestMethod as a bridge, which handles Windows
    proxy/network configurations more reliably than Python's requests on
    some machines (especially in mainland China).
    """
    # Build query string
    qs_parts = [f"{k}={v}" for k, v in params.items()]
    qs = "&".join(qs_parts)
    full_url = f"{url}?{qs}"

    ps_cmd = (
        f"$h = @{{'User-Agent'='Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}}; "
        f"try {{ $r = Invoke-RestMethod -Uri '{full_url}' -Headers $h "
        f"-TimeoutSec {timeout}; $r | ConvertTo-Json -Depth 3 -Compress }} catch {{ '' }}"
    )

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=timeout + 5,
            encoding="utf-8",
        )
        raw = (result.stdout or "").strip()
        if not raw or raw == "''":
            return None
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def fetch_polymarket_signals(
    volume_min: float = 500_000,
    limit: int = 50,
    timeout: int = 20,
) -> list[MacroSignal]:
    """Fetch and classify high-volume Polymarket markets.

    Args:
        volume_min: Minimum total volume in USD.
        limit: Max number of markets to fetch.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of MacroSignal objects that matched a macro module.
    """
    params = {
        "volume_min": str(int(volume_min)),
        "active": "true",
        "closed": "false",
        "limit": str(limit),
    }
    raw_markets = _http_get_json(
        "https://gamma-api.polymarket.com/markets",
        params,
        timeout=timeout,
    )
    if raw_markets is None:
        return []

    signals: list[MacroSignal] = []
    seen_questions: set[str] = set()

    for m in raw_markets:
        question = (m.get("question") or "").strip()
        if not question or len(question) < 10:
            continue

        # De-duplicate similar questions
        q_key = question.lower()[:80]
        if q_key in seen_questions:
            continue
        seen_questions.add(q_key)

        # Classify
        result = classify_module(question)
        if result is None:
            continue
        mod_key, mod_label, is_negative = result

        # Parse prices
        try:
            outcome_prices = json.loads(m.get("outcomePrices", "[]"))
            if not outcome_prices or len(outcome_prices) < 2:
                continue
            yes_price = float(outcome_prices[0])
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

        if not (0 < yes_price <= 1):
            continue

        # Volume
        volume_total = float(m.get("volume", 0) or 0)
        volume_24h = float(m.get("volume24hr", 0) or 0)

        # 24h change
        last_trade = float(m.get("lastTradePrice", 0) or 0)
        prev_price = float(m.get("bestBid", 0) or 0)
        one_day_change = float(m.get("oneWeekPriceChange", 0) or 0)  # gamma API has this
        if one_day_change == 0 and last_trade > 0:
            # Fallback: use bestBid vs lastTrade delta
            one_day_change = last_trade - prev_price if prev_price else 0

        signals.append(MacroSignal(
            question=question,
            probability=yes_price,       # raw for now; normalised in compute
            raw_probability=yes_price,
            volume_24h=volume_24h,
            volume_total=volume_total,
            change_24h=round(one_day_change, 3),
            source="polymarket",
            module=mod_key,
            module_label=mod_label,
            end_date=(m.get("endDateIso") or "")[:10],
            is_negative=is_negative,
        ))

    return signals


# ---------------------------------------------------------------------------
# Sentiment computation
# ---------------------------------------------------------------------------


def _is_negative_module(mod_key: str) -> bool:
    """Check whether a module's default sentiment direction is negative."""
    info = _MODULE_MAP.get(mod_key, {})
    return info.get("sentiment_direction", +1) < 0


def compute_macro_pulse(signals: list[MacroSignal]) -> MacroPulse | None:
    """Aggregate classified signals into a MacroPulse.

    Sentiment normalisation:
      - If the event is negative OR the module sentiment_direction is -1,
        probability is inverted: ``1 - p``.
      - Otherwise, probability is used as-is.
      - The result is a 0–1 score where **higher = more optimistic**.

    Args:
        signals: List of classified MacroSignal objects.

    Returns:
        MacroPulse or None if no signals are available.
    """
    if not signals:
        return None

    # Normalise each signal's probability into a sentiment score
    for s in signals:
        is_neg = s.is_negative or _is_negative_module(s.module)
        s.probability = round(1.0 - s.raw_probability if is_neg else s.raw_probability, 4)

    # Group by module
    by_module: dict[str, list[MacroSignal]] = {}
    for s in signals:
        by_module.setdefault(s.module, []).append(s)

    # Module scores (volume-weighted average sentiment)
    module_scores: dict[str, float] = {}
    modules_detail: dict = {}

    for mod_key, mod_signals in by_module.items():
        total_vol = sum(s.volume_total for s in mod_signals)
        if total_vol > 0:
            weighted = sum(s.probability * s.volume_total for s in mod_signals) / total_vol
        else:
            weighted = sum(s.probability for s in mod_signals) / len(mod_signals)

        module_scores[mod_key] = round(weighted, 3)

        # Top signal (highest volume)
        top = max(mod_signals, key=lambda s: s.volume_total)
        modules_detail[mod_key] = {
            "label": top.module_label,
            "avg_sentiment": round(weighted, 3),
            "top_question": top.question,
            "top_probability": top.probability,
            "top_raw_probability": top.raw_probability,
            "n_signals": len(mod_signals),
        }

    # Overall sentiment (weighted by module count, capped at 0-1)
    if module_scores:
        overall = sum(module_scores.values()) / len(module_scores)
    else:
        overall = 0.5
    overall = max(0.0, min(1.0, overall))

    # Risk level
    if overall < 0.30:
        risk_level, risk_color = "extreme", "#ef4444"
    elif overall < 0.45:
        risk_level, risk_color = "high", "#f97316"
    elif overall < 0.60:
        risk_level, risk_color = "elevated", "#f59e0b"
    else:
        risk_level, risk_color = "low", "#16a34a"

    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")

    return MacroPulse(
        signals=signals,
        module_scores=module_scores,
        overall_sentiment=round(overall, 3),
        risk_level=risk_level,
        risk_color=risk_color,
        refreshed_at=now,
        module_count=len(module_scores),
        total_signals=len(signals),
        modules_detail=modules_detail,
    )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    """Ensure cache directory exists and return cache file path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_FILE


def _read_cache() -> dict | None:
    """Read cached MacroPulse data as a dict, or None if cache is stale/missing."""
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Check TTL
        age = time.time() - data.get("_cached_at", 0)
        if age > _CACHE_TTL_SECONDS:
            return None
        return data
    except Exception:
        return None


def _write_cache(pulse: MacroPulse) -> None:
    """Write MacroPulse to JSON cache."""
    data = pulse.to_dict()
    data["_cached_at"] = time.time()
    path = _cache_path()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Non-critical; next call will re-fetch


def _pulse_from_dict(data: dict) -> MacroPulse:
    """Reconstruct a MacroPulse from cached dict (lightweight version)."""
    # Rebuild minimal signals list from top_signals
    signals = []
    for s in data.get("top_signals", []):
        signals.append(MacroSignal(
            question=s["question"],
            probability=s["probability"],
            raw_probability=s["probability"],  # already normalised in cache
            volume_24h=s.get("volume_24h", 0),
            volume_total=0,
            change_24h=s.get("change_24h", 0),
            source=s.get("source", "polymarket"),
            module="",
            module_label=s.get("module", ""),
        ))

    return MacroPulse(
        signals=signals,
        module_scores=data.get("module_scores", {}),
        overall_sentiment=data.get("overall_sentiment", 0.5),
        risk_level=data.get("risk_level", "low"),
        risk_color=data.get("risk_color", "#16a34a"),
        refreshed_at=data.get("refreshed_at", ""),
        module_count=data.get("module_count", 0),
        total_signals=data.get("total_signals", 0),
        modules_detail=data.get("modules_detail", {}),
    )


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------


def get_macro_pulse(force_refresh: bool = False) -> MacroPulse | None:
    """Get the current macro sentiment pulse.

    Reads from local JSON cache by default (TTL = 1 hour).  Set
    *force_refresh=True* to bypass the cache and fetch live data.

    Returns:
        MacroPulse or None if fetching failed and no cache is available.
    """
    # Serve from cache if fresh enough
    if not force_refresh:
        cached = _read_cache()
        if cached is not None:
            return _pulse_from_dict(cached)

    # Fetch live
    signals = fetch_polymarket_signals()
    if not signals:
        # Fall back to stale cache if available
        path = _cache_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    stale = json.load(fh)
                return _pulse_from_dict(stale)
            except Exception:
                pass
        return None

    pulse = compute_macro_pulse(signals)
    if pulse is None:
        return None

    # Persist
    _write_cache(pulse)
    return pulse


def invalidate_macro_cache() -> None:
    """Delete the macro pulse cache file."""
    path = _cache_path()
    if path.exists():
        try:
            os.remove(path)
        except Exception:
            pass
