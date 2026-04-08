"""
MarketSentinel v4.2.0

Political Risk Agent — detects geopolitical / macro risk events.

Fallback chain (sequential — tries each in order on failure):
  1. GDELT Project API     (free, no key needed)
  2. NewsAPI               (free 100 req/day — NEWSAPI_KEY)
  3. GNews API             (free 100 req/day — GNEWS_KEY)
  4. TheNewsAPI            (free 100 req/day — THENEWSAPI_KEY)
  5. MediaStack            (free 500 req/month — MEDIASTACK_KEY)
  6. CurrentsAPI           (free 600 req/day — CURRENTSAPI_KEY)
  7. Redis snapshot cache  (last known score — no API needed)
  8. Safe default          (score=0.0, label=UNAVAILABLE)

Each provider is tried in sequence. On any failure (timeout,
429, decode error, empty response) the next provider is tried.
The first successful result is cached and returned.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from core.agent.base_agent import BaseAgent

logger = logging.getLogger("marketsentinel.political_agent")


# =========================================================
# PROVIDER BASE
# =========================================================

class _NewsProvider:
    """Base class for all news providers."""

    name = "base"
    timeout = 10

    def fetch(self, country: str) -> List[str]:
        """
        Fetch headlines for the given country.
        Returns list of headline strings.
        Raises on any failure — caller handles fallback.
        """
        raise NotImplementedError


# =========================================================
# PROVIDER 1 — GDELT (free, no key)
# =========================================================

class _GDELTProvider(_NewsProvider):

    name = "gdelt"
    timeout = 10

    ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
    MAX_ARTICLES = 20
    MAX_HEADLINE_LEN = 200

    def fetch(self, country: str) -> List[str]:
        query = f"{country} election OR sanctions OR war OR central bank"
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": self.MAX_ARTICLES,
            "format": "json",
        }
        resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError("GDELT rate limited (429)")
        if resp.status_code != 200:
            raise RuntimeError(f"GDELT HTTP {resp.status_code}")

        data = resp.json()
        articles = data.get("articles", [])
        headlines = []
        for article in articles:
            title = article.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            headlines.append(title.strip()[: self.MAX_HEADLINE_LEN])
        if not headlines:
            raise RuntimeError("GDELT returned no articles")
        return headlines


# =========================================================
# PROVIDER 2 — NewsAPI (free 100 req/day, needs key)
# =========================================================

class _NewsAPIProvider(_NewsProvider):

    name = "newsapi"
    timeout = 10

    ENDPOINT = "https://newsapi.org/v2/everything"

    def __init__(self):
        self._key = os.getenv("NEWSAPI_KEY", "")

    def fetch(self, country: str) -> List[str]:
        if not self._key:
            raise RuntimeError("NEWSAPI_KEY not set")
        params = {
            "q": f"{country} geopolitical risk war sanctions election",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": self._key,
        }
        resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError("NewsAPI rate limited (429)")
        if resp.status_code != 200:
            raise RuntimeError(f"NewsAPI HTTP {resp.status_code}")
        data = resp.json()
        articles = data.get("articles", [])
        headlines = [
            a["title"].strip()
            for a in articles
            if isinstance(a.get("title"), str) and a["title"].strip()
        ]
        if not headlines:
            raise RuntimeError("NewsAPI returned no articles")
        return headlines


# =========================================================
# PROVIDER 3 — GNews (free 100 req/day, needs key)
# =========================================================

class _GNewsProvider(_NewsProvider):

    name = "gnews"
    timeout = 10

    ENDPOINT = "https://gnews.io/api/v4/search"

    def __init__(self):
        self._key = os.getenv("GNEWS_KEY", "")

    def fetch(self, country: str) -> List[str]:
        if not self._key:
            raise RuntimeError("GNEWS_KEY not set")
        params = {
            "q": f"{country} war sanctions election central bank",
            "lang": "en",
            "max": 20,
            "apikey": self._key,
        }
        resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError("GNews rate limited (429)")
        if resp.status_code != 200:
            raise RuntimeError(f"GNews HTTP {resp.status_code}")
        data = resp.json()
        articles = data.get("articles", [])
        headlines = [
            a["title"].strip()
            for a in articles
            if isinstance(a.get("title"), str) and a["title"].strip()
        ]
        if not headlines:
            raise RuntimeError("GNews returned no articles")
        return headlines


# =========================================================
# PROVIDER 4 — TheNewsAPI (free 100 req/day, needs key)
# =========================================================

class _TheNewsAPIProvider(_NewsProvider):

    name = "thenewsapi"
    timeout = 10

    ENDPOINT = "https://api.thenewsapi.com/v1/news/all"

    def __init__(self):
        self._key = os.getenv("THENEWSAPI_KEY", "")

    def fetch(self, country: str) -> List[str]:
        if not self._key:
            raise RuntimeError("THENEWSAPI_KEY not set")
        params = {
            "search": f"{country} geopolitical risk",
            "language": "en",
            "limit": 20,
            "api_token": self._key,
        }
        resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError("TheNewsAPI rate limited (429)")
        if resp.status_code != 200:
            raise RuntimeError(f"TheNewsAPI HTTP {resp.status_code}")
        data = resp.json()
        articles = data.get("data", [])
        headlines = [
            a["title"].strip()
            for a in articles
            if isinstance(a.get("title"), str) and a["title"].strip()
        ]
        if not headlines:
            raise RuntimeError("TheNewsAPI returned no articles")
        return headlines


# =========================================================
# PROVIDER 5 — MediaStack (free 500 req/month, needs key)
# =========================================================

class _MediaStackProvider(_NewsProvider):

    name = "mediastack"
    timeout = 10

    ENDPOINT = "http://api.mediastack.com/v1/news"

    def __init__(self):
        self._key = os.getenv("MEDIASTACK_KEY", "")

    def fetch(self, country: str) -> List[str]:
        if not self._key:
            raise RuntimeError("MEDIASTACK_KEY not set")
        params = {
            "access_key": self._key,
            "keywords": f"{country} war sanctions election geopolitical",
            "languages": "en",
            "limit": 20,
            "sort": "published_desc",
        }
        resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError("MediaStack rate limited (429)")
        if resp.status_code != 200:
            raise RuntimeError(f"MediaStack HTTP {resp.status_code}")
        data = resp.json()
        articles = data.get("data", [])
        headlines = [
            a["title"].strip()
            for a in articles
            if isinstance(a.get("title"), str) and a["title"].strip()
        ]
        if not headlines:
            raise RuntimeError("MediaStack returned no articles")
        return headlines


# =========================================================
# PROVIDER 6 — CurrentsAPI (free 600 req/day, needs key)
# =========================================================

class _CurrentsAPIProvider(_NewsProvider):

    name = "currentsapi"
    timeout = 10

    ENDPOINT = "https://api.currentsapi.services/v1/search"

    def __init__(self):
        self._key = os.getenv("CURRENTSAPI_KEY", "")

    def fetch(self, country: str) -> List[str]:
        if not self._key:
            raise RuntimeError("CURRENTSAPI_KEY not set")
        params = {
            "keywords": f"{country} geopolitical war sanctions election",
            "language": "en",
            "apiKey": self._key,
        }
        resp = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError("CurrentsAPI rate limited (429)")
        if resp.status_code != 200:
            raise RuntimeError(f"CurrentsAPI HTTP {resp.status_code}")
        data = resp.json()
        articles = data.get("news", [])
        headlines = [
            a["title"].strip()
            for a in articles
            if isinstance(a.get("title"), str) and a["title"].strip()
        ]
        if not headlines:
            raise RuntimeError("CurrentsAPI returned no articles")
        return headlines


# =========================================================
# POLITICAL RISK AGENT
# =========================================================

class PoliticalRiskAgent(BaseAgent):

    name = "PoliticalRiskAgent"
    weight = 0.1
    description = (
        "Detects geopolitical and macro risk events via multi-source "
        "news fallback chain. CRITICAL label overrides all trading signals."
    )

    CACHE_TTL = 3600
    MAX_HEADLINE_LEN = 200

    # ── Scoring keywords ──────────────────────────────────────
    KEYWORDS_HIGH = [
        "war", "military", "invasion", "attack",
        "conflict", "sanctions", "nuclear",
    ]
    KEYWORDS_MEDIUM = [
        "election", "government", "policy",
        "central bank", "interest rate",
        "regulation", "geopolitical",
    ]
    KEYWORDS_LOW = [
        "trade", "economic", "inflation", "budget",
    ]

    # ── Label thresholds ─────────────────────────────────────
    THRESHOLD_CRITICAL = 0.75
    THRESHOLD_HIGH = 0.50
    THRESHOLD_MEDIUM = 0.25

    def __init__(self):
        self._cache: Optional[Any] = None

        # ── Build sequential provider chain ──────────────────
        self._providers: List[_NewsProvider] = [
            _GDELTProvider(),
            _NewsAPIProvider(),
            _GNewsProvider(),
            _TheNewsAPIProvider(),
            _MediaStackProvider(),
            _CurrentsAPIProvider(),
        ]

        try:
            from app.inference.cache import RedisCache
            self._cache = RedisCache()
            logger.debug("PoliticalRiskAgent: Redis cache connected.")
        except Exception as exc:
            logger.warning(
                "PoliticalRiskAgent: Redis unavailable — running without cache. %s",
                exc,
            )

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    def _safe_float(self, value, default=0.0):
        try:
            v = float(value)
            return default if not np.isfinite(v) else v
        except Exception:
            return default

    # =========================================================
    # BASE AGENT ENTRYPOINT
    # =========================================================

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ticker = context.get("ticker", "UNKNOWN")
        country = context.get("country", "US")
        result = self.get_political_risk(ticker, country)

        label = result.get("political_risk_label", "LOW")
        score = self._safe_float(result.get("political_risk_score"), 0.0)

        warnings: List[str] = []
        reasoning: List[str] = []

        if label == "CRITICAL":
            warnings.append("CRITICAL political risk — trading disabled")
        elif label == "HIGH":
            warnings.append("HIGH political risk environment")

        source = result.get("source", "unknown")
        reasoning.append(
            f"Political score={score:.2f} label={label} "
            f"country={country} source={source}"
        )

        output = self._format_output(
            score=score,
            confidence=score,
            signals={
                "political_risk_label": label,
                "political_risk_score": score,
                "country": country,
            },
            warnings=warnings,
            reasoning=reasoning,
        )

        output["political_risk_label"] = label
        output["political_risk_score"] = score
        output["top_events"] = result.get("top_events", [])
        output["source"] = source
        output["gdelt_status"] = result.get("gdelt_status", "unknown")

        return output

    # =========================================================
    # FALLBACK CHAIN — try each provider sequentially
    # =========================================================

    def _fetch_with_fallback(self, country: str) -> tuple:
        """
        Try each provider in order. Return (headlines, provider_name)
        on first success. Return ([], 'all_failed') if all fail.
        """
        for provider in self._providers:
            try:
                logger.debug("Trying news provider: %s", provider.name)
                headlines = provider.fetch(country)
                logger.info(
                    "Political risk fetched | provider=%s | headlines=%d",
                    provider.name, len(headlines),
                )
                return headlines, provider.name
            except requests.Timeout:
                logger.warning(
                    "Political provider timeout | provider=%s", provider.name
                )
            except Exception as exc:
                logger.warning(
                    "Political provider failed | provider=%s | error=%s",
                    provider.name, exc,
                )

        logger.warning(
            "All %d political risk providers failed — using fallback",
            len(self._providers),
        )
        return [], "all_failed"

    # =========================================================
    # SCORING
    # =========================================================

    def _score_headline(self, text: str) -> float:
        text = text.lower()
        score = 0.0
        for word in self.KEYWORDS_HIGH:
            if word in text:
                score += 0.4
        for word in self.KEYWORDS_MEDIUM:
            if word in text:
                score += 0.2
        for word in self.KEYWORDS_LOW:
            if word in text:
                score += 0.1
        return min(score, 1.0)

    def _aggregate_score(self, headlines: List[str]) -> float:
        if not headlines:
            return 0.0
        scores = sorted(
            [self._score_headline(h) for h in headlines],
            reverse=True,
        )
        max_score = scores[0]
        top5_mean = sum(scores[:5]) / min(len(scores), 5)
        return float(min(0.6 * max_score + 0.4 * top5_mean, 1.0))

    def _label(self, score: float) -> str:
        if score >= self.THRESHOLD_CRITICAL:
            return "CRITICAL"
        if score >= self.THRESHOLD_HIGH:
            return "HIGH"
        if score >= self.THRESHOLD_MEDIUM:
            return "MEDIUM"
        return "LOW"

    # =========================================================
    # MAIN ENTRY
    # =========================================================

    def get_political_risk(self, ticker: str, country: str = "US") -> Dict:
        """
        Get political risk for a country.

        Flow:
          1. Check Redis cache → return immediately if HIT
          2. Try each news provider sequentially (GDELT → NewsAPI → ... → CurrentsAPI)
          3. If all providers fail → try to serve stale cache
          4. If no cache → return safe default (score=0.0, label=UNAVAILABLE)
        """
        # ── Build cache key ───────────────────────────────────
        cache_key = None
        if self._cache is not None:
            try:
                cache_key = self._cache.build_key({
                    "type": "political_risk",
                    "ticker": ticker,
                    "country": country,
                })
                cached = self._cache.get(cache_key)
                if cached:
                    logger.debug(
                        "Political risk cache HIT | ticker=%s", ticker
                    )
                    return cached
            except Exception:
                pass

        # ── Try fallback chain ────────────────────────────────
        headlines, source = self._fetch_with_fallback(country)

        # ── All providers failed — serve stale cache ──────────
        if not headlines:
            if self._cache is not None and cache_key:
                try:
                    stale = self._cache.get(cache_key + ":stale")
                    if stale:
                        logger.info(
                            "Political risk: serving stale cache | ticker=%s",
                            ticker,
                        )
                        stale["served_from_cache"] = True
                        stale["gdelt_status"] = "all_providers_failed_stale"
                        return stale
                except Exception:
                    pass

            # ── Final safe default ────────────────────────────
            logger.warning(
                "Political risk: all providers failed + no cache | "
                "ticker=%s | returning UNAVAILABLE default",
                ticker,
            )
            return {
                "ticker": ticker,
                "political_risk_score": 0.0,
                "political_risk_label": "UNAVAILABLE",
                "top_events": [],
                "source": "none",
                "gdelt_status": "all_providers_failed",
                "served_from_cache": False,
                "timestamp": int(time.time()),
                "cached": False,
            }

        # ── Score and label ───────────────────────────────────
        score = self._aggregate_score(headlines)
        label = self._label(score)

        result = {
            "ticker": ticker,
            "political_risk_score": float(score),
            "political_risk_label": label,
            "top_events": headlines[:5],
            "source": source,
            "gdelt_status": "ok" if source == "gdelt" else f"gdelt_failed_used_{source}",
            "served_from_cache": False,
            "timestamp": int(time.time()),
            "cached": False,
        }

        # ── Cache result (fresh + stale backup) ──────────────
        if self._cache and cache_key:
            try:
                self._cache.set(cache_key, result, ttl=self.CACHE_TTL)
                # Stale backup has 24h TTL — used when all providers fail
                self._cache.set(
                    cache_key + ":stale", result, ttl=86400
                )
            except Exception:
                pass

        return result
