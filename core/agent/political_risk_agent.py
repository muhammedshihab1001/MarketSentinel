# =========================================================
# POLITICAL RISK AGENT v1.0
# Event Risk Detection using Free News Sources (GDELT)
# Stateless | Cache Enabled | CV Portfolio Ready
# =========================================================

import logging
import time
import requests
from typing import Dict, Any, List

from app.inference.cache import RedisCache

logger = logging.getLogger("marketsentinel.political_agent")


class PoliticalRiskAgent:

    """
    Detects geopolitical / macro risk events.

    Data Source
    -----------
    GDELT Project API (free, no key required)

    https://api.gdeltproject.org/api/v2/doc/doc

    Risk Categories
    ---------------
    - elections
    - war / conflict
    - sanctions
    - central bank policy
    - geopolitical instability

    Returns normalized political risk score.
    """

    GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

    CACHE_TTL = 3600

    KEYWORDS_HIGH = [
        "war",
        "military",
        "invasion",
        "attack",
        "conflict",
        "sanctions",
        "nuclear",
    ]

    KEYWORDS_MEDIUM = [
        "election",
        "government",
        "policy",
        "central bank",
        "interest rate",
        "regulation",
        "geopolitical",
    ]

    KEYWORDS_LOW = [
        "trade",
        "economic",
        "inflation",
        "budget",
    ]

    def __init__(self):

        self.cache = RedisCache()

    # -----------------------------------------------------
    # FETCH NEWS
    # -----------------------------------------------------

    def _fetch_news(self, country: str) -> List[str]:

        query = f"{country} election OR sanctions OR war OR central bank"

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": 20,
            "format": "json",
        }

        try:

            response = requests.get(
                self.GDELT_ENDPOINT,
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                return []

            data = response.json()

            articles = data.get("articles", [])

            headlines = [
                article.get("title", "")
                for article in articles
                if isinstance(article.get("title"), str)
            ]

            return headlines

        except Exception:
            logger.exception("Political risk news fetch failed")
            return []

    # -----------------------------------------------------
    # KEYWORD SCORING
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # AGGREGATE SCORE
    # -----------------------------------------------------

    def _aggregate_score(self, headlines: List[str]) -> float:

        if not headlines:
            return 0.0

        scores = [self._score_headline(h) for h in headlines]

        if not scores:
            return 0.0

        return min(sum(scores) / len(scores), 1.0)

    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    def _label(self, score: float) -> str:

        if score >= 0.75:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"

    # -----------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------

    def get_political_risk(
        self,
        ticker: str,
        country: str = "US"
    ) -> Dict[str, Any]:

        cache_payload = {
            "type": "political_risk",
            "ticker": ticker,
            "country": country,
        }

        key = self.cache.build_key(cache_payload)

        cached = self.cache.get(key)

        if cached:
            return cached

        headlines = self._fetch_news(country)

        score = self._aggregate_score(headlines)

        label = self._label(score)

        result = {
            "ticker": ticker,
            "political_risk_score": float(score),
            "risk_label": label,
            "top_events": headlines[:5],
            "source": "gdelt",
            "timestamp": int(time.time()),
        }

        try:
            self.cache.set(key, result, ttl=self.CACHE_TTL)
        except Exception:
            logger.warning("Political risk cache write failed")

        return result