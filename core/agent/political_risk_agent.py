"""
MarketSentinel v4.1.1

Political Risk Agent — detects geopolitical / macro risk events
using the free GDELT Project API.

Portfolio-safe version for unstable news APIs.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
import numpy as np

from core.agent.base_agent import BaseAgent

logger = logging.getLogger("marketsentinel.political_agent")


class PoliticalRiskAgent(BaseAgent):

    name = "PoliticalRiskAgent"

    weight = 0.6

    description = (
        "Detects geopolitical and macro risk events via GDELT headlines. "
        "CRITICAL label overrides all trading signals in SignalAgent."
    )

    # ---------------------------------------------------------
    # API CONFIG
    # ---------------------------------------------------------

    GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

    CACHE_TTL = 3600

    MAX_ARTICLES = 20

    REQUEST_TIMEOUT = 10

    MAX_HEADLINE_LEN = 200

    # ---------------------------------------------------------
    # KEYWORDS
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # LABEL THRESHOLDS
    # ---------------------------------------------------------

    THRESHOLD_CRITICAL = 0.75
    THRESHOLD_HIGH = 0.50
    THRESHOLD_MEDIUM = 0.25

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    def _safe_float(self, value, default=0.0):

        try:

            v = float(value)

            if not np.isfinite(v):
                return default

            return v

        except Exception:

            return default

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self):

        self._cache: Optional[Any] = None

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
    # BASE AGENT ENTRYPOINT
    # =========================================================

    def analyze(self, context: Dict[str, Any]):

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

        reasoning.append(
            f"GDELT score={score:.2f} label={label} country={country}"
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

        return output

    # =========================================================
    # FETCH NEWS
    # =========================================================

    def _fetch_news(self, country: str):

        query = f"{country} election OR sanctions OR war OR central bank"

        params = {

            "query": query,

            "mode": "ArtList",

            "maxrecords": self.MAX_ARTICLES,

            "format": "json",

        }

        try:

            response = requests.get(

                self.GDELT_ENDPOINT,

                params=params,

                timeout=self.REQUEST_TIMEOUT,

            )

            if response.status_code != 200:

                logger.warning("GDELT HTTP %d", response.status_code)

                return []

            try:

                data = response.json()

            except Exception:

                logger.warning("GDELT JSON decode failed")

                return []

            articles = data.get("articles", [])

            headlines = []

            for article in articles:

                title = article.get("title")

                if not isinstance(title, str):

                    continue

                title = title.strip()

                if not title:

                    continue

                title = title[: self.MAX_HEADLINE_LEN]

                headlines.append(title)

            return headlines

        except requests.Timeout:

            logger.warning("GDELT timeout")

            return []

        except Exception as exc:

            logger.warning("GDELT fetch failed: %s", exc)

            return []

    # =========================================================
    # SCORING
    # =========================================================

    def _score_headline(self, text: str):

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

    def _aggregate_score(self, headlines: List[str]):

        if not headlines:

            return 0.0

        scores = sorted(

            [self._score_headline(h) for h in headlines],

            reverse=True,

        )

        max_score = scores[0]

        top5_mean = sum(scores[:5]) / min(len(scores), 5)

        blended = 0.6 * max_score + 0.4 * top5_mean

        return float(min(blended, 1.0))

    # =========================================================
    # LABEL
    # =========================================================

    def _label(self, score: float):

        if score >= self.THRESHOLD_CRITICAL:

            return "CRITICAL"

        elif score >= self.THRESHOLD_HIGH:

            return "HIGH"

        elif score >= self.THRESHOLD_MEDIUM:

            return "MEDIUM"

        else:

            return "LOW"

    # =========================================================
    # MAIN ENTRY
    # =========================================================

    def get_political_risk(self, ticker: str, country: str = "US"):

        cache_key = None

        if self._cache is not None:

            try:

                cache_payload = {

                    "type": "political_risk",

                    "ticker": ticker,

                    "country": country,

                }

                cache_key = self._cache.build_key(cache_payload)

                cached = self._cache.get(cache_key)

                if cached:

                    return cached

            except Exception:

                pass

        headlines = self._fetch_news(country)

        score = self._aggregate_score(headlines)

        label = self._label(score)

        result = {

            "ticker": ticker,

            "political_risk_score": float(score),

            "political_risk_label": label,

            "top_events": headlines[:5],

            "source": "gdelt",

            "timestamp": int(time.time()),

            "cached": False,

        }

        if self._cache and cache_key:

            try:

                self._cache.set(cache_key, result, ttl=self.CACHE_TTL)

            except Exception:

                pass

        return result