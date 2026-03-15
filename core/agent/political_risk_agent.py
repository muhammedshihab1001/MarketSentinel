"""
MarketSentinel v4.1.0

Political Risk Agent — detects geopolitical / macro risk events
using the free GDELT Project API (no API key required).

Responsibilities:
    - Fetch recent news headlines via GDELT
    - Score headlines by risk keyword presence
    - Produce a normalised risk score + label
    - Cache results in Redis (TTL = 1 hour)
    - Supply political_risk_label to SignalAgent context

Output key: "political_risk_label"  ← must match SignalAgent.analyze() context key
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from core.agent.base_agent import BaseAgent

logger = logging.getLogger("marketsentinel.political_agent")


class PoliticalRiskAgent(BaseAgent):
    """
    Stateless political risk scorer.

    Data source: GDELT Project API (free, no key required).
    https://api.gdeltproject.org/api/v2/doc/doc

    Risk categories:
        HIGH    — war, invasion, sanctions, nuclear
        MEDIUM  — elections, central bank, regulation
        LOW     — trade, inflation, budget

    Output label fed into SignalAgent as "political_risk_label".
    CRITICAL label triggers trading override in SignalAgent.
    """

    name        = "PoliticalRiskAgent"
    weight      = 0.6
    description = (
        "Detects geopolitical and macro risk events via GDELT headlines. "
        "CRITICAL label overrides all trading signals in SignalAgent."
    )

    # ── GDELT endpoint ────────────────────────────────────────────────────────
    GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

    # ── Cache TTL ─────────────────────────────────────────────────────────────
    CACHE_TTL = 3600       # 1 hour — political events don't change minute-to-minute

    # ── Fetch settings ────────────────────────────────────────────────────────
    MAX_ARTICLES   = 20
    REQUEST_TIMEOUT = 10

    # ── Risk keywords by tier ─────────────────────────────────────────────────
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

    # ── Risk label thresholds ─────────────────────────────────────────────────
    THRESHOLD_CRITICAL = 0.75
    THRESHOLD_HIGH     = 0.50
    THRESHOLD_MEDIUM   = 0.25

    # ────────────────────────────────────────────────────────────────────────
    # INIT  (soft Redis dependency — won't crash if Redis is down)
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._cache: Optional[Any] = None
        try:
            from app.inference.cache import RedisCache
            self._cache = RedisCache()
            logger.debug("PoliticalRiskAgent: Redis cache connected.")
        except Exception as exc:
            logger.warning(
                "PoliticalRiskAgent: Redis unavailable — running without cache. "
                "Reason: %s", exc,
            )

    # ────────────────────────────────────────────────────────────────────────
    # BaseAgent CONTRACT
    # ────────────────────────────────────────────────────────────────────────

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseAgent-compliant entry point.

        Expects context keys:
            ticker  : str  (e.g. "AAPL")
            country : str  (e.g. "US", default "US")

        Returns _format_output() compatible dict with political risk
        embedded in signals under key "political_risk_label".
        """
        ticker  = context.get("ticker", "UNKNOWN")
        country = context.get("country", "US")

        result  = self.get_political_risk(ticker=ticker, country=country)

        label   = result.get("political_risk_label", "LOW")
        score   = self._safe_float(result.get("political_risk_score"), 0.0)

        warnings:  List[str] = []
        reasoning: List[str] = []

        if label == "CRITICAL":
            warnings.append("CRITICAL political risk — all trading overridden")
        elif label == "HIGH":
            warnings.append("HIGH political risk — confidence penalty recommended")

        reasoning.append(
            f"GDELT score={score:.2f} | label={label} | "
            f"country={country} | source=gdelt"
        )

        output = self._format_output(
            score=score,
            confidence=score,     # confidence mirrors score for this agent
            signals={
                "political_risk_label": label,   # ← key SignalAgent reads
                "political_risk_score": score,
                "country":              country,
            },
            warnings=warnings,
            reasoning=reasoning,
        )

        # Surface label at top level for easy context unpacking
        output["political_risk_label"] = label
        output["political_risk_score"] = score
        output["top_events"]           = result.get("top_events", [])

        return output

    # ────────────────────────────────────────────────────────────────────────
    # NEWS FETCH
    # ────────────────────────────────────────────────────────────────────────

    def _fetch_news(self, country: str) -> List[str]:
        """
        Fetch recent headlines from GDELT.
        Returns empty list on any failure — agent degrades gracefully.
        """
        query = f"{country} election OR sanctions OR war OR central bank"
        params = {
            "query":      query,
            "mode":       "ArtList",
            "maxrecords": self.MAX_ARTICLES,
            "format":     "json",
        }

        try:
            response = requests.get(
                self.GDELT_ENDPOINT,
                params=params,
                timeout=self.REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                logger.warning(
                    "GDELT returned HTTP %d for country=%s",
                    response.status_code, country,
                )
                return []

            data     = response.json()
            articles = data.get("articles", [])

            headlines = [
                article.get("title", "")
                for article in articles
                if isinstance(article.get("title"), str)
                and article.get("title", "").strip()
            ]

            logger.debug(
                "GDELT fetch: %d headlines for country=%s",
                len(headlines), country,
            )
            return headlines

        except requests.Timeout:
            logger.warning("GDELT fetch timed out for country=%s", country)
            return []
        except Exception as exc:
            logger.warning("GDELT fetch failed for country=%s: %s", country, exc)
            return []

    # ────────────────────────────────────────────────────────────────────────
    # KEYWORD SCORING
    # ────────────────────────────────────────────────────────────────────────

    def _score_headline(self, text: str) -> float:
        """Score a single headline by keyword presence. Max = 1.0."""
        text  = text.lower()
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
        """
        Aggregate headline scores.

        Uses a blend of the MAX score (captures worst-case event) and
        the MEAN of the top-5 scores (captures sustained risk) to avoid
        both under-sensitivity (plain mean across 20 articles) and
        over-sensitivity (single headline spike).
        """
        if not headlines:
            return 0.0

        scores = sorted(
            [self._score_headline(h) for h in headlines],
            reverse=True,
        )

        max_score  = scores[0]
        top5_mean  = float(sum(scores[:5]) / min(len(scores), 5))

        # 60% weight on worst-case, 40% on sustained signal
        blended = 0.6 * max_score + 0.4 * top5_mean

        return float(min(blended, 1.0))

    # ────────────────────────────────────────────────────────────────────────
    # RISK LABEL
    # ────────────────────────────────────────────────────────────────────────

    def _label(self, score: float) -> str:
        """Map score to risk label. CRITICAL triggers SignalAgent override."""
        if score >= self.THRESHOLD_CRITICAL:
            return "CRITICAL"
        elif score >= self.THRESHOLD_HIGH:
            return "HIGH"
        elif score >= self.THRESHOLD_MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ────────────────────────────────────────────────────────────────────────

    def get_political_risk(
        self,
        ticker:  str,
        country: str = "US",
    ) -> Dict[str, Any]:
        """
        Return political risk assessment for a ticker / country pair.

        Output keys
        -----------
        ticker                : str
        political_risk_score  : float  in [0, 1]
        political_risk_label  : str    "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        top_events            : List[str]  top 5 headlines
        source                : str
        timestamp             : int
        cached                : bool

        NOTE: key is "political_risk_label" — matches SignalAgent context key.
        """

        # ── Cache lookup ──────────────────────────────────────────────────────
        cache_key: Optional[str] = None
        if self._cache is not None:
            try:
                cache_payload = {
                    "type":    "political_risk",
                    "ticker":  ticker,
                    "country": country,
                }
                cache_key = self._cache.build_key(cache_payload)
                cached    = self._cache.get(cache_key)
                if cached:
                    logger.debug(
                        "Political risk cache hit | ticker=%s country=%s",
                        ticker, country,
                    )
                    return cached
            except Exception as exc:
                logger.warning("Political risk cache read failed: %s", exc)

        # ── Fetch + score ─────────────────────────────────────────────────────
        headlines = self._fetch_news(country)
        score     = self._aggregate_score(headlines)
        label     = self._label(score)

        result: Dict[str, Any] = {
            "ticker":               ticker,
            "political_risk_score": float(score),
            "political_risk_label": label,          # ← correct key name
            "top_events":           headlines[:5],
            "source":               "gdelt",
            "timestamp":            int(time.time()),
            "cached":               False,
        }

        # ── Cache write ───────────────────────────────────────────────────────
        if self._cache is not None and cache_key is not None:
            try:
                self._cache.set(cache_key, result, ttl=self.CACHE_TTL)
            except Exception as exc:
                logger.warning("Political risk cache write failed: %s", exc)

        logger.info(
            "Political risk assessed | ticker=%s country=%s "
            "score=%.2f label=%s headlines=%d",
            ticker, country, score, label, len(headlines),
        )

        return result