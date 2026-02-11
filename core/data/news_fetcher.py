import feedparser
import pandas as pd
import requests
import logging
import hashlib

from datetime import datetime, timedelta
from typing import Dict, Tuple
import threading


logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:
    """
    Institutional News Fetcher.

    Guarantees:
    - parameter-aware caching
    - timezone normalization
    - deterministic ordering
    - headline safety
    - link deduplication
    - inference-safe fallback
    """

    GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    CACHE_TTL = timedelta(minutes=10)
    MAX_CACHE_KEYS = 500
    MAX_ARTICLE_AGE = timedelta(hours=48)

    _cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
    _lock = threading.Lock()

    HEADERS = {
        "User-Agent": "MarketSentinel/1.0"
    }

    # --------------------------------------------------

    def _cache_key(self, query: str, max_items: int) -> str:
        raw = f"{query}_{max_items}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # --------------------------------------------------

    def _prune_cache(self):

        if len(self._cache) < self.MAX_CACHE_KEYS:
            return

        oldest = sorted(
            self._cache.items(),
            key=lambda x: x[1][0]
        )[:100]

        for k, _ in oldest:
            self._cache.pop(k, None)

    # --------------------------------------------------

    def _normalize_timestamp(self, published):

        ts = pd.to_datetime(published, utc=True)

        return ts.tz_convert(None)

    # --------------------------------------------------

    def _clean_headline(self, text: str):

        text = " ".join(text.split())

        if len(text) < 10:
            return None

        return text.strip()

    # --------------------------------------------------

    def fetch(
        self,
        query: str,
        max_items: int = 50
    ) -> pd.DataFrame:

        key = self._cache_key(query, max_items)
        now = datetime.utcnow()

        # fast path
        if key in self._cache:
            expiry, df = self._cache[key]

            if now < expiry:
                return df.copy()

        with self._lock:

            if key in self._cache:
                expiry, df = self._cache[key]

                if now < expiry:
                    return df.copy()

            try:

                rss_url = self.GOOGLE_NEWS_RSS.format(
                    query=query.replace(" ", "+")
                )

                response = requests.get(
                    rss_url,
                    headers=self.HEADERS,
                    timeout=5
                )

                response.raise_for_status()

                feed = feedparser.parse(response.content)

                articles = []

                for entry in feed.entries[:max_items]:

                    parsed = entry.get("published_parsed")

                    if not parsed:
                        continue

                    published = datetime(*parsed[:6])

                    published = self._normalize_timestamp(published)

                    if now - published > self.MAX_ARTICLE_AGE:
                        continue

                    headline = entry.get("title", "")

                    headline = self._clean_headline(headline)

                    if not headline:
                        continue

                    link = entry.get("link", "")

                    articles.append({
                        "headline": headline,
                        "published_at": published,
                        "source": entry.get("source", {}).get("title", "Unknown"),
                        "link": link
                    })

                if not articles:
                    logger.warning("No fresh news — returning empty dataframe.")
                    return pd.DataFrame()

                df = pd.DataFrame(articles)

                # Institutional dedupe
                df = df.drop_duplicates(subset=["headline", "link"])

                df = df.sort_values("published_at")

                df.reset_index(drop=True, inplace=True)

                # cache safely
                self._prune_cache()

                self._cache[key] = (
                    now + self.CACHE_TTL,
                    df
                )

                return df.copy()

            except Exception:

                logger.exception(
                    "News fetch failure — returning empty dataframe."
                )

                return pd.DataFrame()
