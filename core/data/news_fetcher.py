import feedparser
import pandas as pd
import requests
import logging

from datetime import datetime, timedelta
from typing import Dict, Tuple
import threading


logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:
    """
    Institutional News Fetcher.

    Guarantees:
    - timeout protected
    - deduplicated
    - freshness filtered
    - bounded cache
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
        "User-Agent": "MarketSentinel/1.0 (institutional research bot)"
    }

    # --------------------------------------------------

    def _prune_cache(self):

        if len(self._cache) < self.MAX_CACHE_KEYS:
            return

        # cheap prune
        oldest = sorted(
            self._cache.items(),
            key=lambda x: x[1][0]
        )[:100]

        for k, _ in oldest:
            self._cache.pop(k, None)

    # --------------------------------------------------

    def fetch(
        self,
        query: str,
        max_items: int = 50
    ) -> pd.DataFrame:

        now = datetime.utcnow()

        if query in self._cache:
            expiry, df = self._cache[query]

            if now < expiry:
                return df.copy()

        with self._lock:

            # double check
            if query in self._cache:
                expiry, df = self._cache[query]

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

                    # freshness filter
                    if now - published > self.MAX_ARTICLE_AGE:
                        continue

                    headline = entry.get("title", "").strip()

                    articles.append({
                        "headline": headline,
                        "published_at": published,
                        "source": entry.get("source", {}).get("title", "Unknown"),
                        "link": entry.get("link", "")
                    })

                if not articles:
                    logger.warning("No fresh news — returning empty dataframe.")
                    return pd.DataFrame()

                df = pd.DataFrame(articles)

                # dedupe headlines
                df["headline_norm"] = df["headline"].str.lower().str.strip()

                df = df.drop_duplicates("headline_norm")

                df.drop(columns="headline_norm", inplace=True)

                # cache
                self._prune_cache()

                self._cache[query] = (
                    now + self.CACHE_TTL,
                    df
                )

                return df.copy()

            except Exception:

                logger.exception("News fetch failure — returning empty dataframe.")

                return pd.DataFrame()
