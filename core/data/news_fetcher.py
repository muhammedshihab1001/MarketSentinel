import feedparser
import pandas as pd
import requests
import logging
import hashlib
import threading

from datetime import timedelta
from typing import Dict, Tuple


logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:
    """
    Institutional News Fetcher.

    Guarantees:
    - zero future leakage
    - deterministic ordering
    - single clock domain
    - schema stability
    - thread-safe caching
    """

    GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    CACHE_TTL = timedelta(minutes=10)
    MAX_CACHE_KEYS = 500
    MAX_ARTICLE_AGE = timedelta(hours=48)

    _cache: Dict[str, Tuple[pd.Timestamp, pd.DataFrame]] = {}
    _lock = threading.Lock()

    HEADERS = {
        "User-Agent": "MarketSentinel/1.0"
    }

    EMPTY_SCHEMA = pd.DataFrame(
        columns=["headline", "published_at", "source", "link"]
    )

    # --------------------------------------------------

    @staticmethod
    def _now():
        # SINGLE CLOCK DOMAIN
        return pd.Timestamp.utcnow()

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

        # convert to UTC-naive AFTER normalization
        return ts.tz_convert(None)

    # --------------------------------------------------

    @staticmethod
    def _clean_headline(text: str):

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
        now = self._now()

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

                # deterministic ordering BEFORE slicing
                entries = sorted(
                    feed.entries,
                    key=lambda e: e.get("published_parsed", (0,))
                )

                for entry in entries:

                    parsed = entry.get("published_parsed")

                    if not parsed:
                        continue

                    published = self._normalize_timestamp(
                        pd.Timestamp(*parsed[:6])
                    )

                    # 🚨 BLOCK FUTURE NEWS
                    if published > now:
                        logger.warning(
                            "Future-dated article blocked."
                        )
                        continue

                    if now - published > self.MAX_ARTICLE_AGE:
                        continue

                    headline = self._clean_headline(
                        entry.get("title", "")
                    )

                    if not headline:
                        continue

                    link = entry.get("link", "")

                    articles.append({
                        "headline": headline,
                        "published_at": published,
                        "source": entry.get(
                            "source", {}
                        ).get("title", "Unknown"),
                        "link": link
                    })

                    if len(articles) >= max_items:
                        break

                if not articles:
                    logger.warning(
                        "No fresh news — returning empty schema."
                    )
                    return self.EMPTY_SCHEMA.copy()

                df = pd.DataFrame(articles)

                df = df.drop_duplicates(
                    subset=["headline", "link"]
                )

                df = df.sort_values("published_at")

                df.reset_index(drop=True, inplace=True)

                self._prune_cache()

                self._cache[key] = (
                    now + self.CACHE_TTL,
                    df
                )

                return df.copy()

            except Exception:

                logger.exception(
                    "News fetch failure — returning empty schema."
                )

                return self.EMPTY_SCHEMA.copy()
