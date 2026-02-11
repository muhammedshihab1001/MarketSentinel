import feedparser
import pandas as pd
import requests
import logging
import hashlib
import threading
import os

from datetime import timedelta
from typing import Dict, Tuple


logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:

    GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    CACHE_TTL = timedelta(minutes=10)
    MAX_CACHE_KEYS = 500
    MAX_ARTICLE_AGE = timedelta(hours=48)

    MAX_DAILY_NEWS = 50

    RAW_CACHE_DIR = "data/news_raw"

    _cache: Dict[str, Tuple[pd.Timestamp, pd.DataFrame]] = {}
    _lock = threading.Lock()

    HEADERS = {
        "User-Agent": "MarketSentinel/1.0"
    }

    EMPTY_SCHEMA = pd.DataFrame(
        columns=["headline", "published_at", "source", "link"]
    )

    def __init__(self):
        os.makedirs(self.RAW_CACHE_DIR, exist_ok=True)

    @staticmethod
    def _now():
        return pd.Timestamp.utcnow().tz_localize(None)

    def _cache_key(self, query: str, max_items: int) -> str:
        raw = f"{query}_{max_items}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _raw_path(self, key: str):
        return f"{self.RAW_CACHE_DIR}/{key}.xml"

    def _persist_raw_feed(self, key: str, content: bytes):

        path = self._raw_path(key)

        if os.path.exists(path):
            return content

        tmp = path + ".tmp"

        with open(tmp, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        return content

    def _load_raw_feed(self, key: str):

        path = self._raw_path(key)

        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()

        return None

    def _prune_cache(self):

        if len(self._cache) < self.MAX_CACHE_KEYS:
            return

        oldest = sorted(
            self._cache.items(),
            key=lambda x: x[1][0]
        )[:100]

        for k, _ in oldest:
            self._cache.pop(k, None)

    @staticmethod
    def _normalize_timestamp(published):

        ts = pd.to_datetime(
            published,
            utc=True,
            errors="coerce"
        )

        if pd.isna(ts):
            return None

        return ts.tz_convert(None)

    @staticmethod
    def _clean_headline(text: str):

        text = " ".join(text.split())

        if len(text) < 10:
            return None

        return text.strip()

    def fetch(
        self,
        query: str,
        max_items: int = 50
    ) -> pd.DataFrame:

        key = self._cache_key(query, max_items)
        now = self._now()

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

                raw_feed = self._load_raw_feed(key)

                if raw_feed is None:

                    rss_url = self.GOOGLE_NEWS_RSS.format(
                        query=query.replace(" ", "+")
                    )

                    response = requests.get(
                        rss_url,
                        headers=self.HEADERS,
                        timeout=5
                    )

                    response.raise_for_status()

                    raw_feed = self._persist_raw_feed(
                        key,
                        response.content
                    )

                feed = feedparser.parse(raw_feed)

                articles = []

                entries = sorted(
                    feed.entries,
                    key=lambda e: e.get("published_parsed", (0,)),
                    reverse=True
                )

                for entry in entries:

                    parsed = entry.get("published_parsed")

                    if not parsed:
                        continue

                    published = self._normalize_timestamp(
                        pd.Timestamp(*parsed[:6])
                    )

                    if published is None:
                        continue

                    if published > now:
                        continue

                    if now - published > self.MAX_ARTICLE_AGE:
                        continue

                    headline = self._clean_headline(
                        entry.get("title", "")
                    )

                    if not headline:
                        continue

                    link = entry.get("link", "")

                    source = entry.get(
                        "source", {}
                    )

                    if isinstance(source, dict):
                        source = source.get("title", "Unknown")
                    else:
                        source = "Unknown"

                    articles.append({
                        "headline": headline,
                        "published_at": published,
                        "source": source,
                        "link": link
                    })

                    if len(articles) >= max_items:
                        break

                if not articles:
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
