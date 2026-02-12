import feedparser
import pandas as pd
import requests
import logging
import hashlib
import threading
import os
import time

from datetime import timedelta
from typing import Dict, Tuple


logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:

    GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    CACHE_TTL = timedelta(minutes=10)
    RAW_TTL = timedelta(hours=6)

    MAX_CACHE_KEYS = 500
    MAX_ARTICLE_AGE = timedelta(hours=48)

    RAW_SCHEMA_VERSION = "2.0"

    RAW_CACHE_DIR = "data/news_raw"

    MAX_RETRIES = 4
    BASE_SLEEP = 1.2

    _cache: Dict[str, Tuple[pd.Timestamp, pd.DataFrame]] = {}
    _lock = threading.Lock()

    HEADERS = {
        "User-Agent": "MarketSentinel/2.0"
    }

    EMPTY_SCHEMA = pd.DataFrame(
        columns=["headline", "published_at", "source", "link"]
    )

    def __init__(self):
        os.makedirs(self.RAW_CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------

    @staticmethod
    def _now():
        return pd.Timestamp.utcnow().tz_localize(None)

    # -----------------------------------------------------

    def _cache_key(self, query: str, max_items: int) -> str:

        raw = (
            f"{query}|{max_items}|"
            f"schema={self.RAW_SCHEMA_VERSION}"
        )

        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    # -----------------------------------------------------

    def _raw_path(self, key: str):
        return f"{self.RAW_CACHE_DIR}/{key}.xml"

    # -----------------------------------------------------

    def _raw_expired(self, path):

        if not os.path.exists(path):
            return True

        modified = pd.Timestamp(
            os.path.getmtime(path),
            unit="s"
        )

        return self._now() - modified > self.RAW_TTL

    # -----------------------------------------------------

    def _persist_raw_feed(self, path, content: bytes):

        tmp = path + ".tmp"

        with open(tmp, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    # -----------------------------------------------------

    def _download_feed(self, url):

        for attempt in range(self.MAX_RETRIES):

            try:

                r = requests.get(
                    url,
                    headers=self.HEADERS,
                    timeout=6
                )

                r.raise_for_status()

                return r.content

            except Exception as e:

                sleep = (
                    self.BASE_SLEEP * (2 ** attempt)
                )

                logger.warning(
                    f"News retry {attempt+1}: {e}"
                )

                time.sleep(sleep)

        raise RuntimeError("News download failed after retries.")

    # -----------------------------------------------------

    def _prune_cache(self):

        if len(self._cache) < self.MAX_CACHE_KEYS:
            return

        oldest = sorted(
            self._cache.items(),
            key=lambda x: x[1][0]
        )[:100]

        for k, _ in oldest:
            self._cache.pop(k, None)

    # -----------------------------------------------------

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

    # -----------------------------------------------------

    @staticmethod
    def _clean_headline(text: str):

        text = " ".join(text.split())

        if len(text) < 10:
            return None

        return text.strip()

    # -----------------------------------------------------

    def fetch(self, query: str, max_items: int = 50) -> pd.DataFrame:

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

                rss_url = self.GOOGLE_NEWS_RSS.format(
                    query=query.replace(" ", "+")
                )

                path = self._raw_path(key)

                if self._raw_expired(path):

                    raw = self._download_feed(rss_url)
                    self._persist_raw_feed(path, raw)

                else:
                    with open(path, "rb") as f:
                        raw = f.read()

                feed = feedparser.parse(raw)

                articles = []
                seen_hashes = set()

                for entry in sorted(
                    feed.entries,
                    key=lambda e: e.get("published_parsed", (0,)),
                    reverse=True
                ):

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

                    h = hashlib.sha256(
                        headline.encode()
                    ).hexdigest()

                    if h in seen_hashes:
                        continue

                    seen_hashes.add(h)

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
                        "link": entry.get("link", "")
                    })

                    if len(articles) >= max_items:
                        break

                if not articles:
                    return self.EMPTY_SCHEMA.copy()

                df = pd.DataFrame(articles)
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
