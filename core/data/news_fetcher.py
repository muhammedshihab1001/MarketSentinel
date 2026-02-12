import feedparser
import pandas as pd
import requests
import logging
import hashlib
import threading
import os
import time
import re

from datetime import timedelta
from typing import Dict, Tuple

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:

    GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    CACHE_TTL = timedelta(minutes=10)
    RAW_TTL = timedelta(hours=6)

    MAX_CACHE_KEYS = 500
    MAX_ARTICLE_AGE = timedelta(hours=48)

    INGESTION_DELAY = timedelta(minutes=15)

    RAW_SCHEMA_VERSION = "4.0"

    RAW_CACHE_DIR = "data/news_raw"
    MAX_RAW_FILES = 300   #  prevent disk fill

    MAX_RETRIES = 4
    BASE_SLEEP = 1.2

    _cache: Dict[str, Tuple[pd.Timestamp, pd.DataFrame]] = {}
    _lock = threading.Lock()

    EMPTY_SCHEMA = pd.DataFrame(
        columns=["headline", "published_at", "source", "link"]
    )

    HEADLINE_REGEX = re.compile(
        r"^\s*$|^\W+$|http[s]?://|^\d+$"
    )

    #####################################################

    def __init__(self):

        os.makedirs(self.RAW_CACHE_DIR, exist_ok=True)

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.SESSION = requests.Session()
        self.SESSION.mount("https://", adapter)
        self.SESSION.headers.update({
            "User-Agent": "MarketSentinel/4.0"
        })

    #####################################################

    @staticmethod
    def _now():
        return pd.Timestamp.utcnow().tz_localize(None)

    #####################################################

    def _cache_key(self, query: str, max_items: int) -> str:

        raw = f"{query}|{max_items}|schema={self.RAW_SCHEMA_VERSION}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    #####################################################

    def _raw_path(self, key: str):
        return f"{self.RAW_CACHE_DIR}/{key}.xml"

    #####################################################

    def _prune_raw_disk(self):

        files = sorted(
            [
                os.path.join(self.RAW_CACHE_DIR, f)
                for f in os.listdir(self.RAW_CACHE_DIR)
            ],
            key=os.path.getmtime
        )

        if len(files) <= self.MAX_RAW_FILES:
            return

        for f in files[:50]:
            try:
                os.remove(f)
            except Exception:
                pass

    #####################################################

    def _raw_expired(self, path):

        if not os.path.exists(path):
            return True

        modified = pd.Timestamp(
            os.path.getmtime(path),
            unit="s"
        )

        return self._now() - modified > self.RAW_TTL

    #####################################################

    def _persist_raw_feed(self, path, content: bytes):

        if len(content) < 1500:
            raise RuntimeError("RSS payload suspiciously small.")

        tmp = path + ".tmp"

        with open(tmp, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        self._prune_raw_disk()

    #####################################################

    def _download_feed(self, url):

        for attempt in range(self.MAX_RETRIES):

            try:

                r = self.SESSION.get(
                    url,
                    timeout=6
                )

                r.raise_for_status()

                return r.content

            except Exception as e:

                sleep = self.BASE_SLEEP * (2 ** attempt)

                logger.warning(
                    f"News retry {attempt+1}: {e}"
                )

                time.sleep(sleep)

        raise RuntimeError("News download failed after retries.")

    #####################################################

    @staticmethod
    def _normalize_timestamp(published, now):

        ts = pd.to_datetime(
            published,
            utc=True,
            errors="coerce"
        )

        if pd.isna(ts):
            return None

        ts = ts.tz_convert(None)

        #  HARD FUTURE GUARD
        ts = min(ts, now)

        return ts

    #####################################################

    def _clean_headline(self, text: str):

        if not text:
            return None

        text = " ".join(text.split()).strip()

        if len(text) < 15:
            return None

        if self.HEADLINE_REGEX.search(text):
            return None

        if text.isupper():
            return None

        return text

    #####################################################

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
                        pd.Timestamp(*parsed[:6]),
                        now
                    )

                    if published is None:
                        continue

                    if published > now - self.INGESTION_DELAY:
                        continue

                    if now - published > self.MAX_ARTICLE_AGE:
                        continue

                    headline = self._clean_headline(
                        entry.get("title", "")
                    )

                    if not headline:
                        continue

                    source = entry.get("source", {})

                    if isinstance(source, dict):
                        source = source.get("title", "Unknown")
                    else:
                        source = "Unknown"

                    #  STRONG DEDUP
                    dedup_key = f"{headline}|{source}|{published.floor('H')}"

                    h = hashlib.sha256(
                        dedup_key.encode()
                    ).hexdigest()

                    if h in seen_hashes:
                        continue

                    seen_hashes.add(h)

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
