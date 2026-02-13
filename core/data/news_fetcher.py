import os
import logging
import requests
import pandas as pd
import hashlib
import tempfile

from datetime import datetime, timedelta
from dateutil import parser
from typing import Optional

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config.env_loader import init_env, get_bool

init_env()

logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:

    MARKET_AUX_URL = "https://api.marketaux.com/v1/news/all"
    GNEWS_URL = "https://gnews.io/api/v4/search"

    MAX_AGE_HOURS = 72
    INGESTION_DELAY_MIN = 10
    MIN_ARTICLES = 12

    REQUEST_TIMEOUT = (4, 12)

    CACHE_DIR = "data/news_cache"
    CACHE_TTL_MIN = 20
    CACHE_SCHEMA_VERSION = "v3"   #  bump version after schema hardening

    ########################################################
    #  SCHEMA LOCK
    ########################################################

    EMPTY_SCHEMA = pd.DataFrame({
        "headline": pd.Series(dtype="string"),
        "published_at": pd.Series(dtype="datetime64[ns]"),
        "source": pd.Series(dtype="string"),
        "link": pd.Series(dtype="string"),
    })

    ########################################################

    def __init__(self):

        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.gnews_key = os.getenv("GNEWS_API_KEY")

        self.failover_enabled = get_bool(
            "NEWS_PROVIDER_FAILOVER", True
        )

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.session = self._build_session()

    ########################################################

    def _build_session(self):

        session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=retry
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update(
            {"User-Agent": "MarketSentinel/Institutional"}
        )

        return session

    ########################################################
    # CACHE
    ########################################################

    def _cache_path(self, query: str, limit: int):

        raw = f"{self.CACHE_SCHEMA_VERSION}|{query}|{limit}|{self.MAX_AGE_HOURS}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:20]

        return f"{self.CACHE_DIR}/{key}.parquet"

    ########################################################

    def _load_cache(self, path: str) -> Optional[pd.DataFrame]:

        if not os.path.exists(path):
            return None

        modified = datetime.utcfromtimestamp(
            os.path.getmtime(path)
        )

        if datetime.utcnow() - modified > timedelta(
            minutes=self.CACHE_TTL_MIN
        ):
            return None

        try:
            df = pd.read_parquet(path)
            return df.copy()
        except Exception:
            logger.warning("News cache corrupted — rebuilding.")
            try:
                os.remove(path)
            except Exception:
                pass
            return None

    ########################################################
    # ATOMIC WRITE
    ########################################################

    def _write_cache(self, df, path):

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=self.CACHE_DIR,
                suffix=".tmp"
            ) as tmp:

                df.to_parquet(tmp.name, index=False)
                temp_name = tmp.name

            os.replace(temp_name, path)

        except Exception:
            logger.warning("News cache write failed.")

    ########################################################
    # DATE NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_date(dt):

        if not dt:
            return None

        try:
            ts = parser.parse(dt)
            return ts.replace(tzinfo=None)
        except Exception:
            return None

    ########################################################
    # POST PROCESS — PANDAS SAFE
    ########################################################

    def _post_process(self, df):

        if df.empty:
            return self.EMPTY_SCHEMA.copy()

        cutoff = datetime.utcnow() - timedelta(
            hours=self.MAX_AGE_HOURS
        )

        ingest_guard = datetime.utcnow() - timedelta(
            minutes=self.INGESTION_DELAY_MIN
        )

        mask = (
            (df["published_at"] >= cutoff) &
            (df["published_at"] <= ingest_guard)
        )

        df = df.loc[mask].copy()   #  CRITICAL FIX

        df["headline"] = df["headline"].str.strip()

        df = df.loc[df["headline"].str.len() > 12].copy()

        df["key"] = (
            df["headline"].str.lower() +
            df["source"].str.lower()
        )

        df = df.drop_duplicates("key").drop(columns="key")

        return df.sort_values("published_at").reset_index(drop=True)

    ########################################################
    # PROVIDERS
    ########################################################

    def _fetch_marketaux(self, query, limit):

        if not self.marketaux_key:
            return self.EMPTY_SCHEMA.copy()

        params = {
            "api_token": self.marketaux_key,
            "search": query,
            "language": "en",
            "limit": limit
        }

        r = self.session.get(
            self.MARKET_AUX_URL,
            params=params,
            timeout=self.REQUEST_TIMEOUT
        )

        r.raise_for_status()

        data = r.json().get("data", [])

        rows = []

        for a in data:

            published = self._normalize_date(
                a.get("published_at")
            )

            if not published:
                continue

            rows.append({
                "headline": a.get("title", ""),
                "published_at": published,
                "source": a.get("source", "unknown"),
                "link": a.get("url", "")
            })

        return self._post_process(pd.DataFrame(rows))

    ########################################################

    def _fetch_gnews(self, query, limit):

        if not self.gnews_key:
            return self.EMPTY_SCHEMA.copy()

        params = {
            "q": query,
            "token": self.gnews_key,
            "lang": "en",
            "max": limit
        }

        r = self.session.get(
            self.GNEWS_URL,
            params=params,
            timeout=self.REQUEST_TIMEOUT
        )

        r.raise_for_status()

        data = r.json().get("articles", [])

        rows = []

        for a in data:

            published = self._normalize_date(
                a.get("publishedAt")
            )

            if not published:
                continue

            rows.append({
                "headline": a.get("title", ""),
                "published_at": published,
                "source": a.get("source", {}).get("name", "unknown"),
                "link": a.get("url", "")
            })

        return self._post_process(pd.DataFrame(rows))

    ########################################################
    # MERGE
    ########################################################

    def _merge_sources(self, primary, fallback):

        if primary.empty and fallback.empty:
            return self.EMPTY_SCHEMA.copy()

        if primary.empty:
            return fallback.copy()

        if fallback.empty:
            return primary.copy()

        merged = pd.concat(
            [primary, fallback],
            ignore_index=True
        ).copy()

        merged["key"] = (
            merged["headline"].str.lower() +
            merged["source"].str.lower()
        )

        merged = merged.drop_duplicates("key").drop(columns="key")

        return merged.sort_values("published_at").reset_index(drop=True)

    ########################################################
    # PUBLIC
    ########################################################

    def fetch(self, query: str, max_items=120):

        cache_path = self._cache_path(query, max_items)

        cached = self._load_cache(cache_path)

        if cached is not None:
            logger.info("News cache hit.")
            return cached

        logger.info("Fetching news: %s", query)

        primary = pd.DataFrame()
        fallback = pd.DataFrame()

        try:
            primary = self._fetch_marketaux(query, max_items)
        except Exception as e:
            logger.warning("Marketaux failed: %s", str(e))

        if self.failover_enabled:
            try:
                fallback = self._fetch_gnews(query, max_items)
            except Exception as e:
                logger.warning("Fallback provider failed: %s", str(e))

        merged = self._merge_sources(primary, fallback)

        if len(merged) < self.MIN_ARTICLES:
            logger.warning(
                "Low article count (%s) — training will fallback to neutral sentiment.",
                len(merged)
            )

        self._write_cache(merged, cache_path)

        return merged.copy()
