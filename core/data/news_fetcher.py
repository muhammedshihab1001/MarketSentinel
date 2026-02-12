import os
import logging
import requests
import pandas as pd
import hashlib
import time

from datetime import datetime, timedelta
from dateutil import parser
from typing import Optional

from core.config.env_loader import init_env, get_bool

init_env()

logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:
    """
    Institutional Multi-Provider News Engine

    Priority:
        1. Marketaux
        2. GNews
        3. Safe empty dataframe (never crash training)

    Guarantees:
        ✔ schema normalized
        ✔ deduplicated
        ✔ retry protected
        ✔ rate-limit resilient
        ✔ ingestion-delay safe
        ✔ deterministic ordering
        ✔ variance friendly
        ✔ provider failover
        ✔ disk cache
    """

    MARKET_AUX_URL = "https://api.marketaux.com/v1/news/all"
    GNEWS_URL = "https://gnews.io/api/v4/search"

    MAX_AGE_HOURS = 72
    INGESTION_DELAY_MIN = 10
    MIN_ARTICLES = 12

    REQUEST_TIMEOUT = (4, 12)
    MAX_RETRIES = 3

    CACHE_DIR = "data/news_cache"
    CACHE_TTL_MIN = 20

    EMPTY_SCHEMA = pd.DataFrame(
        columns=["headline", "published_at", "source", "link"]
    )

    ########################################################

    def __init__(self):

        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.gnews_key = os.getenv("GNEWS_API_KEY")

        self.failover_enabled = get_bool(
            "NEWS_PROVIDER_FAILOVER", True
        )

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "MarketSentinel/Institutional"}
        )

    ########################################################
    # CACHE
    ########################################################

    def _cache_path(self, query: str):

        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{self.CACHE_DIR}/{key}.parquet"

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

            if not df.empty:
                return df

        except Exception:
            pass

        return None

    def _write_cache(self, df, path):

        try:
            tmp = path + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        except Exception:
            logger.warning("News cache write failed.")

    ########################################################
    # RETRY WRAPPER
    ########################################################

    def _request(self, url, params):

        for attempt in range(self.MAX_RETRIES):

            try:

                r = self.session.get(
                    url,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT
                )

                if r.status_code == 429:
                    sleep = 1.5 * (attempt + 1)
                    logger.warning(
                        "Rate limited — sleeping %.1fs",
                        sleep
                    )
                    time.sleep(sleep)
                    continue

                r.raise_for_status()

                return r.json()

            except Exception as e:

                if attempt == self.MAX_RETRIES - 1:
                    raise

                sleep = 1.2 * (attempt + 1)
                logger.warning(
                    "News retry %s: %s",
                    attempt + 1,
                    str(e)
                )
                time.sleep(sleep)

        return {}

    ########################################################
    # NORMALIZATION
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

    def _post_process(self, df):

        if df.empty:
            return df

        cutoff = datetime.utcnow() - timedelta(
            hours=self.MAX_AGE_HOURS
        )

        ingest_guard = datetime.utcnow() - timedelta(
            minutes=self.INGESTION_DELAY_MIN
        )

        df = df[
            (df["published_at"] >= cutoff) &
            (df["published_at"] <= ingest_guard)
        ]

        df["headline"] = df["headline"].str.strip()

        df = df[df["headline"].str.len() > 12]

        df["key"] = (
            df["headline"].str.lower() +
            df["source"].str.lower()
        )

        df = df.drop_duplicates("key")
        df = df.drop(columns="key")

        df = df.sort_values("published_at")
        df.reset_index(drop=True, inplace=True)

        return df

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

        data = self._request(
            self.MARKET_AUX_URL,
            params
        ).get("data", [])

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

        data = self._request(
            self.GNEWS_URL,
            params
        ).get("articles", [])

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
    # PUBLIC
    ########################################################

    def fetch(self, query: str, max_items=120):

        cache_path = self._cache_path(query)

        cached = self._load_cache(cache_path)

        if cached is not None:
            logger.info("News cache hit.")
            return cached.copy()

        logger.info("Fetching news: %s", query)

        ###############################################
        # PRIMARY
        ###############################################

        try:

            df = self._fetch_marketaux(query, max_items)

            if len(df) >= self.MIN_ARTICLES:
                self._write_cache(df, cache_path)

                logger.info(
                    "Marketaux OK (%s articles)",
                    len(df)
                )

                return df

            logger.warning(
                "Marketaux sparse (%s)",
                len(df)
            )

        except Exception as e:

            logger.warning(
                "Marketaux failed: %s",
                str(e)
            )

        ###############################################
        # FAILOVER
        ###############################################

        if not self.failover_enabled:
            logger.warning("Failover disabled.")
            return self.EMPTY_SCHEMA.copy()

        try:

            fallback = self._fetch_gnews(query, max_items)

            if not fallback.empty:
                self._write_cache(fallback, cache_path)

                logger.info(
                    "GNews OK (%s articles)",
                    len(fallback)
                )

                return fallback

        except Exception as e:
            logger.warning(
                "Fallback provider failed: %s",
                str(e)
            )

        logger.error("All news providers failed.")

        return self.EMPTY_SCHEMA.copy()
