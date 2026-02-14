import os
import logging
import requests
import pandas as pd
import hashlib
import tempfile
import re

from datetime import datetime, timedelta
from dateutil import parser
from typing import Optional

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config.env_loader import init_env, get_bool, get_env

init_env()

logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:

    FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
    MARKET_AUX_URL = "https://api.marketaux.com/v1/news/all"
    GNEWS_URL = "https://gnews.io/api/v4/search"

    MAX_AGE_HOURS = 72
    INGESTION_DELAY_MIN = 10
    MIN_ARTICLES = 12

    REQUEST_TIMEOUT = (4, 12)

    CACHE_DIR = "data/news_cache"
    CACHE_TTL_MIN = 20
    CACHE_SCHEMA_VERSION = "v7"   # bumped after safety rewrite

    SAFE_TICKER = re.compile(r"^[A-Z0-9._-]{1,12}$")

    EMPTY_SCHEMA = pd.DataFrame({
        "headline": pd.Series(dtype="string"),
        "published_at": pd.Series(dtype="datetime64[ns]"),
        "source": pd.Series(dtype="string"),
        "link": pd.Series(dtype="string"),
    })

    ########################################################

    def __init__(self):

        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.gnews_key = os.getenv("GNEWS_API_KEY")

        self.primary = get_env(
            "NEWS_PROVIDER_PRIMARY",
            "disabled"   # 🔥 SAFE DEFAULT
        ).lower()

        self.failover_enabled = get_bool(
            "NEWS_PROVIDER_FAILOVER", True
        )

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.session = self._build_session()

        self.provider_map = {
            "finnhub": self._fetch_finnhub,
            "marketaux": self._fetch_marketaux,
            "gnews": self._fetch_gnews,
        }

        ####################################################
        # 🔥 INSTITUTIONAL SAFETY LAYER
        ####################################################

        if self.primary == "disabled":
            logger.warning("NewsFetcher running in DISABLED mode.")
            self.disabled = True
            return

        if self.primary not in self.provider_map:
            logger.warning(
                "Unsupported provider '%s' — disabling news.",
                self.primary
            )
            self.disabled = True
            return

        if not any([
            self.finnhub_key,
            self.marketaux_key,
            self.gnews_key
        ]):
            logger.warning(
                "No NEWS API keys detected — disabling news."
            )
            self.disabled = True
            return

        self.disabled = False

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

        session.headers.update(
            {"User-Agent": "MarketSentinel/Institutional"}
        )

        return session

    ########################################################
    # PUBLIC FETCH
    ########################################################

    def fetch(self, query: str, max_items=120):

        ####################################################
        # 🔥 HARD DISABLE GUARD
        ####################################################

        if getattr(self, "disabled", False):
            return self.EMPTY_SCHEMA.copy()

        cache_path = self._cache_path(query, max_items)

        cached = self._load_cache(cache_path)

        if cached is not None:
            return cached

        merged = self._fetch_with_router(query, max_items)

        if len(merged) < self.MIN_ARTICLES:
            logger.warning(
                "Low article count (%s) — neutral sentiment expected.",
                len(merged)
            )

        self._write_cache(merged, cache_path)

        return merged.copy()

    ########################################################
    # CACHE
    ########################################################

    def _cache_path(self, query: str, limit: int):

        raw = f"{self.CACHE_SCHEMA_VERSION}|{query}|{limit}|{self.MAX_AGE_HOURS}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:20]

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
            return pd.read_parquet(path).copy()

        except Exception:
            logger.warning("News cache corrupted — rebuilding.")
            try:
                os.remove(path)
            except Exception:
                pass
            return None

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
    # ROUTER
    ########################################################

    def _fetch_with_router(self, query, limit):

        order = ["finnhub", "marketaux", "gnews"]

        providers = [self.primary] + [
            p for p in order if p != self.primary
        ]

        primary_df = self.EMPTY_SCHEMA.copy()

        for provider in providers:

            fn = self.provider_map.get(provider)

            if fn is None:
                continue

            try:
                df = fn(query, limit)

                if not df.empty:
                    return df if primary_df.empty else self._merge_sources(primary_df, df)

                if primary_df.empty:
                    primary_df = df

            except Exception as e:
                logger.warning("%s provider failed: %s", provider, str(e))

        return primary_df

    ########################################################

    def _merge_sources(self, primary, fallback):

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
    # PROVIDERS (unchanged)
    ########################################################

    def _extract_ticker(self, query: str):

        candidate = query.split()[0].upper()

        if not self.SAFE_TICKER.fullmatch(candidate):
            raise RuntimeError(f"Unsafe ticker parsed from query: {query}")

        return candidate

    def _normalize_date(self, dt):

        if not dt:
            return None

        try:
            ts = parser.parse(str(dt))
            return ts.replace(tzinfo=None)
        except Exception:
            return None

    def _post_process(self, df):

        if df.empty:
            return self.EMPTY_SCHEMA.copy()

        cutoff = datetime.utcnow() - timedelta(hours=self.MAX_AGE_HOURS)
        ingest_guard = datetime.utcnow() - timedelta(minutes=self.INGESTION_DELAY_MIN)

        df = df.dropna(subset=["published_at"])

        df = df.loc[
            (df["published_at"] >= cutoff) &
            (df["published_at"] <= ingest_guard)
        ].copy()

        if df.empty:
            return self.EMPTY_SCHEMA.copy()

        df["headline"] = df["headline"].astype(str).str.strip()
        df = df.loc[df["headline"].str.len() > 12].copy()

        df["key"] = (
            df["headline"].str.lower() +
            df["source"].str.lower()
        )

        df = df.drop_duplicates("key").drop(columns="key")

        return df.sort_values("published_at").reset_index(drop=True)

    ########################################################
    # PROVIDER IMPLEMENTATIONS (same logic)
    ########################################################

    def _fetch_finnhub(self, query, limit):

        if not self.finnhub_key:
            return self.EMPTY_SCHEMA.copy()

        ticker = self._extract_ticker(query)

        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=7)

        params = {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self.finnhub_key
        }

        r = self.session.get(
            self.FINNHUB_URL,
            params=params,
            timeout=self.REQUEST_TIMEOUT
        )

        r.raise_for_status()

        data = r.json()

        rows = []

        for a in data:

            ts = a.get("datetime")
            if not ts:
                continue

            rows.append({
                "headline": a.get("headline", ""),
                "published_at": datetime.utcfromtimestamp(ts),
                "source": a.get("source", "finnhub"),
                "link": a.get("url", "")
            })

        df = pd.DataFrame(rows).head(limit)

        return self._post_process(df)

    def _fetch_marketaux(self, query, limit):

        if not self.marketaux_key:
            return self.EMPTY_SCHEMA.copy()

        ticker = self._extract_ticker(query)

        params = {
            "symbols": ticker,
            "filter_entities": "true",
            "language": "en",
            "limit": min(limit, 100),
            "api_token": self.marketaux_key
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

            rows.append({
                "headline": a.get("title", ""),
                "published_at": self._normalize_date(a.get("published_at")),
                "source": a.get("source", "marketaux"),
                "link": a.get("url", "")
            })

        df = pd.DataFrame(rows)

        return self._post_process(df)

    def _fetch_gnews(self, query, limit):

        if not self.gnews_key:
            return self.EMPTY_SCHEMA.copy()

        ticker = self._extract_ticker(query)

        params = {
            "q": ticker,
            "lang": "en",
            "max": min(limit, 100),
            "token": self.gnews_key
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

            rows.append({
                "headline": a.get("title", ""),
                "published_at": self._normalize_date(a.get("publishedAt")),
                "source": a.get("source", {}).get("name", "gnews"),
                "link": a.get("url", "")
            })

        df = pd.DataFrame(rows)

        return self._post_process(df)
