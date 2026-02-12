import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Dict

logger = logging.getLogger("marketsentinel.news")


class NewsFetcher:
    """
    INSTITUTIONAL MULTI-SOURCE NEWS FETCHER

    Priority:
        1️⃣ Marketaux (structured financial news)
        2️⃣ GNews fallback

    Guarantees:
        ✔ normalized schema
        ✔ deduplicated headlines
        ✔ timezone safe
        ✔ variance friendly
        ✔ sentiment-ready
    """

    MARKET_AUX_URL = "https://api.marketaux.com/v1/news/all"
    GNEWS_URL = "https://gnews.io/api/v4/search"

    MAX_AGE_HOURS = 72
    MIN_ARTICLES = 15   # ensures variance

    def __init__(self):

        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.gnews_key = os.getenv("GNEWS_API_KEY")

        if not self.marketaux_key:
            raise RuntimeError(
                "MARKETAUX_API_KEY not found in environment."
            )

        if not self.gnews_key:
            logger.warning(
                "GNEWS_API_KEY missing — fallback disabled."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "MarketSentinel/Institutional"}
        )

    ############################################################

    @staticmethod
    def _normalize_date(dt):

        if not dt:
            return None

        try:
            ts = parser.parse(dt)
            return ts.replace(tzinfo=None)
        except Exception:
            return None

    ############################################################

    def _filter_age(self, df: pd.DataFrame):

        cutoff = datetime.utcnow() - timedelta(hours=self.MAX_AGE_HOURS)

        return df[df["published_at"] >= cutoff]

    ############################################################

    @staticmethod
    def _dedup(df):

        df["key"] = (
            df["headline"].str.lower()
            + df["source"].str.lower()
        )

        df = df.drop_duplicates("key")

        return df.drop(columns="key")

    ############################################################
    # PRIMARY — MARKETAUX
    ############################################################

    def _fetch_marketaux(self, query, limit=100) -> pd.DataFrame:

        params = {
            "api_token": self.marketaux_key,
            "search": query,
            "language": "en",
            "limit": limit
        }

        r = self.session.get(
            self.MARKET_AUX_URL,
            params=params,
            timeout=10
        )

        r.raise_for_status()

        data = r.json().get("data", [])

        if not data:
            return pd.DataFrame()

        rows = []

        for article in data:

            published = self._normalize_date(
                article.get("published_at")
            )

            if not published:
                continue

            rows.append({
                "headline": article.get("title"),
                "published_at": published,
                "source": article.get("source", "unknown"),
                "link": article.get("url", "")
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        df = self._filter_age(df)
        df = self._dedup(df)

        return df

    ############################################################
    # FALLBACK — GNEWS
    ############################################################

    def _fetch_gnews(self, query, limit=100) -> pd.DataFrame:

        if not self.gnews_key:
            return pd.DataFrame()

        params = {
            "q": query,
            "token": self.gnews_key,
            "lang": "en",
            "max": limit
        }

        r = self.session.get(
            self.GNEWS_URL,
            params=params,
            timeout=10
        )

        r.raise_for_status()

        articles = r.json().get("articles", [])

        rows = []

        for a in articles:

            published = self._normalize_date(
                a.get("publishedAt")
            )

            if not published:
                continue

            rows.append({
                "headline": a.get("title"),
                "published_at": published,
                "source": a.get("source", {}).get("name", "unknown"),
                "link": a.get("url", "")
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        df = self._filter_age(df)
        df = self._dedup(df)

        return df

    ############################################################
    # PUBLIC
    ############################################################

    def fetch(self, query: str, max_items=150) -> pd.DataFrame:

        logger.info("Fetching news for: %s", query)

        try:

            df = self._fetch_marketaux(query, max_items)

            if len(df) >= self.MIN_ARTICLES:
                logger.info(
                    "Marketaux returned %s articles",
                    len(df)
                )
                return df.sort_values("published_at")

            logger.warning(
                "Marketaux sparse (%s). Using fallback.",
                len(df)
            )

        except Exception as e:
            logger.warning("Marketaux failed: %s", e)

        ########################################################

        fallback = self._fetch_gnews(query, max_items)

        if fallback.empty:
            logger.warning("Fallback returned zero articles.")

        else:
            logger.info(
                "Fallback returned %s articles",
                len(fallback)
            )

        return fallback.sort_values("published_at")
