import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import threading


class NewsFetcher:
    """
    Institutional-grade news fetcher.

    Upgrades:
    ✅ In-memory TTL cache
    ✅ Thread-safe
    ✅ Prevents duplicate RSS calls
    ✅ Massive latency reduction
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    # cache structure:
    # { query: (expiry_time, dataframe) }
    _cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
    _lock = threading.Lock()

    CACHE_TTL = timedelta(minutes=10)

    # --------------------------------------------------

    def fetch(
        self,
        query: str,
        max_items: int = 50
    ) -> pd.DataFrame:

        now = datetime.utcnow()

        # ✅ FAST CACHE CHECK
        if query in self._cache:
            expiry, df = self._cache[query]

            if now < expiry:
                return df.copy()

        # ------------------------------------------------
        # LOCKED FETCH (prevents duplicate RSS calls)
        # ------------------------------------------------

        with self._lock:

            # double check after acquiring lock
            if query in self._cache:
                expiry, df = self._cache[query]

                if now < expiry:
                    return df.copy()

            rss_url = self.GOOGLE_NEWS_RSS.format(
                query=query.replace(" ", "+")
            )

            feed = feedparser.parse(rss_url)

            articles = []

            for entry in feed.entries[:max_items]:
                published = self._parse_date(
                    entry.get("published", None)
                )

                articles.append({
                    "headline": entry.get("title", ""),
                    "published_at": published,
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "link": entry.get("link", "")
                })

            if not articles:
                raise ValueError("No news articles fetched")

            df = pd.DataFrame(articles)

            # ✅ STORE CACHE
            self._cache[query] = (
                now + self.CACHE_TTL,
                df
            )

            return df.copy()

    # --------------------------------------------------

    @staticmethod
    def _parse_date(date_str):
        try:
            parsed = feedparser._parse_date(date_str)
            if parsed:
                return datetime(*parsed[:6])
        except Exception:
            pass

        return None
