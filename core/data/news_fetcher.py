import feedparser
import pandas as pd
from datetime import datetime
from typing import List


class NewsFetcher:
    """
    Fetch financial news using Google News RSS.
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    def fetch(
        self,
        query: str,
        max_items: int = 50
    ) -> pd.DataFrame:
        """
        Fetch latest news articles related to a query.

        :param query: Search keyword (e.g. 'Apple stock')
        :param max_items: Max number of articles
        :return: DataFrame with news data
        """

        rss_url = self.GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))
        feed = feedparser.parse(rss_url)

        articles = []

        for entry in feed.entries[:max_items]:
            published = self._parse_date(entry.get("published", None))

            articles.append({
                "headline": entry.get("title", ""),
                "published_at": published,
                "source": entry.get("source", {}).get("title", "Unknown"),
                "link": entry.get("link", "")
            })

        if not articles:
            raise ValueError("No news articles fetched")

        return pd.DataFrame(articles)

    @staticmethod
    def _parse_date(date_str):
        try:
            parsed = feedparser._parse_date(date_str)
            if parsed:
                return datetime(*parsed[:6])
        except Exception:
            pass
        return None

