"""
MarketSentinel Background Data Refresher

Keeps datasets and features warm.

Runs forever inside container.
"""

import time
import datetime

from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer


TICKERS = ["AAPL", "MSFT", "GOOGL"]
REFRESH_INTERVAL = 60 * 60  # hourly


def refresh():

    market = MarketDataService()
    store = FeatureStore()
    news = NewsFetcher()
    sentiment = SentimentAnalyzer()

    today = datetime.date.today().isoformat()

    for ticker in TICKERS:

        print(f"Refreshing {ticker}...")

        price_df = market.get_price_data(
            ticker,
            "2018-01-01",
            today
        )

        try:

            news_df = news.fetch(f"{ticker} stock", max_items=50)
            scored = sentiment.analyze_dataframe(news_df)
            sentiment_df = sentiment.aggregate_daily_sentiment(scored)

        except Exception:

            sentiment_df = None

        store.get_features(
            price_df,
            sentiment_df if sentiment_df is not None else price_df.iloc[0:0],
            ticker=ticker
        )


def main():

    print("🔥 Market refresher started.")

    while True:

        try:
            refresh()

        except Exception as e:
            print("Refresher error:", e)

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
