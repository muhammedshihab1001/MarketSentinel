from app.services.data_fetcher import StockPriceFetcher
from app.services.news_fetcher import NewsFetcher
from app.services.sentiment import SentimentAnalyzer
from app.services.feature_engineering import FeatureEngineer

fetcher = StockPriceFetcher()
news_fetcher = NewsFetcher()
sentiment_analyzer = SentimentAnalyzer()
fe = FeatureEngineer()

price_df = fetcher.fetch("AAPL", "2023-01-01", "2024-01-01")

news_df = news_fetcher.fetch("Apple stock", max_items=20)
scored_df = sentiment_analyzer.analyze_dataframe(news_df)
sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

merged = fe.merge_price_sentiment(price_df, sentiment_df)
print(merged[["date", "close", "avg_sentiment", "news_count"]].tail())
