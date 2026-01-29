from app.services.news_fetcher import NewsFetcher
from app.services.sentiment import SentimentAnalyzer

news_fetcher = NewsFetcher()
sentiment_analyzer = SentimentAnalyzer()

# Fetch news
news_df = news_fetcher.fetch(query="Apple stock", max_items=15)

# Analyze sentiment
scored_df = sentiment_analyzer.analyze_dataframe(news_df)

# Aggregate daily sentiment
daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

print(daily_sentiment)
