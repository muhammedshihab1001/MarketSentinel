# Temporary proxy layer
# Allows safe migration later

from core.data.data_fetcher import StockPriceFetcher
from core.features.feature_engineering import FeatureEngineer
from core.sentiment.sentiment import SentimentAnalyzer
from core.signals.signal_engine import SignalEngine
from core.data.news_fetcher import NewsFetcher