# Temporary proxy layer
# Allows safe migration later

from app.services.data_fetcher import StockPriceFetcher
from app.services.feature_engineering import FeatureEngineer
from app.services.sentiment import SentimentAnalyzer
from app.services.signal_engine import SignalEngine
