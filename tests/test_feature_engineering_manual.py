from app.services.data_fetcher import StockPriceFetcher
from app.services.feature_engineering import FeatureEngineer

fetcher = StockPriceFetcher()
fe = FeatureEngineer()

df = fetcher.fetch(
    ticker="AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

df = fe.add_returns(df)
df = fe.add_volatility(df)
df = fe.add_rsi(df)
df = fe.add_macd(df)

print(df[["date", "return", "volatility", "rsi", "macd"]].tail())
