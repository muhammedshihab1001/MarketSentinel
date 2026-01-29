from app.services.data_fetcher import StockPriceFetcher

fetcher = StockPriceFetcher()

df = fetcher.fetch(
    ticker="AAPL",
    start_date="2025-01-01",
    end_date="2026-01-01"
)

print(df.head())
print(df.columns)
