from app.services.news_fetcher import NewsFetcher

fetcher = NewsFetcher()

df = fetcher.fetch(query="Apple stock", max_items=10)

print(df.head())
print(df.columns)
