from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher


class YahooProvider(MarketDataProvider):

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    def fetch(self, ticker, start_date, end_date, interval):
        return self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )
