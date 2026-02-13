import logging

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger("marketsentinel.provider.yahoo")


class YahooProvider(MarketDataProvider):

    PROVIDER_NAME = "yahoo"

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval):

        df = self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )

        logger.debug(
            "Yahoo served market data | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
