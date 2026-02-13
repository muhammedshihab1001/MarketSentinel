import os

from core.data.providers.market.yahoo_provider import YahooProvider
from core.data.providers.market.finnhub_provider import FinnhubProvider


class MarketProviderRouter:

    def __init__(self):

        provider = os.getenv("MARKET_PROVIDER", "yahoo").lower()

        if provider == "finnhub":
            self.primary = FinnhubProvider()
            self.fallback = YahooProvider()

        else:
            self.primary = YahooProvider()
            self.fallback = FinnhubProvider()

    def fetch(self, ticker, start, end, interval):

        try:
            return self.primary.fetch(
                ticker, start, end, interval
            )

        except Exception:

            return self.fallback.fetch(
                ticker, start, end, interval
            )
