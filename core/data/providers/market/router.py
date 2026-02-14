import os
import logging
import pandas as pd

from core.data.providers.market.yahoo_provider import YahooProvider
from core.data.providers.market.finnhub_provider import FinnhubProvider


logger = logging.getLogger("marketsentinel.market_router")


class MarketProviderRouter:
    """
    Institutional Provider Router (Production Grade)

    Guarantees:
    ✔ No crash when API keys missing
    ✔ Dynamic provider registration
    ✔ Automatic fallback
    ✔ Schema validation
    ✔ Provider observability
    """

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    ########################################################

    def __init__(self):

        self.providers = []

        preferred = os.getenv("MARKET_PROVIDER", "yahoo").lower()

        ############################################
        # SAFE REGISTRATION FUNCTION
        ############################################

        def register(name, builder):
            try:
                provider = builder()
                self.providers.append((name, provider))
                logger.info("Market provider registered → %s", name)
            except Exception as e:
                logger.warning(
                    "Provider disabled → %s | reason=%s",
                    name,
                    str(e)
                )

        ############################################
        # REGISTER PROVIDERS
        ############################################

        if preferred == "finnhub":

            register("finnhub", FinnhubProvider)
            register("yahoo", YahooProvider)

        else:

            register("yahoo", YahooProvider)
            register("finnhub", FinnhubProvider)

        ############################################

        if not self.providers:
            raise RuntimeError(
                "No market providers available. "
                "Check API keys or network."
            )

        logger.info(
            "Market router ready | providers=%s",
            [p[0] for p in self.providers]
        )

    ########################################################

    @classmethod
    def _sanity_check(cls, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataframe.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider schema invalid. Missing={missing}"
            )

        return df

    ########################################################

    def fetch(self, ticker, start, end, interval):

        last_error = None

        for name, provider in self.providers:

            try:

                df = provider.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                df = self._sanity_check(df)

                logger.info(
                    "Market data served | provider=%s ticker=%s",
                    name,
                    ticker
                )

                return df

            except Exception as e:

                last_error = e

                logger.warning(
                    "Provider failed → %s | ticker=%s | error=%s",
                    name,
                    ticker,
                    str(e)
                )

        raise RuntimeError(
            f"All market providers failed for {ticker}"
        ) from last_error
