import os
import logging
import pandas as pd

from core.data.providers.market.yahoo_provider import YahooProvider
from core.data.providers.market.finnhub_provider import FinnhubProvider


logger = logging.getLogger("marketsentinel.market_router")


class MarketProviderRouter:
    """
    Institutional Provider Router.

    Guarantees:
    - explicit provider visibility
    - safe fallback
    - schema sanity check
    - no silent degradation
    """

    def __init__(self):

        provider = os.getenv("MARKET_PROVIDER", "yahoo").lower()

        if provider == "finnhub":
            self.primary = FinnhubProvider()
            self.fallback = YahooProvider()
            self.primary_name = "finnhub"
            self.fallback_name = "yahoo"

        else:
            self.primary = YahooProvider()
            self.fallback = FinnhubProvider()
            self.primary_name = "yahoo"
            self.fallback_name = "finnhub"

        logger.info(
            "Market router initialized | primary=%s fallback=%s",
            self.primary_name,
            self.fallback_name
        )

    ########################################################

    @staticmethod
    def _sanity_check(df: pd.DataFrame):

        if df is None:
            raise RuntimeError("Provider returned None.")

        if df.empty:
            raise RuntimeError("Provider returned empty dataframe.")

        if "date" not in df.columns:
            raise RuntimeError("Provider schema invalid: missing date.")

        return df

    ########################################################

    def fetch(self, ticker, start, end, interval):

        try:

            df = self.primary.fetch(
                ticker,
                start,
                end,
                interval
            )

            df = self._sanity_check(df)

            logger.debug(
                "Market data served by PRIMARY provider=%s ticker=%s",
                self.primary_name,
                ticker
            )

            return df

        except Exception as primary_error:

            logger.warning(
                "Primary provider failed | provider=%s ticker=%s error=%s",
                self.primary_name,
                ticker,
                str(primary_error)
            )

            try:

                df = self.fallback.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                df = self._sanity_check(df)

                logger.critical(
                    "FALLBACK provider engaged | provider=%s ticker=%s",
                    self.fallback_name,
                    ticker
                )

                return df

            except Exception as fallback_error:

                logger.critical(
                    "Both providers failed | primary=%s fallback=%s ticker=%s",
                    self.primary_name,
                    self.fallback_name,
                    ticker
                )

                raise RuntimeError(
                    "Market data unavailable from all providers."
                ) from fallback_error
