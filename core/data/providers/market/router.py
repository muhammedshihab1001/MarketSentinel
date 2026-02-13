import os
import logging
import pandas as pd
import time

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
    - provider cooldown protection
    """

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    PROVIDER_COOLDOWN = 90  # seconds

    ########################################################

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

        self._primary_failed_at = None

        logger.info(
            "Market router initialized | primary=%s fallback=%s",
            self.primary_name,
            self.fallback_name
        )

    ########################################################

    def _primary_available(self):

        if self._primary_failed_at is None:
            return True

        elapsed = time.time() - self._primary_failed_at

        if elapsed > self.PROVIDER_COOLDOWN:
            logger.warning(
                "Primary provider cooldown expired — retrying %s",
                self.primary_name
            )
            self._primary_failed_at = None
            return True

        return False

    ########################################################

    @classmethod
    def _sanity_check(cls, df: pd.DataFrame):

        if df is None:
            raise RuntimeError("Provider returned None.")

        if df.empty:
            raise RuntimeError("Provider returned empty dataframe.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider schema invalid. Missing={missing}"
            )

        return df

    ########################################################

    def fetch(self, ticker, start, end, interval):

        # ---------- TRY PRIMARY ----------

        if self._primary_available():

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

                self._primary_failed_at = time.time()

                logger.warning(
                    "Primary provider failed | provider=%s ticker=%s error=%s",
                    self.primary_name,
                    ticker,
                    str(primary_error)
                )

        # ---------- FALLBACK ----------

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
