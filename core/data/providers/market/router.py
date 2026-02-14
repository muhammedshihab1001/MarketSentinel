import os
import logging
import pandas as pd

from core.data.providers.market.yahoo_provider import YahooProvider
from core.data.providers.market.finnhub_provider import FinnhubProvider


logger = logging.getLogger(__name__)


class MarketProviderRouter:

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    ALLOWED_INTERVALS = {
        "1d", "D",
        "1h", "60m",
        "15m",
        "5m",
        "1m"
    }

    def __init__(self):

        self.providers = []

        preferred = os.getenv("MARKET_PROVIDER", "yahoo").lower()

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

        if preferred == "finnhub":
            register("finnhub", FinnhubProvider)
            register("yahoo", YahooProvider)
        else:
            register("yahoo", YahooProvider)
            register("finnhub", FinnhubProvider)

        if not self.providers:
            raise RuntimeError(
                "No market providers available. Check API keys or network."
            )

        logger.info(
            "Market router ready | providers=%s",
            [p[0] for p in self.providers]
        )

    @classmethod
    def _validate_interval(cls, interval):

        if interval not in cls.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

    @classmethod
    def _sanitize_dataframe(cls, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataframe.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider schema invalid. Missing={missing}"
            )

        df = df.copy()

        # normalize datetime
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        if df["date"].isna().any():
            raise RuntimeError("Invalid datetime values from provider.")

        # enforce numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[numeric_cols].isna().any().any():
            raise RuntimeError("NaNs detected in numeric columns.")

        # remove duplicates
        df = df.drop_duplicates(subset=["date"])

        # enforce chronological order
        df = df.sort_values("date")

        return df

    def fetch(self, ticker, start, end, interval):

        self._validate_interval(interval)

        last_error = None

        for name, provider in self.providers:

            try:

                df = provider.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                df = self._sanitize_dataframe(df)

                logger.info(
                    "Market data served | provider=%s ticker=%s",
                    name,
                    ticker
                )

                return df.copy()

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
