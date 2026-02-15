import os
import logging
import pandas as pd

from core.data.providers.market.yahoo_provider import YahooProvider

# THESE WILL EXIST SOON
try:
    from core.data.providers.market.twelvedata_provider import TwelveDataProvider
except Exception:
    TwelveDataProvider = None

try:
    from core.data.providers.market.alphavantage_provider import AlphaVantageProvider
except Exception:
    AlphaVantageProvider = None


logger = logging.getLogger(__name__)


class MarketProviderRouter:
    """
    Institutional Multi-Vendor Router.

    Priority:
        1️⃣ TwelveData
        2️⃣ AlphaVantage
        3️⃣ Yahoo (fallback)
    """

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

    ############################################################

    def __init__(self):

        self.providers = []

        self._register_providers()

        if not self.providers:
            raise RuntimeError(
                "No market providers available. Check API keys."
            )

        logger.info(
            "Market router ready | priority=%s",
            [p[0] for p in self.providers]
        )

    ############################################################

    def _register_providers(self):
        """
        Registers providers in PRIORITY ORDER.
        """

        def register(name, builder, api_key_env=None):

            # Skip provider if API key missing
            if api_key_env and not os.getenv(api_key_env):
                logger.warning(
                    "Provider skipped → %s | missing %s",
                    name,
                    api_key_env
                )
                return

            if builder is None:
                logger.warning(
                    "Provider unavailable → %s (module missing)",
                    name
                )
                return

            try:
                provider = builder()
                self.providers.append((name, provider))

                logger.info(
                    "Provider registered → %s",
                    name
                )

            except Exception as e:

                logger.warning(
                    "Provider disabled → %s | reason=%s",
                    name,
                    str(e)
                )

        ####################################################
        # PRIORITY ORDER
        ####################################################

        register(
            "twelvedata",
            TwelveDataProvider,
            "TWELVEDATA_API_KEY"
        )

        register(
            "alphavantage",
            AlphaVantageProvider,
            "ALPHAVANTAGE_API_KEY"
        )

        # Yahoo ALWAYS registers (no key needed)
        register("yahoo", YahooProvider)

    ############################################################

    @classmethod
    def _validate_interval(cls, interval):

        if interval not in cls.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

    ############################################################

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

        ####################################################
        # SAFE datetime normalization
        ####################################################

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        if df.empty:
            raise RuntimeError("All datetime values invalid.")

        ####################################################
        # Numeric enforcement
        ####################################################

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        if df.empty:
            raise RuntimeError("All numeric rows invalid.")

        ####################################################
        # Soft invariants (DO NOT over-reject)
        ####################################################

        df = df[df["high"] >= df["low"]]

        if df.empty:
            raise RuntimeError("Price invariant violation.")

        ####################################################

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        return df

    ############################################################

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
                    "Market data served → provider=%s ticker=%s rows=%s",
                    name,
                    ticker,
                    len(df)
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
