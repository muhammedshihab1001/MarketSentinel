import os
import logging
import pandas as pd
import time

from core.data.providers.market.yahoo_provider import YahooProvider

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
        1️⃣ Yahoo (PRIMARY — most stable)
        2️⃣ TwelveData (fallback)
        3️⃣ AlphaVantage (last resort)

    Design Goals:
        ✔ Prefer statistical stability
        ✔ Never silently accept garbage
        ✔ Prefer degraded data over crash
    """

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    MIN_ACCEPTABLE_ROWS = 50
    MAX_DAILY_MOVE = 0.90
    PROVIDER_TIMEOUT_WARN = 8.0

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

        def register(name, builder, api_key_env=None):

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

                logger.info("Provider registered → %s", name)

            except Exception as e:

                logger.warning(
                    "Provider disabled → %s | reason=%s",
                    name,
                    str(e)
                )

        ####################################################
        # 🔥 NEW PRIORITY ORDER (CRITICAL FIX)
        ####################################################

        # PRIMARY — MOST STABLE
        register("yahoo", YahooProvider)

        # fallback
        register(
            "twelvedata",
            TwelveDataProvider,
            "TWELVEDATA_API_KEY"
        )

        # last resort
        register(
            "alphavantage",
            AlphaVantageProvider,
            "ALPHAVANTAGE_API_KEY"
        )

    ############################################################

    @classmethod
    def _validate_interval(cls, interval):

        if interval not in cls.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

    ############################################################
    # INSTITUTIONAL SANITIZER
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

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        if df.empty:
            raise RuntimeError("All datetime values invalid.")

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        if df.empty:
            raise RuntimeError("All numeric rows invalid.")

        df = df[df["high"] >= df["low"]]

        if df.empty:
            raise RuntimeError("Price invariant violation.")

        jumps = df["close"].pct_change().abs()

        if (jumps > cls.MAX_DAILY_MOVE).any():
            raise RuntimeError("Extreme price jump detected.")

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) < cls.MIN_ACCEPTABLE_ROWS:
            raise RuntimeError(
                f"Too few rows from provider ({len(df)})"
            )

        return df

    ############################################################

    def fetch(self, ticker, start, end, interval):

        self._validate_interval(interval)

        last_error = None

        for name, provider in self.providers:

            start_time = time.time()

            try:

                df = provider.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                latency = time.time() - start_time

                if latency > self.PROVIDER_TIMEOUT_WARN:
                    logger.warning(
                        "Slow provider detected → %s (%.2fs)",
                        name,
                        latency
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
