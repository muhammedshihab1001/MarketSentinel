import os
import logging
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    DEFAULT_MIN_ROWS = 50
    MAX_DAILY_MOVE = 0.90
    PROVIDER_TIMEOUT_WARN = 8.0
    MAX_PROVIDER_WORKERS = 3
    FAILURE_COOLDOWN = 30

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
        self._provider_failures = {}
        self._lock = threading.Lock()

        self._register_providers()

        if not self.providers:
            raise RuntimeError("No market providers available.")

        self._single_provider_mode = len(self.providers) == 1

        logger.info(
            "Market router ready | priority=%s | single_provider=%s",
            [p[0] for p in self.providers],
            self._single_provider_mode
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

        register("yahoo", YahooProvider)
        register("twelvedata", TwelveDataProvider, "TWELVEDATA_API_KEY")
        register("alphavantage", AlphaVantageProvider, "ALPHAVANTAGE_API_KEY")

    ############################################################

    @classmethod
    def _validate_interval(cls, interval):
        if interval not in cls.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

    ############################################################
    # FAILURE TRACKING
    ############################################################

    def _provider_allowed(self, name):

        if self._single_provider_mode:
            return True  # 🔥 disable cooldown logic if only one provider

        with self._lock:
            last_fail = self._provider_failures.get(name)
            if not last_fail:
                return True
            if time.time() - last_fail > self.FAILURE_COOLDOWN:
                return True
            return False

    def _record_failure(self, name):

        if self._single_provider_mode:
            return  # 🔥 no cooldown in single-provider mode

        with self._lock:
            self._provider_failures[name] = time.time()

    ############################################################

    def fetch(
        self,
        ticker,
        start,
        end,
        interval,
        min_rows=None,
        provider: str | None = None
    ):

        self._validate_interval(interval)

        if min_rows is None:
            min_rows = self.DEFAULT_MIN_ROWS

        if provider is not None:
            provider = provider.lower()

            match = [
                (name, p)
                for name, p in self.providers
                if name == provider
            ]

            if not match:
                raise RuntimeError(f"Requested provider not available: {provider}")

            name, provider_obj = match[0]

            if not self._provider_allowed(name):
                raise RuntimeError(f"Provider {name} in cooldown.")

            try:
                return self._attempt_provider(
                    name,
                    provider_obj,
                    ticker,
                    start,
                    end,
                    interval,
                    min_rows
                )
            except Exception as e:
                self._record_failure(name)
                raise

        eligible = [
            (name, provider_obj)
            for name, provider_obj in self.providers
            if self._provider_allowed(name)
        ]

        if not eligible:
            raise RuntimeError("All providers in cooldown.")

        futures = {}

        with ThreadPoolExecutor(max_workers=self.MAX_PROVIDER_WORKERS) as executor:

            for name, provider_obj in eligible:

                futures[
                    executor.submit(
                        self._attempt_provider,
                        name,
                        provider_obj,
                        ticker,
                        start,
                        end,
                        interval,
                        min_rows
                    )
                ] = name

            for future in as_completed(futures):

                name = futures[future]

                try:
                    return future.result()

                except Exception as e:
                    logger.warning(
                        "Provider failed → %s | ticker=%s | error=%s",
                        name,
                        ticker,
                        str(e)
                    )
                    self._record_failure(name)

        raise RuntimeError(f"All market providers failed for {ticker}")

    ############################################################

    def _attempt_provider(
        self,
        name,
        provider,
        ticker,
        start,
        end,
        interval,
        min_rows
    ):

        start_time = time.time()

        df = provider.fetch(
            ticker,
            start,
            end,
            interval,
            min_rows=min_rows
        )

        latency = time.time() - start_time

        if latency > self.PROVIDER_TIMEOUT_WARN:
            logger.warning(
                "Slow provider detected → %s (%.2fs)",
                name,
                latency
            )

        logger.info(
            "Market data served → provider=%s ticker=%s rows=%s",
            name,
            ticker,
            len(df) if df is not None else 0
        )

        return df.copy()