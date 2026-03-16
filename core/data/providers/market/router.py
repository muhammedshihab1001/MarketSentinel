import os
import logging
import time
import threading
from contextlib import nullcontext
from typing import Dict, Any, Optional

import pandas as pd

from core.data.providers.market.yahoo_provider import YahooProvider

try:
    from core.data.providers.market.alphavantage_provider import AlphaVantageProvider
except Exception:
    AlphaVantageProvider = None

try:
    from core.data.providers.market.twelvedata_provider import TwelveDataProvider
except Exception:
    TwelveDataProvider = None


logger = logging.getLogger(__name__)


class MarketProviderRouter:
    """
    Routes market-data fetch requests across providers with
    automatic sequential fallback, cooldown tracking, response
    validation, and per-provider health statistics.

    Fallback chain: Yahoo → AlphaVantage → TwelveData
    """

    REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

    DEFAULT_MIN_ROWS = 50
    MAX_DAILY_MOVE = 0.90

    PROVIDER_TIMEOUT_WARN = 8.0
    FAILURE_COOLDOWN = 60

    YAHOO_MAX_CONCURRENT = int(os.getenv("YAHOO_MAX_CONCURRENT", 1))

    # NEW: Yahoo request throttle (seconds between requests)
    YAHOO_MIN_INTERVAL = float(os.getenv("YAHOO_MIN_INTERVAL", 1.0))

    ALLOWED_INTERVALS = {"1d", "D", "1h", "60m", "15m", "5m", "1m"}

    def __init__(self) -> None:

        self.providers: list = []
        self._provider_failures: Dict[str, float] = {}
        self._provider_stats: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.Lock()

        self._yahoo_semaphore = threading.Semaphore(self.YAHOO_MAX_CONCURRENT)

        # NEW: Yahoo throttle state
        self._last_yahoo_request = 0.0

        self._register_providers()

        if not self.providers:
            raise RuntimeError("No market providers available.")

        self._single_provider_mode = len(self.providers) == 1

        logger.info(
            "Market router ready | priority=%s | single_provider=%s | yahoo_max_concurrent=%s",
            [p[0] for p in self.providers],
            self._single_provider_mode,
            self.YAHOO_MAX_CONCURRENT,
        )

    def _register_providers(self) -> None:

        def register(name: str, builder, api_key_env: Optional[str] = None):

            if builder is None:
                logger.debug("Provider class unavailable → %s", name)
                return

            if api_key_env and not os.getenv(api_key_env):
                logger.warning("Provider skipped → %s | missing env var: %s", name, api_key_env)
                return

            try:
                provider = builder()

                self.providers.append((name, provider))

                self._provider_stats[name] = {
                    "success": 0,
                    "failure": 0,
                    "avg_latency": 0.0,
                    "last_failure": None,
                }

                logger.info("Provider registered → %s", name)

            except Exception as exc:
                logger.warning("Provider disabled → %s | reason=%s", name, exc)

        register("yahoo", YahooProvider)
        register("alphavantage", AlphaVantageProvider, "ALPHAVANTAGE_API_KEY")
        register("twelvedata", TwelveDataProvider, "TWELVEDATA_API_KEY")

    @classmethod
    def _validate_interval(cls, interval: str):

        if interval not in cls.ALLOWED_INTERVALS:
            raise ValueError(
                f"Unsupported interval: '{interval}'. Allowed: {sorted(cls.ALLOWED_INTERVALS)}"
            )

    def _provider_allowed(self, name: str) -> bool:

        if self._single_provider_mode:
            return True

        with self._lock:
            last_fail = self._provider_failures.get(name)

            if not last_fail:
                return True

            return (time.time() - last_fail) > self.FAILURE_COOLDOWN

    def _record_failure(self, name: str) -> None:

        if self._single_provider_mode:
            return

        with self._lock:
            now = time.time()

            self._provider_failures[name] = now

            self._provider_stats[name]["failure"] += 1
            self._provider_stats[name]["last_failure"] = now

    def _validate_response(self, df: pd.DataFrame, ticker: str, min_rows: int) -> pd.DataFrame:

        if df is None:
            raise RuntimeError(f"Provider returned None for {ticker}")

        if not hasattr(df, "columns"):
            raise RuntimeError(f"Invalid response type for {ticker}")

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(f"Schema violation for {ticker}. Missing columns: {missing}")

        if len(df) == 0:
            raise RuntimeError(f"Empty dataset returned for {ticker}")

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])

        if len(df) < min_rows:
            raise RuntimeError(f"Insufficient rows for {ticker}: got {len(df)}, need {min_rows}")

        ohlc_ok = (
            (df["high"] >= df["low"]).all()
            and (df["high"] >= df["open"]).all()
            and (df["high"] >= df["close"]).all()
            and (df["low"] <= df["open"]).all()
            and (df["low"] <= df["close"]).all()
        )

        if not ohlc_ok:
            raise RuntimeError(f"OHLC integrity violation for {ticker}")

        max_move = df["close"].pct_change().abs().dropna().max()

        if max_move > self.MAX_DAILY_MOVE:
            raise RuntimeError(f"Extreme price move ({max_move:.1%}) detected for {ticker}")

        return df.reset_index(drop=True)

    def _throttle_yahoo(self):

        with self._lock:

            now = time.time()

            elapsed = now - self._last_yahoo_request

            wait = self.YAHOO_MIN_INTERVAL - elapsed

            if wait > 0:
                time.sleep(wait)

            self._last_yahoo_request = time.time()

    def _execute_provider(
        self,
        name: str,
        provider,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        min_rows: int,
    ) -> pd.DataFrame:

        start_time = time.time()

        ctx = self._yahoo_semaphore if name == "yahoo" else nullcontext()

        with ctx:

            if name == "yahoo":
                self._throttle_yahoo()

            df = provider.fetch(ticker, start, end, interval, min_rows=min_rows)

        latency = time.time() - start_time

        if latency > self.PROVIDER_TIMEOUT_WARN:
            logger.warning("Slow provider → %s (%.2fs)", name, latency)

        df = self._validate_response(df, ticker, min_rows)

        with self._lock:

            stats = self._provider_stats[name]

            stats["success"] += 1

            n = stats["success"]

            stats["avg_latency"] = (
                (stats["avg_latency"] * (n - 1) + latency) / n
            )

        logger.info(
            "Market data served → provider=%s ticker=%s rows=%d latency=%.2fs",
            name,
            ticker,
            len(df),
            latency,
        )

        return df.copy()

    def fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        min_rows: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:

        self._validate_interval(interval)

        if min_rows is None:
            min_rows = self.DEFAULT_MIN_ROWS

        if provider is not None:

            requested = provider.lower()

            for name, provider_obj in self.providers:

                if name == requested:

                    if not self._provider_allowed(name):
                        raise RuntimeError(f"Provider '{name}' is in cooldown.")

                    try:

                        return self._execute_provider(
                            name,
                            provider_obj,
                            ticker,
                            start,
                            end,
                            interval,
                            min_rows,
                        )

                    except Exception:

                        self._record_failure(name)

                        raise

            raise RuntimeError(f"Requested provider not registered: '{requested}'")

        errors: Dict[str, str] = {}

        for name, provider_obj in self.providers:

            if not self._provider_allowed(name):
                logger.debug("Provider in cooldown → %s", name)
                continue

            try:

                return self._execute_provider(
                    name,
                    provider_obj,
                    ticker,
                    start,
                    end,
                    interval,
                    min_rows,
                )

            except Exception as exc:

                errors[name] = str(exc)

                logger.warning(
                    "Provider failed → %s | ticker=%s | error=%s",
                    name,
                    ticker,
                    exc,
                )

                self._record_failure(name)

        raise RuntimeError(f"All providers failed for '{ticker}'. Errors: {errors}")