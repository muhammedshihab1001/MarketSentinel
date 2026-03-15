
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

    # ── Schema ──────────────────────────────────────────────────────────────
    REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

    # ── Validation thresholds ───────────────────────────────────────────────
    DEFAULT_MIN_ROWS   = 50
    MAX_DAILY_MOVE     = 0.90          # 90 % single-bar move → reject bar

    # ── Timing / rate-limiting ──────────────────────────────────────────────
    PROVIDER_TIMEOUT_WARN = 8.0        # seconds before a slow-provider warning
    FAILURE_COOLDOWN      = 60         # seconds a failed provider stays on ice
    YAHOO_MAX_CONCURRENT  = int(os.getenv("YAHOO_MAX_CONCURRENT", 1))

    # ── Supported intervals ─────────────────────────────────────────────────
    ALLOWED_INTERVALS = {"1d", "D", "1h", "60m", "15m", "5m", "1m"}

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.providers: list = []
        self._provider_failures: Dict[str, float] = {}
        self._provider_stats:   Dict[str, Dict[str, Any]] = {}

        self._lock             = threading.Lock()
        self._yahoo_semaphore  = threading.Semaphore(self.YAHOO_MAX_CONCURRENT)

        self._register_providers()

        if not self.providers:
            raise RuntimeError("No market providers available.")

        self._single_provider_mode = len(self.providers) == 1

        logger.info(
            "Market router ready | priority=%s | single_provider=%s"
            " | yahoo_max_concurrent=%s",
            [p[0] for p in self.providers],
            self._single_provider_mode,
            self.YAHOO_MAX_CONCURRENT,
        )

    # ────────────────────────────────────────────────────────────────────────
    # PROVIDER REGISTRATION  (order = priority)
    # ────────────────────────────────────────────────────────────────────────

    def _register_providers(self) -> None:
        """Register providers in strict priority order."""

        def register(name: str, builder, api_key_env: Optional[str] = None) -> None:
            if builder is None:
                logger.debug("Provider class unavailable → %s (import failed)", name)
                return
            if api_key_env and not os.getenv(api_key_env):
                logger.warning(
                    "Provider skipped → %s | missing env var: %s",
                    name, api_key_env,
                )
                return
            try:
                provider = builder()
                self.providers.append((name, provider))
                self._provider_stats[name] = {
                    "success":      0,
                    "failure":      0,
                    "avg_latency":  0.0,
                    "last_failure": None,
                }
                logger.info("Provider registered → %s", name)
            except Exception as exc:
                logger.warning(
                    "Provider disabled → %s | reason=%s", name, exc
                )

        # ── Priority order (DO NOT reorder) ─────────────────────────────────
        register("yahoo",        YahooProvider)                                    # 1 — primary
        register("alphavantage", AlphaVantageProvider, "ALPHAVANTAGE_API_KEY")    # 2 — fallback #1
        register("twelvedata",   TwelveDataProvider,   "TWELVEDATA_API_KEY")      # 3 — fallback #2

    # ────────────────────────────────────────────────────────────────────────
    # INTERVAL VALIDATION
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def _validate_interval(cls, interval: str) -> None:
        if interval not in cls.ALLOWED_INTERVALS:
            raise ValueError(
                f"Unsupported interval: '{interval}'. "
                f"Allowed: {sorted(cls.ALLOWED_INTERVALS)}"
            )

    # ────────────────────────────────────────────────────────────────────────
    # FAILURE TRACKING
    # ────────────────────────────────────────────────────────────────────────

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
            self._provider_stats[name]["failure"]      += 1
            self._provider_stats[name]["last_failure"]  = now

    # ────────────────────────────────────────────────────────────────────────
    # RESPONSE VALIDATION
    # ────────────────────────────────────────────────────────────────────────

    def _validate_response(self, df: pd.DataFrame, ticker: str, min_rows: int) -> pd.DataFrame:
        """
        Validates schema, row count, OHLC integrity, and extreme-move guard.
        Raises RuntimeError for any violation — never silently mutates data.
        """

        # ── Existence checks ─────────────────────────────────────────────────
        if df is None:
            raise RuntimeError(f"Provider returned None for {ticker}")
        if not hasattr(df, "columns"):
            raise RuntimeError(f"Invalid response type for {ticker} — expected DataFrame")

        # ── Schema check ─────────────────────────────────────────────────────
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Schema violation for {ticker}. Missing columns: {missing}"
            )

        # ── Empty guard ──────────────────────────────────────────────────────
        if len(df) == 0:
            raise RuntimeError(f"Empty dataset returned for {ticker}")

        # ── Numeric coercion ─────────────────────────────────────────────────
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        # ── Minimum rows check (after NaN-drop) ──────────────────────────────
        if len(df) < min_rows:
            raise RuntimeError(
                f"Insufficient rows for {ticker}: got {len(df)}, need {min_rows}"
            )

        # ── Full OHLC integrity check ─────────────────────────────────────────
        ohlc_ok = (
            (df["high"] >= df["low"]).all()
            and (df["high"] >= df["open"]).all()
            and (df["high"] >= df["close"]).all()
            and (df["low"]  <= df["open"]).all()
            and (df["low"]  <= df["close"]).all()
        )
        if not ohlc_ok:
            raise RuntimeError(f"OHLC integrity violation for {ticker}")

        # ── Extreme-move guard — REJECT, never mutate ─────────────────────────
        max_move = df["close"].pct_change().abs().dropna().max()
        if max_move > self.MAX_DAILY_MOVE:
            raise RuntimeError(
                f"Extreme price move ({max_move:.1%}) detected for {ticker} "
                f"— rejecting provider response (threshold={self.MAX_DAILY_MOVE:.0%})"
            )

        return df.reset_index(drop=True)

    # ────────────────────────────────────────────────────────────────────────
    # PROVIDER EXECUTION
    # ────────────────────────────────────────────────────────────────────────

    def _execute_provider(
        self,
        name:      str,
        provider,
        ticker:    str,
        start:     str,
        end:       str,
        interval:  str,
        min_rows:  int,
    ) -> pd.DataFrame:

        start_time = time.time()

        ctx = self._yahoo_semaphore if name == "yahoo" else nullcontext()
        with ctx:
            df = provider.fetch(ticker, start, end, interval, min_rows=min_rows)

        latency = time.time() - start_time

        if latency > self.PROVIDER_TIMEOUT_WARN:
            logger.warning("Slow provider → %s (%.2fs)", name, latency)

        # Pass min_rows into validation so the floor is always enforced
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
            name, ticker, len(df), latency,
        )

        return df.copy()

    # ────────────────────────────────────────────────────────────────────────
    # MAIN FETCH ENTRY POINT
    # ────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker:   str,
        start:    str,
        end:      str,
        interval: str,
        min_rows: Optional[int] = None,
        provider: Optional[str] = None,   # Python 3.8-safe (no X | Y syntax)
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for *ticker* between *start* and *end*.

        Parameters
        ----------
        ticker   : Stock symbol, e.g. "AAPL"
        start    : ISO date string, e.g. "2023-01-01"
        end      : ISO date string, e.g. "2024-01-01"
        interval : One of ALLOWED_INTERVALS
        min_rows : Minimum acceptable bar count (default: DEFAULT_MIN_ROWS)
        provider : Pin to a specific provider by name (optional).
                   If omitted, the priority fallback chain is used.
        """

        self._validate_interval(interval)

        if min_rows is None:
            min_rows = self.DEFAULT_MIN_ROWS

        # ── Pinned-provider path ─────────────────────────────────────────────
        if provider is not None:
            requested = provider.lower()
            for name, provider_obj in self.providers:
                if name == requested:
                    if not self._provider_allowed(name):
                        raise RuntimeError(f"Provider '{name}' is in cooldown.")
                    try:
                        return self._execute_provider(
                            name, provider_obj, ticker, start, end, interval, min_rows
                        )
                    except Exception:
                        self._record_failure(name)
                        raise
            raise RuntimeError(f"Requested provider not registered: '{requested}'")

        # ── Sequential fallback chain ─────────────────────────────────────────
        errors: Dict[str, str] = {}
        for name, provider_obj in self.providers:
            if not self._provider_allowed(name):
                logger.debug("Provider in cooldown, skipping → %s", name)
                continue
            try:
                return self._execute_provider(
                    name, provider_obj, ticker, start, end, interval, min_rows
                )
            except Exception as exc:
                errors[name] = str(exc)
                logger.warning(
                    "Provider failed → %s | ticker=%s | error=%s",
                    name, ticker, exc,
                )
                self._record_failure(name)

        raise RuntimeError(
            f"All providers failed for '{ticker}'. Errors: {errors}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # HEALTH & DIAGNOSTICS
    # ────────────────────────────────────────────────────────────────────────

    def provider_health(self) -> Dict[str, Any]:
        """Return a snapshot of per-provider success/failure/latency stats."""
        with self._lock:
            snapshot: Dict[str, Any] = {}
            for name, stats in self._provider_stats.items():
                total        = stats["success"] + stats["failure"]
                failure_rate = stats["failure"] / total if total > 0 else 0.0
                in_cooldown  = (
                    False
                    if self._single_provider_mode
                    else not self._provider_allowed(name)
                )
                snapshot[name] = {
                    "success":      stats["success"],
                    "failure":      stats["failure"],
                    "failure_rate": round(failure_rate, 4),
                    "avg_latency":  round(stats["avg_latency"], 4),
                    "last_failure": stats["last_failure"],
                    "in_cooldown":  in_cooldown,
                }
            return snapshot

    def best_provider(self) -> Optional[str]:
        """
        Return the name of the provider with the lowest average latency
        among those that are currently healthy (not in cooldown and
        have at least one successful call).

        Useful for monitoring dashboards and logging.
        """
        health = self.provider_health()
        candidates = [
            (name, info["avg_latency"])
            for name, info in health.items()
            if info["success"] > 0 and not info["in_cooldown"]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    def reset_provider_stats(self) -> None:
        """
        Reset all failure cooldowns and stats counters.
        Handy for testing or after a known outage has recovered.
        """
        with self._lock:
            self._provider_failures.clear()
            for name in self._provider_stats:
                self._provider_stats[name] = {
                    "success":      0,
                    "failure":      0,
                    "avg_latency":  0.0,
                    "last_failure": None,
                }
        logger.info("Provider stats and cooldowns reset.")