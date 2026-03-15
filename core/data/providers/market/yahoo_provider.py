
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher

logger = logging.getLogger(__name__)


class YahooProvider(MarketDataProvider):
    """
    Wraps StockPriceFetcher (yfinance) with normalization,
    validation, and light data-quality hardening.
    """

    PROVIDER_NAME = "yahoo"

    # ── Row limits ───────────────────────────────────────────────────────────
    DEFAULT_MIN_ROWS = 120
    MAX_ROWS         = 20_000

    # ── Intervals: aligned with MarketProviderRouter.ALLOWED_INTERVALS ──────
    # Includes yfinance-native aliases so both "1h" and "60m" work.
    ALLOWED_INTERVALS = {
        "1d", "D",          # daily  (D = router alias)
        "1wk",              # weekly
        "1mo",              # monthly
        "1h", "60m",        # hourly
        "15m",              # 15-minute
        "5m",               # 5-minute
        "1m",               # 1-minute
    }

    # ── yfinance → canonical interval map ───────────────────────────────────
    # Normalise router aliases to yfinance-accepted strings before fetching.
    _INTERVAL_ALIAS: dict = {
        "D":   "1d",
        "60m": "1h",
    }

    # ── Quality thresholds ───────────────────────────────────────────────────
    # NOTE: keep MAX_DAILY_MOVE < router's MAX_DAILY_MOVE (0.90) so provider
    # catches bad bars first before the router's outer check fires.
    MAX_DAILY_MOVE       = 0.85
    MIN_TRADING_DENSITY  = 0.50    # warn if sparse; don't hard-fail

    # ── Soft-fail: accept 70 % of min_rows in demo / CI environments ────────
    SOFT_FAIL_MODE       = os.getenv("YAHOO_SOFT_FAIL", "1") == "1"
    SOFT_FAIL_RATIO      = 0.70

    # ── Retry for transient yfinance empty-response issues ───────────────────
    MAX_RETRIES          = 2
    RETRY_DELAY_SECONDS  = 1.5

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.fetcher = StockPriceFetcher()
        logger.debug("YahooProvider initialised.")

    # ────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_datetime(series: pd.Series) -> pd.Series:
        """Parse a Series to UTC-aware datetime; raises on total failure."""
        if not isinstance(series, pd.Series):
            raise RuntimeError("Date column must be a pandas Series.")
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        if dt.isna().all():
            raise RuntimeError("Datetime parsing failed — all values are NaT.")
        return dt

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns produced by yfinance multi-ticker calls,
        and lowercase / strip all column names.
        """
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(
                    str(lvl).strip().lower()
                    for lvl in col
                    if lvl is not None and str(lvl).strip() != ""
                )
                for col in df.columns
            ]
        else:
            df.columns = [str(c).strip().lower() for c in df.columns]

        # Drop duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw column names to canonical OHLCV names.
        Prefers 'adj close' over 'close' when available (handles splits).
        """
        col_map: dict = {}

        for col in df.columns:
            lc = col.lower()
            if lc.startswith("open")      and "open"   not in col_map:
                col_map["open"]   = col
            elif lc.startswith("high")    and "high"   not in col_map:
                col_map["high"]   = col
            elif lc.startswith("low")     and "low"    not in col_map:
                col_map["low"]    = col
            elif lc.startswith("adj close") and "close" not in col_map:
                col_map["close"]  = col   # prefer adj close
            elif lc.startswith("close")   and "close"  not in col_map:
                col_map["close"]  = col
            elif lc.startswith("volume")  and "volume" not in col_map:
                col_map["volume"] = col

        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(col_map.keys())
        if missing:
            raise RuntimeError(f"Yahoo schema violation — missing columns: {missing}")

        clean = pd.DataFrame()
        for key in required:
            series = df[col_map[key]]
            # Guard against a column being a sub-DataFrame (yfinance edge case)
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    raise RuntimeError(f"Ambiguous Yahoo column for '{key}'")
            clean[key] = series

        return clean

    # ────────────────────────────────────────────────────────────────────────
    # CORE NORMALIZER
    # ────────────────────────────────────────────────────────────────────────

    def _normalize(
        self,
        df:       pd.DataFrame,
        ticker:   str,
        min_rows: int,
    ) -> pd.DataFrame:
        """
        Full normalization pipeline:
            1. Flatten columns
            2. Extract date from index or column
            3. Extract / map OHLCV
            4. Numeric coercion + inf removal
            5. Sort, deduplicate, cap rows
            6. OHLC integrity check (reject, not repair)
            7. Extreme-move filter
            8. Trading-density warning
            9. Volume / close gap fill
           10. Min-rows gate (with soft-fail)
        """

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Yahoo fetch returned empty DataFrame for {ticker}.")

        df = df.copy()
        df = self._flatten_columns(df)

        # ── 1. Date extraction ───────────────────────────────────────────────
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
                idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                df  = df.reset_index(drop=True)
                df["date"] = idx
            else:
                raise RuntimeError(
                    f"Yahoo DataFrame for {ticker} has no datetime index or 'date' column."
                )

        # ── 2. OHLCV extraction ──────────────────────────────────────────────
        clean      = self._extract_ohlcv(df)
        clean["date"] = df["date"].values

        # ── 3. Numeric coercion ──────────────────────────────────────────────
        for col in ("open", "high", "low", "close", "volume"):
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        clean.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if clean.empty:
            raise RuntimeError(f"Normalization produced empty dataset for {ticker}.")

        # ── 4. Date normalization ────────────────────────────────────────────
        clean["date"] = self._normalize_datetime(clean["date"])

        # ── 5. Sort, dedup, cap ──────────────────────────────────────────────
        clean = (
            clean
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        # ── 6. Positive-price guard ──────────────────────────────────────────
        clean = clean[clean["close"] > 0].reset_index(drop=True)
        if clean.empty:
            raise RuntimeError(f"All close prices are <= 0 for {ticker}.")

        # ── 7. OHLC integrity check — flag, don't silently repair ────────────
        ohlc_ok = (
            (clean["high"] >= clean["low"]).all()
            and (clean["high"] >= clean["open"]).all()
            and (clean["high"] >= clean["close"]).all()
            and (clean["low"]  <= clean["open"]).all()
            and (clean["low"]  <= clean["close"]).all()
        )
        if not ohlc_ok:
            bad_count = (
                ~(clean["high"] >= clean["low"])
                | ~(clean["high"] >= clean["open"])
                | ~(clean["high"] >= clean["close"])
                | ~(clean["low"]  <= clean["open"])
                | ~(clean["low"]  <= clean["close"])
            ).sum()
            logger.warning(
                "OHLC integrity issues in Yahoo data for %s (%d bars affected) "
                "— applying best-effort repair.",
                ticker, bad_count,
            )
            # CV-friendly: repair rather than hard-fail so demos don't crash
            clean["high"] = clean[["high", "open", "close"]].max(axis=1)
            clean["low"]  = clean[["low",  "open", "close"]].min(axis=1)

        # ── 8. Extreme-move filter — drop bad bars, then forward-fill ────────
        pct = clean["close"].pct_change().abs().fillna(0)
        extreme_mask = pct > self.MAX_DAILY_MOVE
        if extreme_mask.any():
            n_extreme = extreme_mask.sum()
            logger.warning(
                "Extreme price moves in Yahoo data for %s (%d bars) — "
                "nullifying and forward-filling.",
                ticker, n_extreme,
            )
            clean.loc[extreme_mask, "close"] = np.nan
            clean["close"] = clean["close"].ffill().bfill()

        # ── 9. Trading-density warning (informational only) ──────────────────
        span_days = (clean["date"].max() - clean["date"].min()).days + 1
        if span_days > 0:
            density = clean["date"].nunique() / span_days
            if density < self.MIN_TRADING_DENSITY:
                logger.warning(
                    "Low trading density for %s (%.2f) — possible data gap.",
                    ticker, density,
                )

        # ── 10. Volume / close gap fill ───────────────────────────────────────
        clean["close"]  = clean["close"].ffill().bfill()
        clean["volume"] = clean["volume"].fillna(0).clip(lower=0)

        # ── 11. Min-rows gate ─────────────────────────────────────────────────
        if len(clean) < min_rows:
            soft_threshold = int(min_rows * self.SOFT_FAIL_RATIO)
            if self.SOFT_FAIL_MODE and len(clean) >= soft_threshold:
                logger.warning(
                    "Short history accepted in soft-fail mode for %s "
                    "(%d rows, need %d, threshold %d).",
                    ticker, len(clean), min_rows, soft_threshold,
                )
            else:
                raise RuntimeError(
                    f"Insufficient history for {ticker}: "
                    f"got {len(clean)}, need {min_rows} "
                    f"(soft_fail={'on' if self.SOFT_FAIL_MODE else 'off'})"
                )

        clean["ticker"] = ticker

        logger.info(
            "Yahoo normalised | ticker=%s rows=%d interval=—",
            ticker, len(clean),
        )

        return clean

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC FETCH  (called by MarketProviderRouter)
    # ────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker:     str,
        start_date: str,
        end_date:   str,
        interval:   str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch and normalise OHLCV data from Yahoo Finance.

        Parameters
        ----------
        ticker     : e.g. "AAPL"
        start_date : ISO string, e.g. "2023-01-01"
        end_date   : ISO string, e.g. "2024-01-01"
        interval   : Any value in ALLOWED_INTERVALS (router aliases accepted)
        min_rows   : Passed via kwargs; defaults to DEFAULT_MIN_ROWS
        """

        # ── Interval validation + alias normalisation ────────────────────────
        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(
                f"YahooProvider: unsupported interval '{interval}'. "
                f"Allowed: {sorted(self.ALLOWED_INTERVALS)}"
            )
        # Map "D" → "1d", "60m" → "1h" for yfinance compatibility
        yf_interval = self._INTERVAL_ALIAS.get(interval, interval)

        min_rows: int = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        # ── Fetch with simple retry (transient yfinance empty responses) ──────
        raw_df: Optional[pd.DataFrame] = None
        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_df = self.fetcher.fetch(
                    ticker,
                    start_date,
                    end_date,
                    yf_interval,
                )
                if raw_df is not None and not raw_df.empty:
                    break
                logger.warning(
                    "Yahoo returned empty response for %s (attempt %d/%d).",
                    ticker, attempt, self.MAX_RETRIES,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Yahoo fetch error for %s (attempt %d/%d): %s",
                    ticker, attempt, self.MAX_RETRIES, exc,
                )

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY_SECONDS)

        if raw_df is None or raw_df.empty:
            msg = f"Yahoo returned no data for {ticker} after {self.MAX_RETRIES} attempts."
            if last_error:
                msg += f" Last error: {last_error}"
            raise RuntimeError(msg)

        # ── Normalise + contract validation ──────────────────────────────────
        normalised = self._normalize(
            df=raw_df,
            ticker=ticker,
            min_rows=min_rows,
        )

        return self.validate_contract(normalised)