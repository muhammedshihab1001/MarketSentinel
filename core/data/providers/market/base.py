

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


# Floating-point tolerance for OHLC invariant checks.
# Tiny adj_close substitutions can produce micro-violations (e.g. 1e-8 diff).
_OHLC_TOLERANCE = 1e-6


class MarketDataProvider(ABC):
    """
    Base contract for ALL market data providers.

    Subclasses must implement fetch() and return a DataFrame
    that passes validate_contract().

    Provider chain:
        Yahoo Finance (primary) → AlphaVantage → TwelveData
    """

    # ── Required output schema ───────────────────────────────────────────────
    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticker",
    }

    # ── Provider identity (override in each subclass) ────────────────────────
    PROVIDER_NAME: str = "unknown"

    # ────────────────────────────────────────────────────────────────────────
    # ABSTRACT INTERFACE
    # ────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def fetch(
        self,
        ticker:     str,
        start_date: str,
        end_date:   str,
        interval:   str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for *ticker* and return a validated DataFrame.

        All implementations must:
            - Return a DataFrame with exactly REQUIRED_COLUMNS (plus extras OK)
            - Call validate_contract() before returning
            - Ignore unknown **kwargs safely (forward-compatibility)

        Supported kwargs (providers may honour or ignore):
            min_rows (int)  : minimum acceptable bar count
            retries  (int)  : override default retry count
        """
        raise NotImplementedError

    # ────────────────────────────────────────────────────────────────────────
    # CONTRACT VALIDATOR  (called by every provider before returning data)
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_contract(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final data-quality gate shared by all providers.

        Checks (in order):
            1. Non-null DataFrame
            2. Required column presence
            3. UTC datetime parsing
            4. Numeric coercion + NaN / inf checks
            5. OHLC price invariants (with float tolerance)
            6. Duplicate (ticker, date) rows

        Returns the validated (and lightly normalised) DataFrame.
        Raises RuntimeError with a descriptive message on any violation.
        """

        # ── 1. Existence ─────────────────────────────────────────────────────
        if df is None:
            raise RuntimeError("Provider returned None instead of a DataFrame.")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(
                f"Provider did not return a DataFrame (got {type(df).__name__})."
            )

        if df.empty:
            raise RuntimeError("Provider returned an empty DataFrame.")

        # ── 2. Schema check ──────────────────────────────────────────────────
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Provider contract violated — missing columns: {missing}"
            )

        df = df.copy()

        # ── 3. UTC datetime ──────────────────────────────────────────────────
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        except Exception as exc:
            raise RuntimeError(f"Date column could not be parsed: {exc}") from exc

        if df["date"].isna().any():
            n_bad = df["date"].isna().sum()
            raise RuntimeError(
                f"Date column has {n_bad} unparseable value(s) after coercion."
            )

        # ── 4. Numeric coercion + NaN / inf checks ───────────────────────────
        numeric_cols = ("open", "high", "low", "close", "volume")

        for col in numeric_cols:
            # Coerce first — gives a useful error rather than a raw crash
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # NaN check (separate from inf — different root causes)
            if df[col].isna().any():
                n_nan = df[col].isna().sum()
                raise RuntimeError(
                    f"Column '{col}' has {n_nan} NaN value(s) after coercion."
                )

            # Inf check
            arr = df[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(arr)):
                n_inf = (~np.isfinite(arr)).sum()
                raise RuntimeError(
                    f"Column '{col}' has {n_inf} non-finite (inf) value(s)."
                )

        # ── 5. OHLC price invariants (float-tolerance aware) ─────────────────
        # Use a small tolerance to avoid false positives from adj_close
        # substitution which can produce micro floating-point differences.
        tol = _OHLC_TOLERANCE

        high_vs_open  = df["high"] + tol < df["open"]
        high_vs_close = df["high"] + tol < df["close"]
        low_vs_open   = df["low"]  - tol > df["open"]
        low_vs_close  = df["low"]  - tol > df["close"]
        high_vs_low   = df["high"] + tol < df["low"]

        violations = (
            high_vs_open | high_vs_close |
            low_vs_open  | low_vs_close  |
            high_vs_low
        )

        if violations.any():
            n_bad = violations.sum()
            raise RuntimeError(
                f"OHLC price invariant violated in {n_bad} bar(s). "
                f"Ensure high >= open/close and low <= open/close."
            )

        # ── 6. Duplicate (ticker, date) rows ─────────────────────────────────
        dupes = df.duplicated(subset=["ticker", "date"])
        if dupes.any():
            n_dupes = dupes.sum()
            raise RuntimeError(
                f"Provider output has {n_dupes} duplicate (ticker, date) row(s)."
            )

        return df

    # ────────────────────────────────────────────────────────────────────────
    # OPTIONAL HELPERS  (available to all subclasses)
    # ────────────────────────────────────────────────────────────────────────

    def provider_info(self) -> dict:
        """
        Return a dict describing this provider.
        Useful for health endpoints and logging dashboards.
        """
        return {
            "provider": self.PROVIDER_NAME,
            "required_columns": sorted(self.REQUIRED_COLUMNS),
        }

    def __repr__(self) -> str:
        return f"<MarketDataProvider: {self.PROVIDER_NAME}>"