from abc import ABC, abstractmethod

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
        Yahoo Finance (primary) → TwelveData (fallback)
    """

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticker",
    }

    PROVIDER_NAME: str = "unknown"

    # ─────────────────────────────────────────────────────────
    # ABSTRACT INTERFACE
    # ─────────────────────────────────────────────────────────

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for *ticker*.

        Implementations must:
        - Return REQUIRED_COLUMNS
        - Call validate_contract()
        - Ignore unknown kwargs

        Supported kwargs:
            min_rows
            retries
        """
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────
    # CONTRACT VALIDATOR
    # ─────────────────────────────────────────────────────────

    @classmethod
    def validate_contract(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final data-quality gate shared by all providers.
        """

        if df is None:
            raise RuntimeError("Provider returned None instead of DataFrame.")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(
                f"Provider did not return DataFrame (got {type(df).__name__})."
            )

        if df.empty:
            raise RuntimeError("Provider returned an empty DataFrame.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider contract violated — missing columns: {missing}"
            )

        df = df.copy()

        # ── Date parsing
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        except Exception as exc:
            raise RuntimeError(f"Date parsing failed: {exc}") from exc

        if df["date"].isna().any():
            raise RuntimeError("Date column contains invalid timestamps.")

        # ── Numeric columns
        numeric_cols = ("open", "high", "low", "close", "volume")

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            if df[col].isna().any():
                raise RuntimeError(f"Column '{col}' contains NaN values.")

            arr = df[col].to_numpy(dtype=float)

            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"Column '{col}' contains non-finite values.")

        # ── OHLC invariants
        tol = _OHLC_TOLERANCE

        violations = (
            (df["high"] + tol < df["open"]) |
            (df["high"] + tol < df["close"]) |
            (df["low"]  - tol > df["open"]) |
            (df["low"]  - tol > df["close"]) |
            (df["high"] + tol < df["low"])
        )

        if violations.any():
            raise RuntimeError(
                "OHLC price invariant violated."
            )

        # ── Ticker validation
        if df["ticker"].isna().any():
            raise RuntimeError("Ticker column contains NaN values.")

        if (df["ticker"].astype(str).str.strip() == "").any():
            raise RuntimeError("Ticker column contains empty values.")

        # ── Duplicate rows
        dupes = df.duplicated(subset=["ticker", "date"])

        if dupes.any():
            raise RuntimeError(
                f"Provider output has {dupes.sum()} duplicate rows."
            )

        # ── Sort chronologically
        df = df.sort_values("date")

        # ── Reset index
        df = df.reset_index(drop=True)

        return df

    # ─────────────────────────────────────────────────────────
    # OPTIONAL HELPERS
    # ─────────────────────────────────────────────────────────

    def provider_info(self) -> dict:
        return {
            "provider": self.PROVIDER_NAME,
            "required_columns": sorted(self.REQUIRED_COLUMNS),
        }

    def __repr__(self) -> str:
        return f"<MarketDataProvider: {self.PROVIDER_NAME}>"
