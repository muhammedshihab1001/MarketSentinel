from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class MarketDataProvider(ABC):
    """
    Base contract for ALL market data providers.

    Guarantees:
    - schema stability
    - provider interchangeability
    - upstream validation anchor
    """

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticker"
    }

    ########################################################

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Must return dataframe with REQUIRED_COLUMNS.

        **kwargs allows forward-compatible arguments such as:
        - min_rows
        - retries
        - provider-specific tuning

        Providers must ignore unknown kwargs safely.
        """
        raise NotImplementedError

    ########################################################
    # CONTRACT ENFORCER (institutional safety)
    ########################################################

    @classmethod
    def validate_contract(cls, df: pd.DataFrame) -> pd.DataFrame:

        if df is None:
            raise RuntimeError("Provider returned None.")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Provider did not return a DataFrame.")

        if df.empty:
            raise RuntimeError("Provider returned empty dataframe.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider contract violated. Missing={missing}"
            )

        # Enforce datetime parsing
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="raise", utc=True)

        # Enforce numeric types + finite checks
        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

            if not np.isfinite(df[col].to_numpy(dtype=float)).all():
                raise RuntimeError(f"Non-finite values detected in {col}")

        # Basic price invariant safety
        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise RuntimeError("High price invariant violated.")

        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise RuntimeError("Low price invariant violated.")

        if df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate rows detected in provider output.")

        return df