from abc import ABC, abstractmethod
import pandas as pd


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
    def validate_contract(cls, df: pd.DataFrame):

        if df is None:
            raise RuntimeError("Provider returned None.")

        if df.empty:
            raise RuntimeError("Provider returned empty dataframe.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider contract violated. Missing={missing}"
            )

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            pd.to_numeric(df[col], errors="raise")

        return df