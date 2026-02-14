import logging
import pandas as pd
import numpy as np

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger("marketsentinel.provider.yahoo")


class YahooProvider(MarketDataProvider):

    PROVIDER_NAME = "yahoo"

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

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    ########################################################
    # NORMALIZATION (INSTITUTIONAL)
    ########################################################

    def _normalize(self, df: pd.DataFrame, ticker: str):

        if df is None or df.empty:
            raise RuntimeError("Yahoo returned empty dataset.")

        df = df.copy()

        ####################################################
        # RESET INDEX (Yahoo often uses date index)
        ####################################################

        if "date" not in df.columns:
            df = df.reset_index()

        ####################################################
        # STANDARDIZE COLUMN NAMES
        ####################################################

        df.columns = [c.lower().strip() for c in df.columns]

        rename_map = {
            "adj close": "close"
        }

        df.rename(columns=rename_map, inplace=True)

        ####################################################
        # REQUIRED CHECK
        ####################################################

        required_price_cols = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        missing = required_price_cols - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Yahoo schema violation. Missing={missing}"
            )

        ####################################################
        # TICKER INJECTION (CRITICAL FIX)
        ####################################################

        df["ticker"] = ticker

        ####################################################
        # NUMERIC HARDENING
        ####################################################

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=numeric_cols)

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices from Yahoo.")

        ####################################################
        # DATE NORMALIZATION
        ####################################################

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        ).dt.tz_convert(None)

        df = df.dropna(subset=["date"])

        ####################################################
        # SORT + DEDUP
        ####################################################

        df = df.sort_values("date")

        df = df.drop_duplicates(
            subset=["ticker", "date"]
        )

        return df.reset_index(drop=True)

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval):

        df = self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._normalize(df, ticker)

        logger.debug(
            "Yahoo served market data | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
