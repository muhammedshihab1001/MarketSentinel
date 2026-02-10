import yfinance as yf
import pandas as pd
import time
import random
from datetime import datetime
from typing import List


class StockPriceFetcher:
    """
    Production-safe Yahoo Finance fetcher with:

    ✅ retry logic
    ✅ exponential backoff
    ✅ ticker fallback (.NS → .BO)
    ✅ validation
    ✅ jitter (prevents rate-limit bans)
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    MAX_RETRIES = 5
    BASE_SLEEP = 1.5   # seconds

    # -----------------------------------------------------

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:

        self._validate_dates(start_date, end_date)

        tickers_to_try = [ticker]

        # Indian fallback
        if ticker.endswith(".NS"):
            tickers_to_try.append(ticker.replace(".NS", ".BO"))

        last_exception = None

        for tk in tickers_to_try:

            for attempt in range(self.MAX_RETRIES):

                try:
                    df = yf.download(
                        tk,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=True,
                        progress=False,
                        threads=False  # ⭐ prevents Docker thread issues
                    )

                    if not df.empty:

                        df.reset_index(inplace=True)
                        self._validate_columns(df)

                        df = df.rename(columns=str.lower)
                        df["ticker"] = tk

                        return df

                except Exception as e:
                    last_exception = e

                # ⭐ Exponential backoff with jitter
                sleep_time = (
                    self.BASE_SLEEP * (2 ** attempt)
                    + random.uniform(0, 1)
                )

                print(
                    f"[Fetcher] Retry {attempt+1}/{self.MAX_RETRIES} "
                    f"for {tk} in {round(sleep_time,2)}s"
                )

                time.sleep(sleep_time)

        raise ValueError(
            f"Yahoo Finance unavailable for ticker {ticker}. "
            f"Tried {tickers_to_try}. "
            f"Last error: {last_exception}"
        )

    # -----------------------------------------------------

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:

        data = []

        for t in tickers:
            data.append(self.fetch(t, start_date, end_date, interval))

        return pd.concat(data, ignore_index=True)

    # -----------------------------------------------------

    @staticmethod
    def _validate_dates(start_date: str, end_date: str):

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise ValueError("start_date must be before end_date")

    # -----------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame):

        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")
