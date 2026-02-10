import yfinance as yf
import pandas as pd
import time
import random
import os

from datetime import datetime
from typing import List


class StockPriceFetcher:
    """
    Institutional-grade market data fetcher.

    Features:
    ✅ Multi-provider fallback (Yahoo → Stooq → Cache)
    ✅ Retry + exponential backoff
    ✅ Local parquet cache
    ✅ Fast reload
    ✅ Fail-safe mode
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    MAX_RETRIES = 5
    BASE_SLEEP = 1.5

    CACHE_DIR = "data/cache"

    # -----------------------------------------------------

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:

        self._validate_dates(start_date, end_date)

        cache_file = f"{self.CACHE_DIR}/{ticker}_{interval}.parquet"

        # ==========================================
        # ⭐ FAST PATH — Load Cache If Exists
        # ==========================================

        if os.path.exists(cache_file):

            try:
                cached = pd.read_parquet(cache_file)

                # If cache already covers requested range
                if (
                    cached["date"].min() <= pd.to_datetime(start_date)
                    and cached["date"].max() >= pd.to_datetime(end_date)
                ):
                    print(f"[Fetcher] Loaded from cache: {ticker}")
                    return cached

            except Exception:
                pass  # corrupt cache fallback

        # ==========================================
        # Provider Priority
        # ==========================================

        providers = [
            self._fetch_yahoo,
            self._fetch_stooq
        ]

        last_exception = None

        for provider in providers:

            try:

                df = provider(
                    ticker,
                    start_date,
                    end_date,
                    interval
                )

                if df is not None and not df.empty:

                    df.to_parquet(cache_file, index=False)

                    print(
                        f"[Fetcher] Saved cache for {ticker}"
                    )

                    return df

            except Exception as e:
                last_exception = e
                print(f"[Fetcher] Provider failed: {provider.__name__}")

        # ==========================================
        # FAIL SAFE → LOAD OLD CACHE
        # ==========================================

        if os.path.exists(cache_file):
            print("[Fetcher] Using stale cache (provider failure)")
            return pd.read_parquet(cache_file)

        raise ValueError(
            f"All providers failed for {ticker}. "
            f"Last error: {last_exception}"
        )

    # -----------------------------------------------------
    # YAHOO
    # -----------------------------------------------------

    def _fetch_yahoo(
        self,
        ticker,
        start_date,
        end_date,
        interval
    ):

        for attempt in range(self.MAX_RETRIES):

            try:

                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if df.empty:
                    raise ValueError("Yahoo returned empty dataframe")

                df.reset_index(inplace=True)

                self._validate_columns(df)

                df = df.rename(columns=str.lower)
                df["ticker"] = ticker

                return df

            except Exception as e:

                sleep_time = (
                    self.BASE_SLEEP * (2 ** attempt)
                    + random.uniform(0, 1)
                )

                print(
                    f"[Yahoo] Retry {attempt+1}/{self.MAX_RETRIES} "
                    f"in {round(sleep_time,2)}s"
                )

                time.sleep(sleep_time)

        raise RuntimeError("Yahoo failed")

    # -----------------------------------------------------
    # STOOQ (EXTREMELY RELIABLE FREE PROVIDER)
    # -----------------------------------------------------

    def _fetch_stooq(
        self,
        ticker,
        start_date,
        end_date,
        interval
    ):

        # Stooq expects lowercase
        tk = ticker.lower()

        url = f"https://stooq.com/q/d/l/?s={tk}&i=d"

        df = pd.read_csv(url)

        if df.empty:
            raise ValueError("Stooq returned empty dataframe")

        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        df["date"] = pd.to_datetime(df["date"])

        mask = (
            (df["date"] >= start_date) &
            (df["date"] <= end_date)
        )

        df = df.loc[mask]

        df["ticker"] = ticker

        return df

    # -----------------------------------------------------

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:

        data = [
            self.fetch(t, start_date, end_date, interval)
            for t in tickers
        ]

        return pd.concat(data, ignore_index=True)

    # -----------------------------------------------------

    @staticmethod
    def _validate_dates(start_date, end_date):

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise ValueError("start_date must be before end_date")

    # -----------------------------------------------------

    def _validate_columns(self, df):

        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")
