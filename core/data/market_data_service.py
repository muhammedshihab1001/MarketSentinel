import logging
from pathlib import Path
import pandas as pd
import time
import os

from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger("marketsentinel.market_data")


class MarketDataService:
    """
    Production market data layer.

    Guarantees:
    - zero future leakage
    - timezone enforcement
    - revision correction
    - atomic persistence
    - deterministic datasets
    """

    DATA_DIR = Path("data/lake")

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    MIN_HISTORY_ROWS = 120
    REVISION_DAYS = 5

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._fetcher = StockPriceFetcher()

    def _dataset_path(self, ticker: str, interval: str):
        return self.DATA_DIR / f"{ticker}_{interval}.parquet"

    @staticmethod
    def _cap_to_yesterday(date_str: str):

        requested = pd.Timestamp(date_str).normalize()
        yesterday = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)

        return min(requested, yesterday)

    def _validate_dataset(self, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Market data schema violation. Missing={missing}"
            )

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

        if df["date"].duplicated().any():
            logger.warning("Duplicate timestamps detected — deduplicating.")
            df = df.drop_duplicates("date")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if len(df) < self.MIN_HISTORY_ROWS:
            raise RuntimeError(
                "Insufficient market history for safe inference."
            )

        return df.sort_values("date").reset_index(drop=True)

    def _load_local(self, path: Path):

        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            return self._validate_dataset(df)

        except Exception:

            logger.exception(
                "Local dataset corrupted — rebuilding."
            )

            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

            return None

    def _atomic_save(self, df: pd.DataFrame, path: Path):

        df = (
            df.sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        tmp = path.with_suffix(".tmp")

        df.to_parquet(tmp, index=False)

        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())

        tmp.replace(path)

    def _fetch_with_retry(
        self,
        ticker,
        start,
        end,
        interval,
        retries=3
    ):

        last_error = None

        for attempt in range(retries):

            try:
                df = self._fetcher.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                return self._validate_dataset(df)

            except Exception as e:

                last_error = e

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d",
                    ticker,
                    attempt + 1,
                    retries
                )

                time.sleep(1.5)

        raise RuntimeError(
            f"Market fetch failed after retries: {ticker}"
        ) from last_error

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):

        end_date = self._cap_to_yesterday(end_date)

        path = self._dataset_path(ticker, interval)

        local_df = self._load_local(path)

        if local_df is None:

            logger.info(f"Building dataset for {ticker}")

            df = self._fetch_with_retry(
                ticker,
                start_date,
                end_date.strftime("%Y-%m-%d"),
                interval
            )

            self._atomic_save(df, path)

            return df

        revision_start = (
            pd.to_datetime(local_df["date"].max())
            - pd.Timedelta(days=self.REVISION_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(
            f"Fetching revision window for {ticker}: {revision_start} → {end_date.date()}"
        )

        try:

            revision_df = self._fetch_with_retry(
                ticker,
                revision_start,
                end_date.strftime("%Y-%m-%d"),
                interval
            )

            local_df = pd.concat(
                [local_df, revision_df],
                ignore_index=True
            )

        except Exception:
            logger.exception("Revision fetch failed — continuing with cached data.")

        local_df = (
            local_df
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        self._atomic_save(local_df, path)

        mask = (
            (local_df["date"] >= pd.Timestamp(start_date)) &
            (local_df["date"] <= end_date)
        )

        return local_df.loc[mask].reset_index(drop=True)
