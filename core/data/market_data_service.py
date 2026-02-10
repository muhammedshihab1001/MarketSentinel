import os
import logging
import pandas as pd

from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger("marketsentinel.market_data")


class MarketDataService:
    """
    Institutional Market Data Layer.

    Guarantees:
    - schema validation
    - numeric safety
    - duplicate protection
    - incremental correctness
    - corruption recovery
    """

    DATA_DIR = "data/lake"

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    MIN_HISTORY_ROWS = 120

    # --------------------------------------------------

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self._fetcher = StockPriceFetcher()

    # --------------------------------------------------

    def _dataset_path(self, ticker: str, interval: str):
        return f"{self.DATA_DIR}/{ticker}_{interval}.parquet"

    # --------------------------------------------------

    def _validate_dataset(self, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Market data schema violation. Missing={missing}"
            )

        if df["close"].isna().any():
            raise RuntimeError("Close price contains NaNs.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if df["date"].duplicated().any():
            logger.warning("Duplicate timestamps detected — deduplicating.")
            df = df.drop_duplicates("date")

        if len(df) < self.MIN_HISTORY_ROWS:
            raise RuntimeError(
                "Insufficient market history for safe inference."
            )

        return df.sort_values("date")

    # --------------------------------------------------

    def _load_local(self, path: str):

        if not os.path.exists(path):
            return None

        try:

            df = pd.read_parquet(path)

            return self._validate_dataset(df)

        except Exception:

            logger.exception(
                "Local dataset corrupted — rebuilding."
            )

            try:
                os.remove(path)
            except Exception:
                pass

            return None

    # --------------------------------------------------

    def _save_local(self, df: pd.DataFrame, path: str):

        df = (
            df
            .sort_values("date")
            .drop_duplicates("date")
        )

        tmp = path + ".tmp"

        df.to_parquet(tmp, index=False)

        os.replace(tmp, path)

    # --------------------------------------------------

    def _fetch_safe(
        self,
        ticker,
        start,
        end,
        interval
    ):

        df = self._fetcher.fetch(
            ticker,
            start,
            end,
            interval
        )

        return self._validate_dataset(df)

    # --------------------------------------------------

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):

        path = self._dataset_path(ticker, interval)

        local_df = self._load_local(path)

        # -------------------------------------------
        # NO DATASET
        # -------------------------------------------

        if local_df is None:

            logger.info(
                f"Building dataset for {ticker}"
            )

            df = self._fetch_safe(
                ticker,
                start_date,
                end_date,
                interval
            )

            self._save_local(df, path)

            return df

        # -------------------------------------------
        # INCREMENTAL FETCH
        # -------------------------------------------

        local_start = pd.to_datetime(local_df["date"].min())
        local_end = pd.to_datetime(local_df["date"].max())

        request_start = pd.to_datetime(start_date)
        request_end = pd.to_datetime(end_date)

        missing_ranges = []

        if request_start < local_start:
            missing_ranges.append(
                (start_date, (local_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            )

        if request_end > local_end:
            missing_ranges.append(
                ((local_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end_date)
            )

        for start, end in missing_ranges:

            try:

                logger.info(
                    f"Fetching missing slice {ticker}: {start} → {end}"
                )

                new_df = self._fetch_safe(
                    ticker,
                    start,
                    end,
                    interval
                )

                local_df = pd.concat(
                    [local_df, new_df],
                    ignore_index=True
                )

            except Exception:

                logger.exception(
                    "Provider failure — serving best available dataset."
                )

        self._save_local(local_df, path)

        mask = (
            (pd.to_datetime(local_df["date"]) >= request_start) &
            (pd.to_datetime(local_df["date"]) <= request_end)
        )

        return local_df.loc[mask].reset_index(drop=True)
