import os
import pandas as pd

from core.data.data_fetcher import StockPriceFetcher


class MarketDataService:
    """
    Institutional Market Data Layer.

    This is now the SINGLE source of truth for price history.

    Guarantees:
    ✅ Canonical local datasets
    ✅ Incremental updates
    ✅ Offline inference capability
    ✅ Provider failure resilience
    ✅ Faster training cycles
    """

    DATA_DIR = "data/lake"

    def __init__(self):

        os.makedirs(self.DATA_DIR, exist_ok=True)

        self._fetcher = StockPriceFetcher()

    # --------------------------------------------------

    def _dataset_path(self, ticker: str, interval: str):

        return f"{self.DATA_DIR}/{ticker}_{interval}.parquet"

    # --------------------------------------------------

    def _load_local(self, path: str):

        if os.path.exists(path):

            try:
                df = pd.read_parquet(path)

                if not df.empty:
                    return df.sort_values("date")

            except Exception:
                # corrupted dataset fallback
                pass

        return None

    # --------------------------------------------------

    def _save_local(self, df: pd.DataFrame, path: str):

        df = df.sort_values("date").drop_duplicates("date")

        df.to_parquet(path, index=False)

    # --------------------------------------------------

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):
        """
        Dataset-first retrieval.

        Behavior:

        1️⃣ Load local dataset if exists
        2️⃣ Check coverage
        3️⃣ Incrementally fetch missing range
        4️⃣ Append + persist
        """

        path = self._dataset_path(ticker, interval)

        local_df = self._load_local(path)

        # --------------------------------------------------
        # CASE 1 — No dataset yet
        # --------------------------------------------------

        if local_df is None:

            df = self._fetcher.fetch(
                ticker,
                start_date,
                end_date,
                interval
            )

            self._save_local(df, path)

            return df

        # --------------------------------------------------
        # CASE 2 — Dataset exists
        # --------------------------------------------------

        local_start = pd.to_datetime(local_df["date"].min())
        local_end = pd.to_datetime(local_df["date"].max())

        request_start = pd.to_datetime(start_date)
        request_end = pd.to_datetime(end_date)

        missing_ranges = []

        # Need earlier history
        if request_start < local_start:
            missing_ranges.append(
                (start_date, local_start.strftime("%Y-%m-%d"))
            )

        # Need newer data
        if request_end > local_end:
            missing_ranges.append(
                (local_end.strftime("%Y-%m-%d"), end_date)
            )

        # --------------------------------------------------
        # Fetch ONLY missing slices
        # --------------------------------------------------

        for start, end in missing_ranges:

            try:

                new_df = self._fetcher.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                if new_df is not None and not new_df.empty:

                    local_df = pd.concat(
                        [local_df, new_df],
                        ignore_index=True
                    )

            except Exception:
                # Provider failure does NOT block inference
                pass

        # Persist merged dataset
        self._save_local(local_df, path)

        # Return requested slice only
        mask = (
            (pd.to_datetime(local_df["date"]) >= request_start) &
            (pd.to_datetime(local_df["date"]) <= request_end)
        )

        return local_df.loc[mask].reset_index(drop=True)
