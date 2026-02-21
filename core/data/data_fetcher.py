import logging
import time
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class StockPriceFetcher:

    MAX_RETRIES = 3
    RETRY_SLEEP = 2
    MIN_ROWS = 100

    ########################################################
    # BULLETPROOF DATE EXTRACTION
    ########################################################

    def _extract_date_column(self, df):

        # CASE 1 — DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # Normalize column names early
        df.columns = [str(c).lower().strip() for c in df.columns]

        possible = ["date", "datetime", "timestamp", "index"]

        for col in possible:
            if col in df.columns:
                df.rename(columns={col: "date"}, inplace=True)
                return df

        raise RuntimeError("Yahoo failed to produce date column.")

    ########################################################

    @staticmethod
    def _ensure_utc(series):

        s = pd.to_datetime(series, errors="coerce")

        if getattr(s.dt, "tz", None) is None:
            return s.dt.tz_localize("UTC")

        return s.dt.tz_convert("UTC")

    ########################################################

    def _flatten_columns(self, df):

        # Handles MultiIndex from Yahoo
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df.columns = [
            str(c).lower().replace(" ", "_")
            for c in df.columns
        ]

        return df

    ########################################################

    def _validate_prices(self, df):

        numeric = ["open", "high", "low", "close", "volume"]

        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError("All price rows invalid.")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise RuntimeError("Non-positive prices detected.")

        if (df["volume"] < 0).any():
            raise RuntimeError("Negative volume detected.")

        return df

    ########################################################

    def _download(self, ticker, start, end, interval):

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )

                if df is None or df.empty:
                    raise RuntimeError("Yahoo returned empty frame.")

                return df

            except Exception as e:

                logger.warning(
                    "Fetch failed (%s) attempt %s/%s | %s",
                    ticker,
                    attempt,
                    self.MAX_RETRIES,
                    str(e)
                )

                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"Market fetch failed after retries: {ticker}"
                    )

                time.sleep(self.RETRY_SLEEP)

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        df = self._download(
            ticker,
            start_date,
            end_date,
            interval
        )

        ####################################################
        # 🔥 FIX 1 — FLATTEN FIRST
        ####################################################

        df = self._flatten_columns(df)

        ####################################################
        # 🔥 FIX 2 — DATE EXTRACTION
        ####################################################

        df = self._extract_date_column(df)

        ####################################################
        # 🔥 FIX 3 — UTC SAFE
        ####################################################

        df["date"] = self._ensure_utc(df["date"])

        df.dropna(subset=["date"], inplace=True)

        ####################################################

        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = {"open", "high", "low", "close", "volume"}

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Yahoo schema drift detected. Missing={missing}"
            )

        df = self._validate_prices(df)

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        ####################################################
        # FUTURE LEAKAGE GUARD
        ####################################################

        now_utc = pd.Timestamp.now(tz="UTC")

        if df["date"].max() > now_utc:
            raise RuntimeError("Future candle detected.")

        if len(df) < self.MIN_ROWS:
            raise RuntimeError(
                f"Insufficient history for {ticker}"
            )

        df["ticker"] = ticker

        logger.info(
            "Yahoo fetch success | %s rows=%s",
            ticker,
            len(df)
        )

        return df
