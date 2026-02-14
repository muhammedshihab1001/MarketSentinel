import logging
import time
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class StockPriceFetcher:
    """
    Institutional-grade Yahoo fetcher.

    Fixes:
    ✔ tz_localize crash
    ✔ mixed timezone columns
    ✔ naive timestamps
    ✔ silent NaT
    ✔ retry instability
    ✔ yahoo schema drift
    """

    MAX_RETRIES = 3
    RETRY_SLEEP = 2

    MIN_ROWS = 100

    ########################################################
    # SAFE UTC NORMALIZER
    ########################################################

    @staticmethod
    def _ensure_utc(series: pd.Series) -> pd.Series:
        """
        Converts ANY datetime garbage into clean UTC.

        Handles:
        ✔ tz-aware
        ✔ naive
        ✔ mixed
        ✔ object dtype
        """

        s = pd.to_datetime(series, errors="coerce")

        # If timezone naive → localize
        if getattr(s.dt, "tz", None) is None:
            return s.dt.tz_localize("UTC")

        # Already tz-aware → convert
        return s.dt.tz_convert("UTC")

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

    def _normalize_columns(self, df):

        df.columns = [
            str(c).lower().replace(" ", "_")
            for c in df.columns
        ]

        # adjusted close normalization
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = {"open", "high", "low", "close", "volume"}

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Yahoo schema drift detected. Missing={missing}"
            )

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

        # invariant checks
        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise RuntimeError("High invariant violated.")

        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise RuntimeError("Low invariant violated.")

        return df

    ########################################################

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval="1d"
    ) -> pd.DataFrame:

        df = self._download(
            ticker,
            start_date,
            end_date,
            interval
        )

        ####################################################
        # RESET INDEX SAFELY
        ####################################################

        if not isinstance(df.index, pd.DatetimeIndex):
            raise RuntimeError("Yahoo index is not datetime.")

        df = df.reset_index()

        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "date"}, inplace=True)

        ####################################################
        # 🔥 CRITICAL FIX — NO tz_localize
        ####################################################

        df["date"] = self._ensure_utc(df["date"])

        df.dropna(subset=["date"], inplace=True)

        ####################################################

        df = self._normalize_columns(df)

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

        logger.debug(
            "Yahoo fetch success | %s rows=%s",
            ticker,
            len(df)
        )

        return df
