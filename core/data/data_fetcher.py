import logging
import time
import pandas as pd
import numpy as np
import yfinance as yf
import random
import os

logger = logging.getLogger(__name__)


class StockPriceFetcher:

    MAX_RETRIES = 4
    BASE_RETRY_SLEEP = 1.0
    MAX_BACKOFF = 6.0

    MIN_ROWS = 100
    MAX_DAILY_RETURN = 0.60
    MAX_VOLUME_SPIKE = 50

    REQUEST_TIMEOUT = 20

    SOFT_FAIL_MODE = os.getenv("YFINANCE_SOFT_MODE", "1") == "1"

    ########################################################
    # DATE EXTRACTION
    ########################################################

    def _extract_date_column(self, df):

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        df.columns = [str(c).lower().strip() for c in df.columns]

        for col in ["date", "datetime", "timestamp", "index"]:
            if col in df.columns:
                df.rename(columns={col: "date"}, inplace=True)
                return df

        raise RuntimeError("Yahoo failed to produce date column.")

    ########################################################

    @staticmethod
    def _ensure_utc(series):

        s = pd.to_datetime(series, errors="coerce")

        if s.isna().all():
            raise RuntimeError("Invalid datetime series.")

        if getattr(s.dt, "tz", None) is None:
            return s.dt.tz_localize("UTC")

        return s.dt.tz_convert("UTC")

    ########################################################
    # MULTIINDEX FLATTENER (STABILIZED)
    ########################################################

    def _flatten_columns(self, df):

        if isinstance(df.columns, pd.MultiIndex):

            normalized = []

            for col in df.columns:

                parts = [str(x).lower().strip() for x in col]

                detected = None

                for field in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "volume"
                ]:
                    if field in parts:
                        detected = field.replace(" ", "_")
                        break

                normalized.append(detected or parts[0])

            df.columns = normalized

        else:
            df.columns = [
                str(c).lower().replace(" ", "_")
                for c in df.columns
            ]

        if len(set(df.columns)) != len(df.columns):
            logger.warning("Duplicate columns detected after flattening.")
            df = df.loc[:, ~df.columns.duplicated()]

        return df

    ########################################################
    # VALIDATION (SOFT-STABILIZED)
    ########################################################

    def _validate_prices(self, df):

        numeric = ["open", "high", "low", "close", "volume"]

        for col in numeric:

            if col not in df.columns:
                raise RuntimeError(f"Missing required column: {col}")

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError("All price rows invalid.")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise RuntimeError("Non-positive prices detected.")

        if (df["volume"] < 0).any():
            raise RuntimeError("Negative volume detected.")

        ####################################################
        # OHLC REPAIR
        ####################################################

        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        ####################################################
        # EXTREME RETURN GUARD (SOFT MODE SUPPORT)
        ####################################################

        returns = df["close"].pct_change().abs().fillna(0)

        if returns.max() > self.MAX_DAILY_RETURN:

            if self.SOFT_FAIL_MODE:
                logger.warning("Extreme price jump detected — smoothing.")
                df.loc[returns > self.MAX_DAILY_RETURN, "close"] = np.nan
                df["close"] = df["close"].ffill().bfill()
            else:
                raise RuntimeError("Extreme price jump detected.")

        ####################################################
        # VOLUME SPIKE GUARD
        ####################################################

        vol_ratio = df["volume"] / (
            df["volume"].rolling(20).mean() + 1e-6
        )

        if vol_ratio.max() > self.MAX_VOLUME_SPIKE:
            logger.warning("Unusual volume spike detected.")

        return df

    ########################################################
    # DOWNLOAD WITH BACKOFF (HARDENED)
    ########################################################

    def _download(self, ticker, start, end, interval):

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                # Fresh session to reduce Yahoo throttle coupling
                ticker_obj = yf.Ticker(ticker)

                df = ticker_obj.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    timeout=self.REQUEST_TIMEOUT
                )

                if df is None or df.empty:
                    raise RuntimeError("Yahoo returned empty frame.")

                if len(df.columns) <= 1:
                    raise RuntimeError("Suspicious Yahoo response structure.")

                return df

            except Exception as e:

                backoff = min(
                    (2 ** (attempt - 1)) * self.BASE_RETRY_SLEEP +
                    random.uniform(0.2, 0.8),
                    self.MAX_BACKOFF
                )

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d | backoff %.2fs | %s",
                    ticker,
                    attempt,
                    self.MAX_RETRIES,
                    backoff,
                    str(e)
                )

                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"Market fetch failed after retries: {ticker}"
                    )

                time.sleep(backoff)

    ########################################################
    # PUBLIC FETCH
    ########################################################

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        df = self._download(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._flatten_columns(df)
        df = self._extract_date_column(df)

        df["date"] = self._ensure_utc(df["date"])
        df.dropna(subset=["date"], inplace=True)

        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Yahoo schema drift detected. Missing={missing}"
            )

        df = self._validate_prices(df)

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps detected.")

        df = df.sort_values("date").reset_index(drop=True)

        now_utc = pd.Timestamp.now(tz="UTC")

        if df["date"].max() > now_utc:
            raise RuntimeError("Future candle detected.")

        if len(df) < self.MIN_ROWS:
            if self.SOFT_FAIL_MODE and len(df) >= int(self.MIN_ROWS * 0.7):
                logger.warning("Short history accepted in soft mode.")
            else:
                raise RuntimeError(
                    f"Insufficient history for {ticker}"
                )

        date_diff = df["date"].diff().dt.days

        if date_diff.max() > 10:
            logger.warning(
                "Large calendar gap detected for %s",
                ticker
            )

        df["ticker"] = ticker

        logger.info(
            "Yahoo fetch success | %s rows=%s",
            ticker,
            len(df)
        )

        return df