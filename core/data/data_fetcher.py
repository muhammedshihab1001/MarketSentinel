import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List


class StockPriceFetcher:
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:

        self._validate_dates(start_date, end_date)

        tickers_to_try = [ticker]

        if ticker.endswith(".NS"):
            tickers_to_try.append(ticker.replace(".NS", ".BO"))

        for tk in tickers_to_try:
            try:
                stock = yf.Ticker(tk)
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True
                )

                if not df.empty:
                    df.reset_index(inplace=True)
                    self._validate_columns(df)
                    df = df.rename(columns=str.lower)
                    df["ticker"] = tk
                    return df

            except Exception:
                continue

        raise ValueError(
            f"Yahoo Finance unavailable for ticker {ticker}. Tried {tickers_to_try}"
        )

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

    @staticmethod
    def _validate_dates(start_date: str, end_date: str):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("start_date must be before end_date")

    def _validate_columns(self, df: pd.DataFrame):
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")
