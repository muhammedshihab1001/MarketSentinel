import pandas as pd


class MarketRegimeDetector:
    """
    Simple but powerful market regime classifier.

    Regimes:
    - BULL
    - BEAR
    - SIDEWAYS
    """

    def __init__(self, trend_window=200, volatility_window=50):
        self.trend_window = trend_window
        self.volatility_window = volatility_window

    def detect(self, df: pd.DataFrame):

        df = df.copy()

        # Long-term trend
        df["ma_long"] = df["close"].rolling(self.trend_window).mean()

        # Returns
        df["returns"] = df["close"].pct_change()

        # Volatility
        df["volatility"] = df["returns"].rolling(self.volatility_window).std()

        regimes = []

        for i in range(len(df)):

            if pd.isna(df["ma_long"].iloc[i]):
                regimes.append("UNKNOWN")
                continue

            price = df["close"].iloc[i]
            ma = df["ma_long"].iloc[i]
            vol = df["volatility"].iloc[i]

            if price > ma and vol < 0.02:
                regimes.append("BULL")

            elif price < ma and vol > 0.025:
                regimes.append("BEAR")

            else:
                regimes.append("SIDEWAYS")

        df["regime"] = regimes

        return df
