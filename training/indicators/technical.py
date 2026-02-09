import pandas as pd


class TechnicalIndicators:
    """
    Modular technical indicator engine.
    Designed for extension.
    """

    @staticmethod
    def moving_average(df: pd.DataFrame, window: int = 20):
        return df["Close"].rolling(window=window).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, window: int = 14):
        delta = df["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss

        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, window: int = 20):
        ma = TechnicalIndicators.moving_average(df, window)
        std = df["Close"].rolling(window).std()

        upper = ma + 2 * std
        lower = ma - 2 * std

        return upper, lower

    @staticmethod
    def macd(df: pd.DataFrame):
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()

        return macd_line, signal
