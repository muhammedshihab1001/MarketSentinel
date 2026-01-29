import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Create technical and sentiment-based features for ML models.
    """

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["return"] = df["close"].pct_change()
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = df.copy()
        df["volatility"] = df["return"].rolling(window).std()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        df = df.copy()
        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        return df
    
    @staticmethod
    def merge_price_sentiment(
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:

        price_df = price_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

        sentiment_df = sentiment_df.copy()
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

        merged = pd.merge(
            price_df,
            sentiment_df,
            on="date",
            how="left"
        )

        # Fill missing sentiment with neutral values
        merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0)
        merged["news_count"] = merged["news_count"].fillna(0)
        merged["sentiment_std"] = merged["sentiment_std"].fillna(0)

        return merged
