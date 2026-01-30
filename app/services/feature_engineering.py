import pandas as pd
import numpy as np
from app.monitoring.metrics import MISSING_VALUE_RATIO

class FeatureEngineer:
    """
    Create technical and sentiment-based features for ML models.
    All methods are stateless and return new DataFrames.
    """

    # ------------------------------------------------------------------
    # PRICE-BASED FEATURES
    # ------------------------------------------------------------------

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add daily percentage returns.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        df["return"] = df["close"].pct_change()
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Add rolling volatility of returns.
        """
        if "return" not in df.columns:
            raise ValueError("Call add_returns() before add_volatility()")

        df = df.copy()
        df["volatility"] = df["return"].rolling(window).std()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

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
        """
        Add MACD and MACD signal line.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        return df

    # ------------------------------------------------------------------
    # SENTIMENT MERGING
    # ------------------------------------------------------------------

    @staticmethod
    def merge_price_sentiment(
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge daily stock prices with aggregated sentiment data.

        Required columns:
        - price_df: ['date', 'close', ...]
        - sentiment_df: ['date', 'avg_sentiment', 'news_count', 'sentiment_std']
        """

        if "date" not in price_df.columns:
            raise ValueError("price_df must contain 'date' column")

        if "date" not in sentiment_df.columns:
            raise ValueError("sentiment_df must contain 'date' column")

        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()

        # Normalize date format (remove timezone, keep date only)
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

        merged = pd.merge(
            price_df,
            sentiment_df,
            on="date",
            how="left"
        )

        # Neutral defaults (industry standard)
        for col in ["avg_sentiment", "news_count", "sentiment_std"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
            else:
                merged[col] = 0.0

        return merged.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # FINAL ML DATASET CREATION
    # ------------------------------------------------------------------

    @staticmethod
    def create_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create final supervised learning dataset.

        Target:
        - target = 1 if next-day return > 0 else 0

        Features:
        - return
        - volatility
        - rsi
        - macd
        - macd_signal
        - avg_sentiment
        - lagged features
        """

        required_cols = ["return", "avg_sentiment"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.copy()

        # Target: next-day direction
        df["target"] = (df["return"].shift(-1) > 0).astype(int)

        # Lag features
        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        # Drop rows with NaNs created by rolling/lagging
        df = df.dropna().reset_index(drop=True)

        return df
    
    @staticmethod
    def monitor_data_quality(df):
        missing_ratio = df.isnull().mean().mean()
        MISSING_VALUE_RATIO.set(float(missing_ratio))
