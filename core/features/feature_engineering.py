import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Institutional-grade feature engineering.

    Improvements:
    ✅ Reduced DataFrame copying
    ✅ Faster rolling ops
    ✅ Stable merges
    ✅ Lower memory churn
    ✅ Inference-friendly pipeline
    """

    # ------------------------------------------------------------------
    # PRICE-BASED FEATURES
    # ------------------------------------------------------------------

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:

        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df["return"] = df["close"].pct_change()

        return df

    # -------------------------------------------------------------

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:

        if "return" not in df.columns:
            raise ValueError("Call add_returns() before add_volatility()")

        df["volatility"] = (
            df["return"]
            .rolling(window, min_periods=window)
            .std()
        )

        return df

    # -------------------------------------------------------------

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:

        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

        rs = avg_gain / avg_loss

        df["rsi"] = 100 - (100 / (1 + rs))

        return df

    # -------------------------------------------------------------

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:

        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

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

        if "date" not in price_df.columns:
            raise ValueError("price_df must contain 'date' column")

        if "date" not in sentiment_df.columns:
            raise ValueError("sentiment_df must contain 'date' column")

        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

        merged = pd.merge(
            price_df,
            sentiment_df,
            on="date",
            how="left",
            sort=False
        )

        for col in ["avg_sentiment", "news_count", "sentiment_std"]:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col].fillna(0.0, inplace=True)

        return merged

    # ------------------------------------------------------------------
    # FINAL ML DATASET
    # ------------------------------------------------------------------

    @staticmethod
    def create_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:

        required_cols = ["return", "avg_sentiment"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["target"] = (df["return"].shift(-1) > 0).astype(int)

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df.dropna(inplace=True)

        return df

    # ------------------------------------------------------------------
    # CANONICAL PIPELINE
    # ------------------------------------------------------------------

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Canonical pipeline.

        Optimized for inference workloads.
        """

        # SINGLE COPY ONLY
        df = price_df.copy()

        cls.add_returns(df)
        cls.add_volatility(df)
        cls.add_rsi(df)
        cls.add_macd(df)

        df = cls.merge_price_sentiment(df, sentiment_df)

        df = cls.create_ml_dataset(df)

        return df.reset_index(drop=True)
