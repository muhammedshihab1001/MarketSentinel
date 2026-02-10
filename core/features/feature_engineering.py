import pandas as pd
import numpy as np

from core.schema.feature_schema import validate_feature_schema


class FeatureEngineer:
    """
    Institutional-grade feature engineering.

    Guarantees:
    - no lookahead leakage in inference
    - deterministic ordering
    - numeric stability
    - schema enforcement
    """

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

        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

        # prevent divide-by-zero
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss

        df["rsi"] = 100 - (100 / (1 + rs))

        df["rsi"] = df["rsi"].fillna(50)  # neutral fallback

        return df

    # -------------------------------------------------------------

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:

        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        return df

    # ------------------------------------------------------------------

    @staticmethod
    def merge_price_sentiment(
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:

        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

        merged = pd.merge(
            price_df,
            sentiment_df,
            on="date",
            how="left",
            sort=False
        )

        merged = merged.sort_values("date")

        for col in ["avg_sentiment", "news_count", "sentiment_std"]:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0.0)

        return merged

    # ------------------------------------------------------------------
    # TRAINING ONLY
    # ------------------------------------------------------------------

    @staticmethod
    def create_training_dataset(df: pd.DataFrame) -> pd.DataFrame:

        df["target"] = (df["return"].shift(-1) > 0).astype(int)

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df = df.dropna()

        return df

    # ------------------------------------------------------------------
    # INFERENCE SAFE
    # ------------------------------------------------------------------

    @staticmethod
    def create_inference_dataset(df: pd.DataFrame) -> pd.DataFrame:

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df = df.dropna()

        return df

    # ------------------------------------------------------------------
    # CANONICAL PIPELINE
    # ------------------------------------------------------------------

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        training: bool = False
    ) -> pd.DataFrame:

        df = price_df.copy()

        df = df.sort_values("date")

        cls.add_returns(df)
        cls.add_volatility(df)
        cls.add_rsi(df)
        cls.add_macd(df)

        df = cls.merge_price_sentiment(df, sentiment_df)

        if training:
            df = cls.create_training_dataset(df)
        else:
            df = cls.create_inference_dataset(df)

        df = validate_feature_schema(df)

        return df.reset_index(drop=True)
