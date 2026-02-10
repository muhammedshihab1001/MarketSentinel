import pandas as pd
import numpy as np

from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)


class FeatureEngineer:
    """
    Institutional Feature Pipeline.

    Guarantees:
    - zero training/inference skew
    - no target leakage
    - deterministic ordering
    - numeric stability
    - schema compliance
    """

    MIN_ROWS_REQUIRED = 120

    # -------------------------------------------------------------

    @staticmethod
    def _validate_price_frame(df: pd.DataFrame):

        if df.empty:
            raise RuntimeError("Price dataframe is empty.")

        required = {"date", "close"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        if df["close"].isna().any():
            raise RuntimeError("Close price contains NaNs.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps in price data.")

    # -------------------------------------------------------------

    @staticmethod
    def add_returns(df: pd.DataFrame):
        df["return"] = df["close"].pct_change()

    # -------------------------------------------------------------

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 5):

        df["volatility"] = (
            df["return"]
            .rolling(window, min_periods=window)
            .std()
        )

    # -------------------------------------------------------------

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14):

        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

        rs = avg_gain / (avg_loss + 1e-9)

        rsi = 100 - (100 / (1 + rs))

        df["rsi"] = rsi.clip(0, 100).fillna(50)

    # -------------------------------------------------------------

    @staticmethod
    def add_macd(df: pd.DataFrame):

        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # -------------------------------------------------------------

    @staticmethod
    def merge_price_sentiment(
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:

        price = price_df.copy()
        sentiment = sentiment_df.copy()

        price["date"] = pd.to_datetime(price["date"]).dt.date
        sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date

        if sentiment["date"].duplicated().any():
            sentiment = sentiment.groupby("date").mean().reset_index()

        merged = pd.merge(
            price,
            sentiment,
            on="date",
            how="left",
            validate="one_to_one"
        ).sort_values("date")

        for col in ("avg_sentiment", "news_count", "sentiment_std"):

            if col not in merged:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0.0)

        return merged

    # -------------------------------------------------------------

    @staticmethod
    def _post_feature_guard(df: pd.DataFrame):

        if len(df) < FeatureEngineer.MIN_ROWS_REQUIRED:
            raise RuntimeError(
                "Feature dataset too small for safe inference/training."
            )

    # -------------------------------------------------------------

    @staticmethod
    def _sanitize_features(df: pd.DataFrame):

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().copy()

        return df

    # -------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------

    @staticmethod
    def create_training_dataset(df: pd.DataFrame):

        df = df.copy()

        df["target"] = (df["return"].shift(-1) > 0).astype(int)

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df = df.dropna()

        FeatureEngineer._post_feature_guard(df)

        return df

    # -------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------

    @staticmethod
    def create_inference_dataset(df: pd.DataFrame):

        df = df.copy()

        if "target" in df.columns:
            raise RuntimeError(
                "Target column detected in inference pipeline."
            )

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df = df.dropna()

        FeatureEngineer._post_feature_guard(df)

        return df

    # -------------------------------------------------------------
    # CANONICAL PIPELINE
    # -------------------------------------------------------------

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        training: bool = False
    ) -> pd.DataFrame:

        cls._validate_price_frame(price_df)

        df = price_df.copy()

        if not df["date"].is_monotonic_increasing:
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

        df = cls._sanitize_features(df)

        # Validate ONLY model features
        validated_features = validate_feature_schema(
            df[list(MODEL_FEATURES)]
        )

        # Re-attach target safely if training
        if training:
            validated_features["target"] = df["target"].values

        return validated_features.reset_index(drop=True)
