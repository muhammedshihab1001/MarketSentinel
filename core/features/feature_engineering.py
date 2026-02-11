import pandas as pd
import numpy as np
import os
import logging

from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)

logger = logging.getLogger("marketsentinel.features")


class FeatureEngineer:
    """
    Institutional Feature Pipeline.

    Guarantees:
    - zero lookahead bias
    - deterministic ordering
    - timezone normalization
    - schema enforcement
    - audit logging
    - float32 consistency
    """

    MIN_ROWS_REQUIRED = 120

    # -------------------------------------------------------------

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame):

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True
        ).dt.tz_convert(None)

        return df

    # -------------------------------------------------------------

    @staticmethod
    def _validate_price_frame(df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe is empty.")

        required = {"date", "close"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        df = FeatureEngineer._normalize_datetime(df)

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps in price data.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if not df["date"].is_monotonic_increasing:
            df.sort_values("date", inplace=True)

        return df

    # -------------------------------------------------------------

    @staticmethod
    def add_returns(df):
        df["return"] = df["close"].pct_change()

    @staticmethod
    def add_volatility(df, window=5):
        df["volatility"] = df["return"].rolling(
            window,
            min_periods=window
        ).std()

    @staticmethod
    def add_rsi(df, window=14):

        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

        rs = avg_gain / (avg_loss + 1e-9)

        df["rsi"] = 100 - (100 / (1 + rs))

    @staticmethod
    def add_macd(df):

        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # -------------------------------------------------------------
    # ZERO LOOKAHEAD MERGE
    # -------------------------------------------------------------

    @staticmethod
    def merge_price_sentiment(price_df, sentiment_df):

        price = FeatureEngineer._normalize_datetime(price_df.copy())
        sentiment = FeatureEngineer._normalize_datetime(sentiment_df.copy())

        if not sentiment["date"].is_monotonic_increasing:
            sentiment.sort_values("date", inplace=True)

        if sentiment["date"].duplicated().any():
            sentiment = sentiment.groupby("date", as_index=False).mean()

        merged = pd.merge_asof(
            price,
            sentiment,
            on="date",
            direction="backward",
            allow_exact_matches=True
        )

        for col in ("avg_sentiment", "news_count", "sentiment_std"):
            merged[col] = merged.get(col, 0.0).fillna(0.0)

        return merged

    # -------------------------------------------------------------

    @classmethod
    def _post_feature_guard(cls, df):

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError(
                "Feature dataset too small for safe usage."
            )

    # -------------------------------------------------------------

    @staticmethod
    def _sanitize_features(df):

        before = len(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df = df.dropna(subset=MODEL_FEATURES).copy()

        dropped = before - len(df)

        if dropped > 0:
            logger.warning(
                f"{dropped} rows dropped during feature sanitation."
            )

        return df

    # -------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values("date").copy()

        df["target"] = (df["return"].shift(-1) > 0).astype(int)

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df.dropna(inplace=True)

        cls._post_feature_guard(df)

        return df

    # -------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------

    @classmethod
    def create_inference_dataset(cls, df):

        if "target" in df.columns:
            raise RuntimeError(
                "Target detected in inference pipeline."
            )

        df = df.sort_values("date").copy()

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df.dropna(inplace=True)

        cls._post_feature_guard(df)

        return df

    # -------------------------------------------------------------
    # CANONICAL PIPELINE
    # -------------------------------------------------------------

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df,
        sentiment_df,
        training=False
    ):

        price_df = cls._validate_price_frame(price_df)

        df = price_df.copy()

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

        validated = validate_feature_schema(df)

        non_features = [
            col for col in df.columns
            if col not in MODEL_FEATURES
        ]

        final = pd.concat(
            [
                df[non_features].reset_index(drop=True),
                validated.reset_index(drop=True)
            ],
            axis=1
        )

        return final.reset_index(drop=True)
