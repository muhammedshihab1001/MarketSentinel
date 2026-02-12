import pandas as pd
import numpy as np
import logging

from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)

logger = logging.getLogger("marketsentinel.features")


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 120

    SENTIMENT_COLUMNS = [
        "date",
        "avg_sentiment",
        "news_count",
        "sentiment_std"
    ]

    RETURN_CLAMP = (-0.5, 0.5)
    MERGE_TOLERANCE = pd.Timedelta("3D")
    VOL_FLOOR = 1e-4

    ####################################################
    # DATETIME NORMALIZATION
    ####################################################

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame):

        df = df.copy()

        df["date"] = (
            pd.to_datetime(df["date"], utc=True)
            .dt.tz_convert(None)
        )

        return df

    ####################################################
    # PRICE VALIDATION
    ####################################################

    @staticmethod
    def _validate_price_frame(df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe is empty.")

        required = {"date", "close"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        df = FeatureEngineer._normalize_datetime(df)
        df = df.sort_values("date")

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps in price data.")

        if not np.isfinite(df["close"]).all():
            raise RuntimeError("Non-finite close prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if len(df) < FeatureEngineer.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        return df

    ####################################################
    # FLOAT32 ENFORCEMENT
    ####################################################

    @staticmethod
    def _enforce_float32(df):

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")

        return df

    ####################################################
    # FEATURE BUILDERS
    ####################################################

    @staticmethod
    def add_returns(df):

        returns = df["close"].pct_change()
        lo, hi = FeatureEngineer.RETURN_CLAMP

        df["return"] = returns.clip(lo, hi)

    @staticmethod
    def add_volatility(df, window=5):

        df["volatility"] = (
            df["return"]
            .rolling(window, min_periods=window)
            .std()
            .clip(lower=FeatureEngineer.VOL_FLOOR, upper=5)
        )

    @staticmethod
    def add_rsi(df, window=14):

        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

        rs = avg_gain / (avg_loss + 1e-9)

        df["rsi"] = (100 - (100 / (1 + rs))).clip(0, 100)

    @staticmethod
    def add_macd(df):

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()

        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        df["macd"] = macd.clip(-50, 50)
        df["macd_signal"] = signal.clip(-50, 50)

    ####################################################
    # SENTIMENT MERGE (LOOKAHEAD SAFE)
    ####################################################

    @classmethod
    def _build_zero_sentiment(cls, price_df):

        return pd.DataFrame({
            "date": price_df["date"],
            "avg_sentiment": 0.0,
            "news_count": 0.0,
            "sentiment_std": 0.0
        })

    @classmethod
    def merge_price_sentiment(cls, price_df, sentiment_df):

        price = cls._normalize_datetime(price_df).sort_values("date")

        if sentiment_df is None or sentiment_df.empty:
            sentiment = cls._build_zero_sentiment(price)
        else:

            sentiment = sentiment_df.copy()

            missing = set(cls.SENTIMENT_COLUMNS) - set(sentiment.columns)

            if missing:
                sentiment = cls._build_zero_sentiment(price)
            else:
                sentiment = sentiment.loc[:, cls.SENTIMENT_COLUMNS]

        sentiment = cls._normalize_datetime(sentiment).sort_values("date")

        if sentiment["date"].duplicated().any():
            sentiment = sentiment.groupby(
                "date",
                as_index=False
            )[cls.SENTIMENT_COLUMNS[1:]].mean()

        # LOOKAHEAD SAFE SHIFT
        sentiment["date"] += pd.Timedelta(days=1)

        merged = pd.merge_asof(
            price,
            sentiment,
            on="date",
            direction="backward",
            tolerance=cls.MERGE_TOLERANCE,
            allow_exact_matches=False
        )

        for col in ("avg_sentiment", "news_count", "sentiment_std"):
            merged[col] = merged[col].fillna(0.0)

        return merged

    ####################################################
    # TARGET ENGINEERING
    ####################################################

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values("date").copy()

        log_close = np.log(df["close"])

        forward = log_close.shift(-1) - log_close

        risk_adj = forward / df["volatility"].clip(lower=cls.VOL_FLOOR)

        DEAD_ZONE = 0.06

        df["target"] = np.where(
            risk_adj > DEAD_ZONE, 1,
            np.where(risk_adj < -DEAD_ZONE, 0, np.nan)
        )

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df.dropna(inplace=True)

        df["target"] = df["target"].astype("int8")

        return df

    @classmethod
    def create_inference_dataset(cls, df):

        if "target" in df.columns:
            raise RuntimeError("Target detected in inference pipeline.")

        df = df.sort_values("date").copy()

        df["return_lag1"] = df["return"].shift(1)
        df["sentiment_lag1"] = df["avg_sentiment"].shift(1)

        df.dropna(inplace=True)

        return df

    ####################################################
    # MASTER PIPELINE
    ####################################################

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

        df = cls._enforce_float32(df)

        if training:
            df = cls.create_training_dataset(df)
        else:
            df = cls.create_inference_dataset(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        ###########################################
        # STRICT FEATURE BLOCK
        ###########################################

        feature_block = df.loc[:, MODEL_FEATURES]

        validated = validate_feature_schema(feature_block)

        ###########################################
        # KEEP CRITICAL MARKET COLUMNS
        ###########################################

        allowed_non_features = {"date", "close", "target"}

        non_features = [
            col for col in df.columns
            if col in allowed_non_features
        ]

        final = pd.concat(
            [
                df[non_features].reset_index(drop=True),
                validated.reset_index(drop=True)
            ],
            axis=1
        )

        logger.info(
            "Feature pipeline built | rows=%s | training=%s",
            len(final),
            training
        )

        return final.reset_index(drop=True)
