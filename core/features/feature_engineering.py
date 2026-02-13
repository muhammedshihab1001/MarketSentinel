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

    MERGE_TOLERANCE = pd.Timedelta("2D")

    VOL_FLOOR = 1e-4
    SENTIMENT_STD_FLOOR = 0.02

    ###################################################
    # DATETIME
    ###################################################

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame):

        df = df.copy()

        dt = pd.to_datetime(
            df["date"],
            utc=True,
            errors="raise"
        )

        df["date"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)

        return df

    ###################################################
    # PRICE VALIDATION
    ###################################################

    @staticmethod
    def _validate_price_frame(df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe is empty.")

        required = {"date", "close"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        df = FeatureEngineer._normalize_datetime(df)
        df = df.sort_values("date")

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Price dates must be strictly increasing.")

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps in price data.")

        df["close"] = pd.to_numeric(df["close"], errors="raise")

        if not np.isfinite(df["close"]).all():
            raise RuntimeError("Non-finite close prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if len(df) < FeatureEngineer.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        return df

    ###################################################
    # NUMERIC ENFORCEMENT
    ###################################################

    @staticmethod
    def _force_numeric(df):

        for col in df.columns:
            if col == "date":
                continue

            df[col] = pd.to_numeric(df[col], errors="raise")

        return df.astype(
            {c: "float32" for c in df.select_dtypes(include="number").columns}
        )

    ###################################################
    # FEATURES
    ###################################################

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

    ###################################################
    # ALIGNMENT
    ###################################################

    @staticmethod
    def align_features_to_prediction_time(df):

        leak_prone = list(MODEL_FEATURES)

        for col in leak_prone:
            if col in df.columns:
                df[col] = df[col].shift(1)

        return df

    ###################################################
    # SENTIMENT MERGE
    ###################################################

    @classmethod
    def merge_price_sentiment(cls, price_df, sentiment_df):

        if sentiment_df is None or sentiment_df.empty:
            raise RuntimeError(
                "Sentiment dataset empty — refusing to fabricate signal."
            )

        price = cls._normalize_datetime(price_df).sort_values("date")

        sentiment = sentiment_df.copy()

        missing = set(cls.SENTIMENT_COLUMNS) - set(sentiment.columns)

        if missing:
            raise RuntimeError(
                f"Sentiment schema violation: {missing}"
            )

        sentiment = sentiment.loc[:, cls.SENTIMENT_COLUMNS]
        sentiment = cls._normalize_datetime(sentiment).sort_values("date")

        if not sentiment["date"].is_monotonic_increasing:
            raise RuntimeError("Sentiment dates must be monotonic.")

        sentiment = cls._force_numeric(sentiment)

        if sentiment["date"].duplicated().any():
            sentiment = sentiment.groupby(
                "date",
                as_index=False
            )[cls.SENTIMENT_COLUMNS[1:]].mean()

        sentiment["date"] += pd.Timedelta(days=1)

        merged = pd.merge_asof(
            price,
            sentiment,
            on="date",
            direction="backward",
            tolerance=cls.MERGE_TOLERANCE,
            allow_exact_matches=False
        )

        merged["avg_sentiment"] = merged["avg_sentiment"].ffill(limit=3)
        merged["news_count"] = merged["news_count"].ffill(limit=3)
        merged["sentiment_std"] = (
            merged["sentiment_std"]
            .fillna(cls.SENTIMENT_STD_FLOOR)
            .clip(lower=cls.SENTIMENT_STD_FLOOR)
        )

        merged.dropna(
            subset=["avg_sentiment", "news_count"],
            inplace=True
        )

        if merged["avg_sentiment"].std() < 1e-4:
            raise RuntimeError(
                "Sentiment variance collapsed — check news pipeline."
            )

        return merged

    ###################################################
    # TARGET
    ###################################################

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values("date").copy()

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Datetime ordering violated.")

        log_close = np.log(df["close"])

        forward = log_close.shift(-1) - log_close

        risk_adj = forward / df["volatility"].clip(lower=cls.VOL_FLOOR)

        DEAD_ZONE = 0.06

        df["target"] = np.where(
            risk_adj > DEAD_ZONE, 1,
            np.where(risk_adj < -DEAD_ZONE, 0, np.nan)
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < 100:
            raise RuntimeError(
                "Feature collapse — too few rows after target creation."
            )

        df["target"] = df["target"].astype("int8")

        return df

    ###################################################
    # MASTER PIPELINE
    ###################################################

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

        df = cls.align_features_to_prediction_time(df)

        df = cls._force_numeric(df)

        if training:
            df = cls.create_training_dataset(df)
        else:
            raise RuntimeError(
                "Inference pipeline should be built separately."
            )

        feature_block = df.loc[:, MODEL_FEATURES]

        validated = validate_feature_schema(feature_block)

        allowed_non_features = {"date", "close", "target", "ticker"}

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
