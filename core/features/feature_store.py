import os
import logging
import hashlib
import re
import sys
import uuid
from typing import Optional

import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES,
    get_schema_signature
)

logger = logging.getLogger("marketsentinel.feature_store")


class FeatureStore:

    FEATURE_DIR = os.path.abspath("data/features")

    REQUIRED_COLUMNS = {"date", "close", "ticker"}

    CACHE_VERSION = "v11"   # 🔥 bumped after hardening
    MAX_CACHE_FILES_PER_TICKER = 6

    MIN_ROWS_REQUIRED = 100
    MIN_FILE_BYTES = 5_000

    ABS_FEATURE_LIMIT = 1e5
    MIN_ROW_STABILITY_RATIO = 0.65

    ##################################################

    def __init__(self):

        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = get_schema_signature()[:12]
        self.engineer_hash = self._fingerprint_engineer()[:12]
        self.env_hash = self._environment_fingerprint()[:12]

    ##################################################

    def _sanitize_ticker(self, ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", ticker)

    ##################################################

    def _fingerprint_engineer(self) -> str:

        module_path = sys.modules[
            FeatureEngineer.__module__
        ].__file__

        with open(module_path, "rb") as f:
            payload = f.read()

        return hashlib.sha256(payload).hexdigest()

    ##################################################

    def _environment_fingerprint(self) -> str:

        payload = (
            sys.version +
            pd.__version__ +
            np.__version__
        )

        return hashlib.sha256(payload.encode()).hexdigest()

    ##################################################
    # CANONICALIZE
    ##################################################

    def _canonicalize_df(self, df: pd.DataFrame):

        df = df.copy()

        if df.empty:
            raise RuntimeError("Cannot hash empty dataframe.")

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["ticker"] = df["ticker"].astype(str)
        df["close"] = pd.to_numeric(df["close"]).astype("float64")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df[numeric_cols] = df[numeric_cols].astype("float64")

        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ##################################################

    def _dataset_hash(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame]
    ) -> str:

        h = hashlib.sha256()

        price_df = self._canonicalize_df(price_df)

        h.update(
            pd.util.hash_pandas_object(
                price_df,
                index=True
            ).values.tobytes()
        )

        if sentiment_df is not None and not sentiment_df.empty:

            sentiment_df = self._canonicalize_df(sentiment_df)

            sentiment_df = sentiment_df[
                sentiment_df["date"].isin(price_df["date"])
            ]

            h.update(
                pd.util.hash_pandas_object(
                    sentiment_df,
                    index=True
                ).values.tobytes()
            )

        return h.hexdigest()[:20]

    ##################################################

    def _feature_path(
        self,
        ticker: str,
        dataset_hash: str,
        training: bool
    ):

        suffix = "train" if training else "infer"
        ticker = self._sanitize_ticker(ticker)

        return (
            f"{self.FEATURE_DIR}/"
            f"{self.CACHE_VERSION}_"
            f"{ticker}_{suffix}_"
            f"{dataset_hash}_"
            f"{self.schema_hash}_"
            f"{self.engineer_hash}_"
            f"{self.env_hash}.parquet"
        )

    ##################################################
    # STRUCTURE VALIDATION
    ##################################################

    def _validate_dataset_structure(self, df: pd.DataFrame):

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise RuntimeError("date must be datetime.")

        if not pd.api.types.is_float_dtype(df["close"]):
            raise RuntimeError("close must be float.")

        # 🔥 FIXED — multi ticker duplicate guard
        if df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate ticker/date detected.")

        if not df.sort_values(["ticker", "date"]) \
                .reset_index(drop=True) \
                .equals(df.reset_index(drop=True)):
            raise RuntimeError("Dataset not strictly ordered.")

    ##################################################
    # ATOMIC WRITE
    ##################################################

    def _atomic_write(
        self,
        df: pd.DataFrame,
        path: str,
        dataset_hash: str,
        training: bool
    ):

        if len(df) < self.MIN_ROWS_REQUIRED:
            raise RuntimeError("Feature collapse detected.")

        base_cols = ["date", "close", "ticker"]

        if training:
            if "target" not in df.columns:
                raise RuntimeError("Training dataset missing target.")

            base_cols.insert(2, "target")

        df = df[base_cols + list(MODEL_FEATURES)]

        self._validate_dataset_structure(df)
        validate_feature_schema(df.loc[:, MODEL_FEATURES])

        arr = df.loc[:, MODEL_FEATURES].to_numpy()

        if arr.size > 0 and np.abs(arr).max() > self.ABS_FEATURE_LIMIT:
            raise RuntimeError("Feature explosion detected.")

        # 🔥 metadata AFTER validation
        df["__dataset_hash"] = dataset_hash
        df["__schema_hash"] = self.schema_hash
        df["__engineer_hash"] = self.engineer_hash

        tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"

        df.to_parquet(
            tmp_path,
            index=False,
            engine="pyarrow",
            compression="zstd"
        )

        os.replace(tmp_path, path)

    ##################################################
    # SAFE LOAD
    ##################################################

    def _load_features(self, path: str, dataset_hash: str):

        if not os.path.exists(path):
            return None

        if os.path.getsize(path) < self.MIN_FILE_BYTES:
            os.remove(path)
            return None

        try:

            df = pd.read_parquet(path)

            meta = [
                "__dataset_hash",
                "__schema_hash",
                "__engineer_hash"
            ]

            if not all(c in df.columns for c in meta):
                os.remove(path)
                return None

            if df["__dataset_hash"].iloc[0] != dataset_hash:
                os.remove(path)
                return None

            df.drop(columns=meta, inplace=True)

            self._validate_dataset_structure(df)
            validate_feature_schema(df.loc[:, MODEL_FEATURES])

            if len(df) < self.MIN_ROWS_REQUIRED * self.MIN_ROW_STABILITY_RATIO:
                os.remove(path)
                return None

            return df

        except Exception:

            logger.exception("Feature corruption detected — rebuilding.")

            try:
                os.remove(path)
            except Exception:
                pass

            return None

    ##################################################
    # PUBLIC
    ##################################################

    def get_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str = "unknown",
        training: bool = False
    ):

        dataset_hash = self._dataset_hash(
            price_df,
            sentiment_df
        )

        path = self._feature_path(
            ticker,
            dataset_hash,
            training
        )

        stored = self._load_features(path, dataset_hash)

        if stored is not None:
            return stored

        logger.info("Feature cache miss — rebuilding.")

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training,
            ticker=ticker
        )

        self._atomic_write(
            features,
            path,
            dataset_hash,
            training
        )

        return features
