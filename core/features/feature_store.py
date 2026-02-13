import os
import logging
import hashlib
import re
import sys
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

    CACHE_VERSION = "v8"  # 🔥 bump cache version after bug fix
    MAX_CACHE_FILES_PER_TICKER = 6

    MIN_ROWS_REQUIRED = 100
    MIN_FILE_BYTES = 5_000

    ##################################################

    def __init__(self):

        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = get_schema_signature()[:12]
        self.engineer_hash = self._fingerprint_engineer()[:12]
        self.env_hash = self._environment_fingerprint()[:12]

    ##################################################
    # SAFE TICKER
    ##################################################

    def _sanitize_ticker(self, ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", ticker)

    ##################################################
    # ENGINEER HASH
    ##################################################

    def _fingerprint_engineer(self) -> str:

        module_path = sys.modules[
            FeatureEngineer.__module__
        ].__file__

        if not module_path or not os.path.exists(module_path):
            raise RuntimeError(
                "Unable to fingerprint FeatureEngineer module."
            )

        with open(module_path, "rb") as f:
            payload = f.read()

        return hashlib.sha256(payload).hexdigest()

    ##################################################
    # STABLE ENV HASH
    ##################################################

    def _environment_fingerprint(self) -> str:

        payload = (
            sys.version +
            pd.__version__ +
            np.__version__
        )

        return hashlib.sha256(payload.encode()).hexdigest()

    ##################################################
    # 🚨 CRITICAL FIX — DO NOT NUMERIC CAST IDENTIFIERS
    ##################################################

    def _canonicalize_df(self, df: pd.DataFrame):

        df = df.copy()

        for col in df.columns:

            # NEVER cast identifiers
            if col == "date":
                df[col] = pd.to_datetime(
                    df[col],
                    utc=True,
                    errors="raise"
                )
                continue

            if col == "ticker":
                df[col] = df[col].astype(str)
                continue

            df[col] = (
                pd.to_numeric(df[col], errors="raise")
                .astype("float64")
                .round(10)
            )

        return df.sort_values("date").reset_index(drop=True)

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
    # FSYNC
    ##################################################

    def _fsync_dir(self, path):

        if os.name == "nt":
            return

        fd = os.open(path, os.O_DIRECTORY)

        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    ##################################################
    # CACHE PRUNE
    ##################################################

    def _prune_cache(self, ticker):

        ticker = self._sanitize_ticker(ticker)

        files = sorted(
            [
                f for f in os.listdir(self.FEATURE_DIR)
                if f"_{ticker}_" in f
            ],
            reverse=True
        )

        for old in files[self.MAX_CACHE_FILES_PER_TICKER:]:

            try:
                os.remove(os.path.join(self.FEATURE_DIR, old))
            except Exception:
                logger.warning("Failed pruning cache file: %s", old)

    ##################################################
    # VALIDATION
    ##################################################

    def _validate_dataset_structure(self, df: pd.DataFrame):

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Feature file missing required columns: {missing}"
            )

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps detected.")

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

    ##################################################
    # ATOMIC WRITE
    ##################################################

    def _atomic_write(self, df: pd.DataFrame, path: str, dataset_hash: str):

        if len(df) < self.MIN_ROWS_REQUIRED:
            raise RuntimeError("Feature collapse detected.")

        tmp_path = path + ".tmp"

        df = df.sort_values("date").reset_index(drop=True)

        base_cols = ["date", "close", "ticker"]

        if "target" in df.columns:
            base_cols.insert(2, "target")

        df = df[base_cols + list(MODEL_FEATURES)]

        df["__dataset_hash"] = dataset_hash

        self._validate_dataset_structure(df)

        validate_feature_schema(df.loc[:, MODEL_FEATURES])

        if not np.isfinite(
            df.loc[:, MODEL_FEATURES].to_numpy()
        ).all():
            raise RuntimeError("Non-finite feature values.")

        df.to_parquet(tmp_path, index=False)

        if os.name != "nt":
            fd = os.open(tmp_path, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)

        os.replace(tmp_path, path)
        self._fsync_dir(self.FEATURE_DIR)

    ##################################################
    # SAFE LOAD
    ##################################################

    def _load_features(self, path: str, dataset_hash: str):

        if not os.path.exists(path):
            return None

        if os.path.getsize(path) < self.MIN_FILE_BYTES:
            logger.warning("Feature file too small — rebuilding.")
            os.remove(path)
            return None

        try:

            df = pd.read_parquet(path)

            if "__dataset_hash" not in df.columns:
                os.remove(path)
                return None

            if df["__dataset_hash"].iloc[0] != dataset_hash:
                os.remove(path)
                return None

            df.drop(columns="__dataset_hash", inplace=True)

            self._validate_dataset_structure(df)

            validate_feature_schema(
                df.loc[:, MODEL_FEATURES]
            )

            return df.sort_values("date")

        except Exception:

            logger.exception(
                "Feature corruption detected — rebuilding."
            )

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

        if price_df.empty:
            raise RuntimeError(
                "FeatureStore received empty price dataframe."
            )

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

        logger.info(
            "Feature cache miss — rebuilding."
        )

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training,
            ticker=ticker  # 🔥 VERY IMPORTANT
        )

        self._atomic_write(
            features,
            path,
            dataset_hash
        )

        self._prune_cache(ticker)

        return features
