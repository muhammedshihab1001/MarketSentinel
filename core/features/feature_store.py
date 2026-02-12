import os
import logging
import hashlib
import re
import inspect
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

    FEATURE_DIR = "data/features"

    REQUIRED_COLUMNS = {"date", "close"}

    CACHE_VERSION = "v4"
    MAX_CACHE_FILES_PER_TICKER = 6

    ##################################################

    def __init__(self):

        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = hashlib.sha256(
            get_schema_signature().encode()
        ).hexdigest()[:12]

        self.engineer_hash = self._fingerprint_engineer()[:12]

        self.env_hash = self._environment_fingerprint()[:8]

    ##################################################
    # SAFE TICKER
    ##################################################

    def _sanitize_ticker(self, ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", ticker)

    ##################################################
    # ENGINEER FINGERPRINT — FULL SOURCE
    ##################################################

    def _fingerprint_engineer(self) -> str:

        try:
            source = inspect.getsource(FeatureEngineer)
        except Exception:
            raise RuntimeError(
                "Unable to fingerprint FeatureEngineer source."
            )

        return hashlib.sha256(source.encode()).hexdigest()

    ##################################################
    # ENVIRONMENT FINGERPRINT
    ##################################################

    def _environment_fingerprint(self) -> str:

        payload = (
            sys.version +
            pd.__version__ +
            np.__version__
        )

        return hashlib.sha256(payload.encode()).hexdigest()

    ##################################################
    # FULL DATASET HASH
    ##################################################

    def _dataset_hash(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame]
    ) -> str:

        h = hashlib.sha256()

        price_df = price_df.sort_values("date")

        price_arr = pd.util.hash_pandas_object(
            price_df,
            index=True
        ).values

        h.update(price_arr.tobytes())

        if sentiment_df is not None and not sentiment_df.empty:

            sentiment_df = sentiment_df.sort_values("date")

            sent_arr = pd.util.hash_pandas_object(
                sentiment_df,
                index=True
            ).values

            h.update(sent_arr.tobytes())

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
    # CACHE PRUNING
    ##################################################

    def _prune_cache(self, ticker):

        ticker = self._sanitize_ticker(ticker)

        files = sorted(
            [
                f for f in os.listdir(self.FEATURE_DIR)
                if f"_{ticker}_" in f
            ],
            key=lambda x: os.path.getmtime(
                os.path.join(self.FEATURE_DIR, x)
            ),
            reverse=True
        )

        for old in files[self.MAX_CACHE_FILES_PER_TICKER:]:

            try:
                os.remove(os.path.join(self.FEATURE_DIR, old))
            except Exception:
                logger.warning("Failed pruning cache file: %s", old)

    ##################################################
    # FSYNC
    ##################################################

    def _fsync_file(self, path):

        if os.name == "nt":
            return

        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

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

    def _atomic_write(self, df: pd.DataFrame, path: str):

        tmp_path = path + ".tmp"

        df = df.sort_values("date").reset_index(drop=True)

        self._validate_dataset_structure(df)

        feature_block = df.loc[:, MODEL_FEATURES]

        if list(feature_block.columns) != list(MODEL_FEATURES):
            raise RuntimeError("Feature ordering violation.")

        validate_feature_schema(feature_block)

        arr = feature_block.to_numpy(dtype=float)

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature values.")

        if np.isnan(arr[0]).any():
            raise RuntimeError(
                "Feature warmup NaNs detected — pipeline unsafe."
            )

        df.to_parquet(tmp_path, index=False)

        self._fsync_file(tmp_path)

        os.replace(tmp_path, path)

        self._fsync_file(path)

    ##################################################
    # SAFE LOAD
    ##################################################

    def _load_features(self, path: str) -> Optional[pd.DataFrame]:

        if not os.path.exists(path):
            return None

        try:

            df = pd.read_parquet(path)

            if df.empty:
                raise RuntimeError("Feature file empty.")

            self._validate_dataset_structure(df)

            validate_feature_schema(
                df.loc[:, MODEL_FEATURES]
            )

            if not np.isfinite(
                df.loc[:, MODEL_FEATURES].to_numpy()
            ).all():
                raise RuntimeError("Corrupted feature file.")

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
    # PUBLIC API
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

        stored = self._load_features(path)

        if stored is not None:
            return stored

        logger.info(
            "Feature cache miss or lineage change — rebuilding."
        )

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training
        )

        self._atomic_write(features, path)

        self._prune_cache(ticker)

        return features
