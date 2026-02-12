import os
import logging
import hashlib
import re
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
    """
    Institutional Feature Store.

    Guarantees:
    - schema-aware cache
    - dataset lineage binding
    - feature-code invalidation
    - atomic persistence
    - corruption recovery
    - deterministic ordering
    """

    FEATURE_DIR = "data/features"
    REQUIRED_DATASET_COLUMNS = {"date"}

    def __init__(self):
        os.makedirs(self.FEATURE_DIR, exist_ok=True)
        self.engineer = FeatureEngineer()

        self.schema_hash = hashlib.sha256(
            get_schema_signature().encode()
        ).hexdigest()[:10]

        self.engineer_hash = self._fingerprint_engineer()[:10]

    ##################################################

    def _sanitize_ticker(self, ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", ticker)

    ##################################################

    def _fingerprint_engineer(self) -> str:

        path = FeatureEngineer.__module__.replace(".", "/") + ".py"

        if not os.path.exists(path):
            return "unknown_engineer"

        h = hashlib.sha256()

        with open(path, "rb") as f:
            h.update(f.read())

        return h.hexdigest()

    ##################################################

    def _dataset_hash(self, df: pd.DataFrame) -> str:

        ordered = df.sort_values("date").reset_index(drop=True)

        arr = pd.util.hash_pandas_object(
            ordered,
            index=False
        ).values

        h = hashlib.sha256()
        h.update(arr.tobytes())

        return h.hexdigest()[:12]

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
            f"{ticker}_{suffix}_"
            f"{dataset_hash}_"
            f"{self.schema_hash}_"
            f"{self.engineer_hash}.parquet"
        )

    ##################################################

    def _validate_dataset_structure(self, df: pd.DataFrame):

        missing = self.REQUIRED_DATASET_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Dataset missing required columns: {missing}"
            )

        if df["date"].duplicated().any():
            raise RuntimeError(
                "Duplicate timestamps detected."
            )

    ##################################################

    def _fsync_dir(self, directory):

        if os.name == "nt":
            return

        fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    ##################################################

    def _atomic_write(self, df: pd.DataFrame, path: str):

        tmp_path = path + ".tmp"

        df = df.sort_values("date").reset_index(drop=True)

        self._validate_dataset_structure(df)

        feature_block = df.loc[:, MODEL_FEATURES]

        if list(feature_block.columns) != list(MODEL_FEATURES):
            raise RuntimeError(
                "Feature ordering violation detected."
            )

        validate_feature_schema(feature_block)

        if not np.isfinite(feature_block.to_numpy()).all():
            raise RuntimeError(
                "Non-finite values detected in feature block."
            )

        df.to_parquet(tmp_path, index=False)

        os.replace(tmp_path, path)

        self._fsync_dir(os.path.dirname(path))

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
                raise RuntimeError("Corrupted feature file detected.")

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

        dataset_hash = self._dataset_hash(price_df)

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

        return features
