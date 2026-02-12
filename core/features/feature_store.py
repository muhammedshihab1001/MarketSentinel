import os
import logging
import hashlib
from typing import Optional

import pandas as pd

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
    - automatic invalidation
    - atomic persistence
    - corruption recovery
    - deterministic ordering
    - inference/training parity
    """

    FEATURE_DIR = "data/features"
    REQUIRED_DATASET_COLUMNS = {"date"}

    def __init__(self):
        os.makedirs(self.FEATURE_DIR, exist_ok=True)
        self.engineer = FeatureEngineer()

        # CRITICAL — binds cache to schema
        self.schema_hash = hashlib.sha256(
            get_schema_signature().encode()
        ).hexdigest()[:10]

    # --------------------------------------------------

    def _feature_path(self, ticker: str, training: bool):

        suffix = "train" if training else "infer"

        return (
            f"{self.FEATURE_DIR}/"
            f"{ticker}_{suffix}_{self.schema_hash}.parquet"
        )

    # --------------------------------------------------

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

    # --------------------------------------------------

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

        df.to_parquet(tmp_path, index=False)

        os.replace(tmp_path, path)

    # --------------------------------------------------

    def _load_features(self, path: str) -> Optional[pd.DataFrame]:

        if not os.path.exists(path):
            return None

        try:

            df = pd.read_parquet(path)

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

    # --------------------------------------------------

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

        path = self._feature_path(ticker, training)

        stored = self._load_features(path)

        if stored is not None:
            return stored

        logger.info(
            "Feature cache miss or schema changed — rebuilding."
        )

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training
        )

        self._atomic_write(features, path)

        return features
