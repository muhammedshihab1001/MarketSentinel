import os
import logging
from typing import Optional

import pandas as pd

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)

logger = logging.getLogger("marketsentinel.feature_store")


class FeatureStore:
    """
    Deterministic feature cache.

    Guarantees:
    - rebuild on cache miss
    - atomic persistence
    - corruption recovery
    - schema validation
    - stable feature ordering
    """

    FEATURE_DIR = "data/features"
    REQUIRED_DATASET_COLUMNS = {"date"}

    def __init__(self):
        os.makedirs(self.FEATURE_DIR, exist_ok=True)
        self.engineer = FeatureEngineer()

    def _feature_path(self, ticker: str, training: bool):
        suffix = "train" if training else "infer"
        return f"{self.FEATURE_DIR}/{ticker}_{suffix}.parquet"

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

    def _atomic_write(self, df: pd.DataFrame, path: str):

        tmp_path = path + ".tmp"

        df = df.sort_values("date").reset_index(drop=True)

        self._validate_dataset_structure(df)

        feature_block = df.loc[:, MODEL_FEATURES]

        if list(feature_block.columns) != MODEL_FEATURES:
            raise RuntimeError(
                "Feature ordering violation detected."
            )

        validate_feature_schema(feature_block)

        df.to_parquet(tmp_path, index=False)

        os.replace(tmp_path, path)

    def _load_features(self, path: str) -> Optional[pd.DataFrame]:

        if not os.path.exists(path):
            return None

        try:

            df = pd.read_parquet(path)

            self._validate_dataset_structure(df)

            validate_feature_schema(df.loc[:, MODEL_FEATURES])

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

        logger.info("Feature cache miss — rebuilding features.")

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training
        )

        self._atomic_write(features, path)

        return features
