import os
import json
import hashlib
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    validate_feature_schema,
    get_schema_signature,
    SCHEMA_VERSION,
    MODEL_FEATURES
)

logger = logging.getLogger("marketsentinel.feature_store")


class FeatureStore:
    """
    Institutional Feature Store.

    Guarantees:
    - lineage-bound datasets
    - deterministic fingerprints
    - crash-safe persistence
    - schema enforcement
    - corruption auto-recovery
    - feature ordering guarantees
    """

    FEATURE_DIR = "data/features"
    META_SUFFIX = ".meta.json"
    META_VERSION = "5.0"

    REQUIRED_DATASET_COLUMNS = {"date"}

    def __init__(self):
        os.makedirs(self.FEATURE_DIR, exist_ok=True)
        self.engineer = FeatureEngineer()

    # --------------------------------------------------

    def _feature_path(self, ticker: str):
        return f"{self.FEATURE_DIR}/{ticker}_features.parquet"

    def _meta_path(self, ticker: str):
        return f"{self._feature_path(ticker)}{self.META_SUFFIX}"

    # --------------------------------------------------
    # DETERMINISTIC HASH
    # --------------------------------------------------

    def _stable_hash(self, df: pd.DataFrame):

        if df.empty:
            raise RuntimeError("Cannot hash empty dataframe.")

        df_copy = df.copy()

        df_copy = df_copy.reindex(sorted(df_copy.columns), axis=1)

        float_cols = df_copy.select_dtypes(
            include=["float32", "float64"]
        ).columns

        df_copy[float_cols] = df_copy[float_cols].round(10)

        df_copy = df_copy.sort_values(
            by=list(df_copy.columns)
        ).reset_index(drop=True)

        return pd.util.hash_pandas_object(
            df_copy,
            index=False
        ).values.tobytes()

    def _fingerprint(self, price_df, sentiment_df):

        hasher = hashlib.sha256()

        hasher.update(self._stable_hash(price_df))
        hasher.update(self._stable_hash(sentiment_df))
        hasher.update(get_schema_signature().encode())

        return hasher.hexdigest()

    # --------------------------------------------------
    # VALIDATION
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
    # ATOMIC WRITE
    # --------------------------------------------------

    def _atomic_write(self, df, path, meta, meta_path):

        tmp_data = path + ".tmp"
        tmp_meta = meta_path + ".tmp"

        df = df.sort_values("date").reset_index(drop=True)

        self._validate_dataset_structure(df)

        feature_block = df[MODEL_FEATURES]

        if list(feature_block.columns) != MODEL_FEATURES:
            raise RuntimeError(
                "Feature ordering violation detected."
            )

        validate_feature_schema(feature_block)

        # write parquet safely
        df.to_parquet(tmp_data, index=False)

        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_data, path)
        os.replace(tmp_meta, meta_path)

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def _load_features(self, path, meta_path):

        if not os.path.exists(path) or not os.path.exists(meta_path):
            return None, None

        try:

            df = pd.read_parquet(path)

            if df.empty:
                raise RuntimeError("Stored dataset empty.")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("metadata_type") != "feature_store":
                raise RuntimeError("Invalid metadata type.")

            if meta.get("schema_signature") != get_schema_signature():
                raise RuntimeError("Schema mismatch.")

            self._validate_dataset_structure(df)

            feature_block = df[MODEL_FEATURES]

            if list(feature_block.columns) != MODEL_FEATURES:
                raise RuntimeError("Feature ordering corrupted.")

            validate_feature_schema(feature_block)

            return df.sort_values("date"), meta

        except Exception:

            logger.exception(
                "Feature corruption detected — rebuilding."
            )

            try:
                os.remove(path)
                os.remove(meta_path)
            except Exception:
                pass

            return None, None

    # --------------------------------------------------

    def _build_metadata(
        self,
        dataset_fp: str,
        features: pd.DataFrame
    ) -> Dict[str, Any]:

        return {
            "metadata_type": "feature_store",
            "meta_version": self.META_VERSION,
            "schema_version": SCHEMA_VERSION,
            "schema_signature": get_schema_signature(),
            "dataset_fingerprint": dataset_fp,
            "feature_count": len(MODEL_FEATURES),
            "features": list(MODEL_FEATURES),
            "row_count": len(features),
            "created_at": pd.Timestamp.utcnow().isoformat()
        }

    # --------------------------------------------------

    def get_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str = "unknown"
    ):

        if price_df.empty:
            raise RuntimeError(
                "FeatureStore received empty price dataframe."
            )

        if "date" not in price_df.columns:
            raise RuntimeError(
                "Price dataframe missing 'date' column."
            )

        path = self._feature_path(ticker)
        meta_path = self._meta_path(ticker)

        dataset_fp = self._fingerprint(
            price_df,
            sentiment_df
        )

        stored, meta = self._load_features(path, meta_path)

        # --------------------------------------------
        # NO STORED DATA
        # --------------------------------------------

        if stored is None:

            features = self.engineer.build_feature_pipeline(
                price_df,
                sentiment_df,
                training=False
            )

            meta = self._build_metadata(dataset_fp, features)

            self._atomic_write(features, path, meta, meta_path)

            return features

        # --------------------------------------------
        # LINEAGE MATCH
        # --------------------------------------------

        if (
            meta.get("dataset_fingerprint") == dataset_fp
            and meta.get("schema_signature") == get_schema_signature()
        ):
            return stored

        logger.info(
            "Feature rebuild triggered due to lineage change."
        )

        # --------------------------------------------
        # REBUILD
        # --------------------------------------------

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=False
        )

        meta = self._build_metadata(dataset_fp, features)

        self._atomic_write(features, path, meta, meta_path)

        return features
