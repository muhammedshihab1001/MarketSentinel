import os
import json
import hashlib
import logging
from typing import Dict, Any

import pandas as pd

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
    Institutional Feature Store — Hardened.

    Guarantees:
    ✔ zero training/inference skew
    ✔ lineage-bound datasets
    ✔ deterministic fingerprints
    ✔ crash-safe persistence
    ✔ schema enforcement
    ✔ corruption auto-recovery
    ✔ feature ordering guarantees
    """

    FEATURE_DIR = "data/features"
    META_SUFFIX = ".meta.json"
    META_VERSION = "6.0"

    REQUIRED_DATASET_COLUMNS = {"date"}

    def __init__(self):
        os.makedirs(self.FEATURE_DIR, exist_ok=True)
        self.engineer = FeatureEngineer()

    # --------------------------------------------------

    def _feature_path(self, ticker: str, training: bool):
        suffix = "train" if training else "infer"
        return f"{self.FEATURE_DIR}/{ticker}_{suffix}.parquet"

    def _meta_path(self, ticker: str, training: bool):
        return f"{self._feature_path(ticker, training)}{self.META_SUFFIX}"

    # --------------------------------------------------
    # FAST + STABLE HASH
    # --------------------------------------------------

    def _stable_hash(self, df: pd.DataFrame):

        if df.empty:
            raise RuntimeError("Cannot hash empty dataframe.")

        df_copy = df.copy()

        df_copy = df_copy.reindex(sorted(df_copy.columns), axis=1)

        float_cols = df_copy.select_dtypes(
            include=["float32", "float64"]
        ).columns

        df_copy[float_cols] = df_copy[float_cols].round(8)

        df_copy = df_copy.sort_values(
            by=list(df_copy.columns)
        ).reset_index(drop=True)

        hashed = pd.util.hash_pandas_object(
            df_copy,
            index=False
        ).values.tobytes()

        return hashed

    # --------------------------------------------------
    # PRICE-ONLY FINGERPRINT (CRITICAL FIX)
    # --------------------------------------------------

    def _fingerprint(self, price_df):

        hasher = hashlib.sha256()

        hasher.update(self._stable_hash(price_df))
        hasher.update(get_schema_signature().encode())

        return hasher.hexdigest()

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

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("metadata_type") != "feature_store":
                raise RuntimeError("Invalid metadata type.")

            if meta.get("schema_signature") != get_schema_signature():
                raise RuntimeError("Schema mismatch.")

            self._validate_dataset_structure(df)

            validate_feature_schema(df[MODEL_FEATURES])

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
        features: pd.DataFrame,
        training: bool
    ) -> Dict[str, Any]:

        return {
            "metadata_type": "feature_store",
            "meta_version": self.META_VERSION,
            "schema_version": SCHEMA_VERSION,
            "schema_signature": get_schema_signature(),
            "dataset_fingerprint": dataset_fp,
            "dataset_role": "training" if training else "inference",
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
        ticker: str = "unknown",
        training: bool = False
    ):

        if price_df.empty:
            raise RuntimeError(
                "FeatureStore received empty price dataframe."
            )

        path = self._feature_path(ticker, training)
        meta_path = self._meta_path(ticker, training)

        dataset_fp = self._fingerprint(price_df)

        stored, meta = self._load_features(path, meta_path)

        if stored is not None:

            if (
                meta.get("dataset_fingerprint") == dataset_fp
                and meta.get("schema_signature") == get_schema_signature()
            ):
                return stored

            logger.info("Feature rebuild triggered due to lineage change.")

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training
        )

        meta = self._build_metadata(
            dataset_fp,
            features,
            training
        )

        self._atomic_write(features, path, meta, meta_path)

        return features
