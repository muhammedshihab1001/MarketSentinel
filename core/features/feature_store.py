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
    Institutional Feature Store.

    Guarantees:
    - schema lineage
    - deterministic fingerprints
    - atomic persistence
    - corruption recovery
    - strict feature validation without destroying dataset lineage
    """

    FEATURE_DIR = "data/features"
    META_SUFFIX = ".meta.json"
    META_VERSION = "4.0"

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
    # DATASET FINGERPRINT
    # --------------------------------------------------

    def _stable_hash(self, df: pd.DataFrame):

        df = (
            df.sort_index(axis=1)
            .sort_values(by=list(df.columns))
            .reset_index(drop=True)
        )

        return pd.util.hash_pandas_object(
            df,
            index=False
        ).values.tobytes()

    def _fingerprint(self, price_df, sentiment_df):

        hasher = hashlib.sha256()

        hasher.update(self._stable_hash(price_df))
        hasher.update(self._stable_hash(sentiment_df))
        hasher.update(SCHEMA_VERSION.encode())

        return hasher.hexdigest()

    # --------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------

    def _validate_dataset_structure(self, df: pd.DataFrame):

        missing = self.REQUIRED_DATASET_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Dataset missing required columns: {missing}"
            )

    # --------------------------------------------------

    def _load_features(self, path, meta_path):

        if not os.path.exists(path) or not os.path.exists(meta_path):
            return None, None

        try:

            df = pd.read_parquet(path)

            if df.empty:
                return None, None

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("metadata_type") != "feature_store":
                raise RuntimeError("Invalid metadata type.")

            self._validate_dataset_structure(df)

            validate_feature_schema(df[list(MODEL_FEATURES)])

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

    def _atomic_save(self, df, path, meta, meta_path):

        tmp_path = path + ".tmp"
        tmp_meta = meta_path + ".tmp"

        df = df.sort_values("date").drop_duplicates("date")

        self._validate_dataset_structure(df)

        validate_feature_schema(df[list(MODEL_FEATURES)])

        df.to_parquet(tmp_path, index=False)

        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=4)

        os.replace(tmp_path, path)
        os.replace(tmp_meta, meta_path)

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
            "row_count": len(features)
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

        # CASE 1 — no stored features

        if stored is None:

            features = self.engineer.build_feature_pipeline(
                price_df,
                sentiment_df,
                training=False
            )

            meta = self._build_metadata(
                dataset_fp,
                features
            )

            self._atomic_save(features, path, meta, meta_path)

            return features

        # CASE 2 — lineage match

        if (
            meta.get("dataset_fingerprint") == dataset_fp
            and meta.get("schema_signature") == get_schema_signature()
        ):
            return stored

        logger.info(
            "Feature rebuild triggered due to lineage change."
        )

        # CASE 3 — rebuild

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=False
        )

        meta = self._build_metadata(
            dataset_fp,
            features
        )

        self._atomic_save(features, path, meta, meta_path)

        return features
