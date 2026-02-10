import os
import json
import hashlib
from typing import Dict, Any

import pandas as pd

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import validate_feature_schema


class FeatureStore:
    """
    Institutional Feature Store (Phase 2).

    Guarantees:
    - schema validation
    - dataset fingerprint lineage
    - atomic persistence
    - corruption auto-rebuild
    - training/inference parity
    - deterministic feature sets
    """

    FEATURE_DIR = "data/features"
    META_SUFFIX = ".meta.json"

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

    def _fingerprint(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame):

        hasher = hashlib.sha256()

        # deterministic hashing
        hasher.update(
            pd.util.hash_pandas_object(
                price_df.sort_values("date"),
                index=False
            ).values
        )

        hasher.update(
            pd.util.hash_pandas_object(
                sentiment_df.sort_values("date"),
                index=False
            ).values
        )

        return hasher.hexdigest()

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

            validate_feature_schema(df)

            return df.sort_values("date"), meta

        except Exception as e:

            # corruption detected → delete + rebuild
            try:
                os.remove(path)
            except Exception:
                pass

            try:
                os.remove(meta_path)
            except Exception:
                pass

            return None, None

    # --------------------------------------------------

    def _atomic_save(self, df, path, meta, meta_path):

        tmp_path = path + ".tmp"
        tmp_meta = meta_path + ".tmp"

        df = df.sort_values("date").drop_duplicates("date")

        validate_feature_schema(df)

        df.to_parquet(tmp_path, index=False)

        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=4)

        os.replace(tmp_path, path)
        os.replace(tmp_meta, meta_path)

    # --------------------------------------------------

    def get_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str = "unknown"
    ):

        if price_df.empty:
            raise RuntimeError("FeatureStore received empty price dataframe.")

        if "date" not in price_df.columns:
            raise RuntimeError("Price dataframe missing 'date' column.")

        path = self._feature_path(ticker)
        meta_path = self._meta_path(ticker)

        dataset_fp = self._fingerprint(price_df, sentiment_df)

        stored, meta = self._load_features(path, meta_path)

        # --------------------------------------------------
        # CASE 1 — No stored features
        # --------------------------------------------------

        if stored is None:

            features = self.engineer.build_feature_pipeline(
                price_df,
                sentiment_df
            )

            meta = {
                "dataset_fingerprint": dataset_fp,
                "feature_count": len(features.columns),
                "row_count": len(features)
            }

            self._atomic_save(features, path, meta, meta_path)

            return features

        # --------------------------------------------------
        # CASE 2 — Fingerprint match → guaranteed parity
        # --------------------------------------------------

        if meta["dataset_fingerprint"] == dataset_fp:
            return stored

        # --------------------------------------------------
        # CASE 3 — Fingerprint mismatch → rebuild
        # --------------------------------------------------

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df
        )

        meta = {
            "dataset_fingerprint": dataset_fp,
            "feature_count": len(features.columns),
            "row_count": len(features)
        }

        self._atomic_save(features, path, meta, meta_path)

        return features
