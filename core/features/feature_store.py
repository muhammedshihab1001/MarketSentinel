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
    Institutional Feature Store — Production Safe
    """

    FEATURE_DIR = "data/features"

    REQUIRED_COLUMNS = {"date", "close"}

    CACHE_VERSION = "v3"
    MAX_CACHE_FILES_PER_TICKER = 6

    ##################################################

    def __init__(self):

        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = hashlib.sha256(
            get_schema_signature().encode()
        ).hexdigest()[:10]

        self.engineer_hash = self._fingerprint_engineer()[:10]

    ##################################################
    # SAFE TICKER
    ##################################################

    def _sanitize_ticker(self, ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", ticker)

    ##################################################
    # BYTECODE FINGERPRINT
    ##################################################

    def _fingerprint_engineer(self) -> str:

        code_bytes = (
            FeatureEngineer.build_feature_pipeline
            .__code__.co_code
        )

        return hashlib.sha256(code_bytes).hexdigest()

    ##################################################
    #  STRONG DATASET HASH (PRICE + SENTIMENT)
    ##################################################

    def _dataset_hash(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame]
    ) -> str:

        price_df = price_df.sort_values("date")

        head = price_df.head(25)
        tail = price_df.tail(25)

        sample = pd.concat([head, tail])

        h = hashlib.sha256()

        price_arr = pd.util.hash_pandas_object(
            sample[["date", "close"]],
            index=False
        ).values

        h.update(price_arr.tobytes())
        h.update(str(len(price_df)).encode())

        ##################################################
        # SENTIMENT INCLUDED (CRITICAL)
        ##################################################

        if sentiment_df is not None and not sentiment_df.empty:

            sentiment_df = sentiment_df.sort_values("date")

            s_head = sentiment_df.head(15)
            s_tail = sentiment_df.tail(15)

            s_sample = pd.concat([s_head, s_tail])

            cols = [
                c for c in
                ["date", "avg_sentiment", "news_count"]
                if c in s_sample.columns
            ]

            if cols:

                sent_arr = pd.util.hash_pandas_object(
                    s_sample[cols],
                    index=False
                ).values

                h.update(sent_arr.tobytes())
                h.update(str(len(sentiment_df)).encode())

        return h.hexdigest()[:16]

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
            f"{self.engineer_hash}.parquet"
        )

    ##################################################
    # CACHE PRUNING — SAFER
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

    def _fsync_dir(self, directory):

        if os.name == "nt":
            return

        fd = os.open(directory, os.O_DIRECTORY)
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

        if not np.isfinite(feature_block.to_numpy()).all():
            raise RuntimeError("Non-finite feature values.")

        df.to_parquet(tmp_path, index=False)

        os.replace(tmp_path, path)
        self._fsync_dir(os.path.dirname(path))

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
