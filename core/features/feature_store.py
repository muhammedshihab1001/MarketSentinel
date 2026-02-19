import os
import logging
import hashlib
import re
import sys
import uuid
from typing import Optional

import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer
from core.indicators.technical_indicators import TechnicalIndicators
from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES,
    get_schema_signature
)

logger = logging.getLogger(__name__)


class FeatureStore:

    FEATURE_DIR = os.getenv("FEATURE_STORE_PATH", "data/features")

    REQUIRED_COLUMNS = {"date", "close", "ticker"}

    CACHE_VERSION = "v15"
    MAX_CACHE_FILES_PER_TICKER = 6

    MIN_ROWS_REQUIRED = 100
    MIN_FILE_BYTES = 5_000

    ABS_FEATURE_LIMIT = 1e5
    MIN_ROW_STABILITY_RATIO = 0.65
    MIN_FEATURE_VARIANCE = 1e-10

    def __init__(self):
        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = get_schema_signature()[:12]
        self.engineer_hash = self._fingerprint_engineer()[:12]
        self.env_hash = self._environment_fingerprint()[:12]

    ########################################################
    # NUMERICAL FIREWALL
    ########################################################

    def _validate_numeric_integrity(self, df: pd.DataFrame):

        features = df.loc[:, MODEL_FEATURES]
        arr = features.to_numpy(dtype=float)

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature values detected.")

        row_valid = np.isfinite(arr).all(axis=1)
        stability_ratio = row_valid.mean()

        if stability_ratio < self.MIN_ROW_STABILITY_RATIO:
            raise RuntimeError(
                f"Row stability collapse detected: {stability_ratio:.2f}"
            )

        variances = np.var(arr, axis=0)

        if np.min(variances) < self.MIN_FEATURE_VARIANCE:
            raise RuntimeError("Feature variance collapse detected.")

        if arr.size > 0 and np.abs(arr).max() > self.ABS_FEATURE_LIMIT:
            raise RuntimeError("Feature explosion detected.")

    ########################################################
    # DETERMINISTIC HASHING
    ########################################################

    def _stable_hash_df(self, df: pd.DataFrame) -> str:

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # deterministic CSV-based hash
        payload = df.to_csv(index=False).encode()
        return hashlib.sha256(payload).hexdigest()

    def _fingerprint_engineer(self) -> str:

        def module_hash(module):
            path = sys.modules[module.__module__].__file__
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        payload = (
            module_hash(FeatureEngineer)
            + module_hash(TechnicalIndicators)
        )

        return hashlib.sha256(payload.encode()).hexdigest()

    def _environment_fingerprint(self) -> str:

        payload = (
            sys.version +
            pd.__version__ +
            np.__version__
        )

        return hashlib.sha256(payload.encode()).hexdigest()

    ########################################################

    def _dataset_hash(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame]
    ) -> str:

        h = hashlib.sha256()

        price_core = price_df[["ticker", "date", "close"]].copy()
        h.update(self._stable_hash_df(price_core).encode())

        if sentiment_df is not None and not sentiment_df.empty:
            sent_core = sentiment_df.copy()

            if "avg_sentiment" in sent_core.columns:
                sent_core = sent_core[["date", "avg_sentiment"]]

                sent_core["date"] = pd.to_datetime(
                    sent_core["date"], utc=True
                )

                sent_core = sent_core.sort_values("date").reset_index(drop=True)

                h.update(sent_core.to_csv(index=False).encode())

        return h.hexdigest()[:20]

    ########################################################
    # CACHE CLEANUP
    ########################################################

    def _cleanup_old_cache(self, ticker: str):

        ticker = re.sub(r"[^A-Za-z0-9_]", "_", ticker)

        files = [
            f for f in os.listdir(self.FEATURE_DIR)
            if f.startswith(self.CACHE_VERSION + "_" + ticker)
        ]

        if len(files) <= self.MAX_CACHE_FILES_PER_TICKER:
            return

        files.sort()
        for f in files[:-self.MAX_CACHE_FILES_PER_TICKER]:
            try:
                os.remove(os.path.join(self.FEATURE_DIR, f))
            except Exception:
                pass

    ########################################################

    def get_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str = "unknown",
        training: bool = False
    ):

        # structural validation
        if not self.REQUIRED_COLUMNS.issubset(price_df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        if price_df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate rows detected in price data.")

        dataset_hash = self._dataset_hash(price_df, sentiment_df)

        suffix = "train" if training else "infer"
        ticker_safe = re.sub(r"[^A-Za-z0-9_]", "_", ticker)

        path = os.path.join(
            self.FEATURE_DIR,
            f"{self.CACHE_VERSION}_"
            f"{ticker_safe}_{suffix}_"
            f"{dataset_hash}_"
            f"{self.schema_hash}_"
            f"{self.engineer_hash}_"
            f"{self.env_hash}.parquet"
        )

        if os.path.exists(path) and os.path.getsize(path) >= self.MIN_FILE_BYTES:
            try:
                df = pd.read_parquet(path)
                validate_feature_schema(df.loc[:, MODEL_FEATURES])
                self._validate_numeric_integrity(df)
                return df
            except Exception:
                os.remove(path)

        logger.info("Feature cache miss — rebuilding.")

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training,
            ticker=ticker
        )

        validate_feature_schema(features.loc[:, MODEL_FEATURES])
        self._validate_numeric_integrity(features)

        tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
        features.to_parquet(
            tmp_path,
            index=False,
            engine="pyarrow",
            compression="zstd"
        )
        os.replace(tmp_path, path)

        self._cleanup_old_cache(ticker)

        return features
