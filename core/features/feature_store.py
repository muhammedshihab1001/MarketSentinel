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

    FEATURE_DIR = os.path.abspath("data/features")

    REQUIRED_COLUMNS = {"date", "close", "ticker"}

    CACHE_VERSION = "v12"
    MAX_CACHE_FILES_PER_TICKER = 6

    MIN_ROWS_REQUIRED = 100
    MIN_FILE_BYTES = 5_000

    ABS_FEATURE_LIMIT = 1e5
    MIN_ROW_STABILITY_RATIO = 0.65

    def __init__(self):

        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = get_schema_signature()[:12]
        self.engineer_hash = self._fingerprint_engineer()[:12]
        self.env_hash = self._environment_fingerprint()[:12]

    ########################################################
    # SAFE HASH (NO PANDAS HASH)
    ########################################################

    def _stable_hash_df(self, df: pd.DataFrame):

        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values(list(df.columns)).reset_index(drop=True)

        payload = df.to_csv(index=False).encode()

        return hashlib.sha256(payload).hexdigest()

    ########################################################

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

    ########################################################

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

        price_core = price_df[["date", "close"]].copy()
        h.update(self._stable_hash_df(price_core).encode())

        if sentiment_df is not None and not sentiment_df.empty:

            sent_core = sentiment_df[["date"]].copy()
            h.update(self._stable_hash_df(sent_core).encode())

        return h.hexdigest()[:20]

    ########################################################

    def _sanitize_ticker(self, ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", ticker)

    ########################################################

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

    ########################################################
    # CACHE GC (VERY IMPORTANT)
    ########################################################

    def _cleanup_old_cache(self, ticker):

        ticker = self._sanitize_ticker(ticker)

        files = sorted(
            [
                f for f in os.listdir(self.FEATURE_DIR)
                if f"_{ticker}_" in f
            ],
            reverse=True
        )

        for f in files[self.MAX_CACHE_FILES_PER_TICKER:]:

            try:
                os.remove(os.path.join(self.FEATURE_DIR, f))
            except Exception:
                pass

    ########################################################

    def _atomic_write(
        self,
        df: pd.DataFrame,
        path: str,
        dataset_hash: str,
        ticker: str,
        training: bool
    ):

        if len(df) < self.MIN_ROWS_REQUIRED:
            raise RuntimeError("Feature collapse detected.")

        validated = validate_feature_schema(
            df.loc[:, MODEL_FEATURES]
        )

        if isinstance(validated, tuple):
            validated = validated[0]

        arr = validated.to_numpy()

        if arr.size > 0 and np.abs(arr).max() > self.ABS_FEATURE_LIMIT:
            raise RuntimeError("Feature explosion detected.")

        df = df.copy()

        df["__dataset_hash"] = dataset_hash
        df["__schema_hash"] = self.schema_hash
        df["__engineer_hash"] = self.engineer_hash

        tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"

        df.to_parquet(
            tmp_path,
            index=False,
            engine="pyarrow",
            compression="zstd"
        )

        os.replace(tmp_path, path)

        self._cleanup_old_cache(ticker)

    ########################################################

    def _load_features(self, path: str, dataset_hash: str):

        if not os.path.exists(path):
            return None

        if os.path.getsize(path) < self.MIN_FILE_BYTES:
            os.remove(path)
            return None

        try:

            df = pd.read_parquet(path, engine="pyarrow")

            meta = [
                "__dataset_hash",
                "__schema_hash",
                "__engineer_hash"
            ]

            if not all(c in df.columns for c in meta):
                os.remove(path)
                return None

            if df["__dataset_hash"].iloc[0] != dataset_hash:
                os.remove(path)
                return None

            df.drop(columns=meta, inplace=True)

            validate_feature_schema(df.loc[:, MODEL_FEATURES])

            return df

        except Exception:

            logger.exception("Feature corruption detected — rebuilding.")

            try:
                os.remove(path)
            except Exception:
                pass

            return None

    ########################################################

    def get_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str = "unknown",
        training: bool = False
    ):

        dataset_hash = self._dataset_hash(
            price_df,
            sentiment_df
        )

        path = self._feature_path(
            ticker,
            dataset_hash,
            training
        )

        stored = self._load_features(path, dataset_hash)

        if stored is not None:
            return stored

        logger.info("Feature cache miss — rebuilding.")

        features = self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=training,
            ticker=ticker
        )

        self._atomic_write(
            features,
            path,
            dataset_hash,
            ticker,
            training
        )

        return features
