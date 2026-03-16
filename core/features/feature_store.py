"""
MarketSentinel v4.2.0

Feature Store — per-ticker feature cache for INFERENCE.
"""

import os
import logging
import hashlib
import re
import sys
import uuid
import time
from typing import Optional
from collections import OrderedDict

import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer
from core.indicators.technical_indicators import TechnicalIndicators
from core.schema.feature_schema import get_schema_signature, MODEL_FEATURES

logger = logging.getLogger(__name__)


class FeatureStore:

    FEATURE_DIR = os.getenv("FEATURE_STORE_PATH", "data/features")

    REQUIRED_COLUMNS = {"date", "close", "ticker"}

    CACHE_VERSION = "v26"

    MAX_CACHE_FILES_PER_TICKER = 6
    MAX_TOTAL_CACHE_FILES = 2000

    MIN_FILE_BYTES = 5_000

    MEMORY_CACHE_LIMIT = 100
    MEMORY_CACHE_TTL = 120

    _memory_cache = OrderedDict()
    _memory_cache_ts = {}

    _cache_hits = 0
    _cache_misses = 0

    ########################################################

    def __init__(self):

        os.makedirs(self.FEATURE_DIR, exist_ok=True)

        self.engineer = FeatureEngineer()

        self.schema_hash = get_schema_signature()[:12]
        self.engineer_hash = self._fingerprint_engineer()[:12]
        self.env_hash = self._environment_fingerprint()[:12]

    ########################################################
    # BASIC SANITY
    ########################################################

    def _validate_basic_integrity(self, df: pd.DataFrame):

        if not {"date", "ticker"}.issubset(df.columns):
            raise RuntimeError("Feature dataset missing core columns.")

        if df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate feature rows detected.")

        numeric = df.select_dtypes("number")

        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise RuntimeError("Non-finite feature values detected.")

        # ensure model features exist
        missing = set(MODEL_FEATURES) - set(df.columns)

        if missing:
            raise RuntimeError(
                f"FeatureStore output missing MODEL_FEATURES: {missing}"
            )

    ########################################################
    # HASHING
    ########################################################

    def _stable_hash_df(self, df: pd.DataFrame) -> str:

        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], utc=True)

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

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
            sys.version
            + pd.__version__
            + np.__version__
        )

        return hashlib.sha256(payload.encode()).hexdigest()

    ########################################################
    # DATASET HASH
    ########################################################

    def _dataset_hash(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame]
    ) -> str:

        price_core = price_df[["ticker", "date", "close"]].copy()

        price_core["date"] = pd.to_datetime(price_core["date"], utc=True)

        price_core = price_core.sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True)

        if len(price_core) > 300:
            price_core = price_core.tail(300)

        h = hashlib.sha256()

        h.update(self._stable_hash_df(price_core).encode())

        return h.hexdigest()[:20]

    ########################################################
    # CACHE CLEANUP
    ########################################################

    def _cleanup_global_cache(self):

        files = [
            f for f in os.listdir(self.FEATURE_DIR)
            if f.endswith(".parquet")
        ]

        if len(files) > self.MAX_TOTAL_CACHE_FILES:

            files.sort()

            excess = files[:-self.MAX_TOTAL_CACHE_FILES]

            for f in excess:
                try:
                    os.remove(os.path.join(self.FEATURE_DIR, f))
                except Exception:
                    pass

    def _cleanup_old_cache(self, ticker: str):

        ticker = re.sub(r"[^A-Za-z0-9_]", "_", ticker)

        files = [
            f for f in os.listdir(self.FEATURE_DIR)
            if f.startswith(self.CACHE_VERSION + "_" + ticker)
        ]

        if len(files) > self.MAX_CACHE_FILES_PER_TICKER:

            files.sort()

            for f in files[:-self.MAX_CACHE_FILES_PER_TICKER]:

                try:
                    os.remove(os.path.join(self.FEATURE_DIR, f))
                except Exception:
                    pass

        self._cleanup_global_cache()

    ########################################################
    # MEMORY CACHE
    ########################################################

    def _set_memory_cache(self, key, df):

        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        else:

            if len(self._memory_cache) >= self.MEMORY_CACHE_LIMIT:

                old_key, _ = self._memory_cache.popitem(last=False)

                self._memory_cache_ts.pop(old_key, None)

        self._memory_cache[key] = df
        self._memory_cache_ts[key] = time.time()

    def _get_memory_cache(self, key):

        if key not in self._memory_cache:
            return None

        ts = self._memory_cache_ts.get(key)

        if ts and time.time() - ts > self.MEMORY_CACHE_TTL:

            self._memory_cache.pop(key, None)
            self._memory_cache_ts.pop(key, None)

            return None

        self._memory_cache.move_to_end(key)

        return self._memory_cache[key]

    ########################################################
    # MAIN ENTRY
    ########################################################

    def get_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame],
        ticker: str = "unknown",
        training: bool = False
    ):

        if not self.REQUIRED_COLUMNS.issubset(price_df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        if price_df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate rows detected in price data.")

        price_df = price_df.sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True)

        dataset_hash = self._dataset_hash(price_df, sentiment_df)

        ticker_safe = re.sub(r"[^A-Za-z0-9_]", "_", ticker)

        cache_key = (
            f"{self.CACHE_VERSION}_"
            f"{ticker_safe}_"
            f"{dataset_hash}_"
            f"{self.schema_hash}_"
            f"{self.engineer_hash}_"
            f"{self.env_hash}"
        )

        ####################################################
        # MEMORY CACHE
        ####################################################

        cached = self._get_memory_cache(cache_key)

        if cached is not None:

            FeatureStore._cache_hits += 1

            return cached

        FeatureStore._cache_misses += 1

        path = os.path.join(
            self.FEATURE_DIR,
            f"{cache_key}.parquet"
        )

        ####################################################
        # LOAD CACHE
        ####################################################

        if os.path.exists(path) and os.path.getsize(path) >= self.MIN_FILE_BYTES:

            try:

                df = pd.read_parquet(path)

                self._validate_basic_integrity(df)

                self._set_memory_cache(cache_key, df)

                return df

            except Exception:

                logger.warning("Corrupted feature cache removed.")

                try:
                    os.remove(path)
                except Exception:
                    pass

        ####################################################
        # REBUILD
        ####################################################

        logger.info("Feature cache miss — rebuilding for %s.", ticker)

        df = self.engineer._validate_price_frame(price_df, ticker)

        df = self.engineer.add_core_features(df)

        df = self.engineer.finalize(df)

        df = df.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        self._validate_basic_integrity(df)

        ####################################################
        # SAFE WRITE
        ####################################################

        tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"

        try:

            df.to_parquet(
                tmp_path,
                index=False,
                engine="pyarrow",
                compression="zstd"
            )

        except Exception:

            logger.warning("pyarrow unavailable — using fallback parquet.")

            df.to_parquet(tmp_path, index=False)

        os.replace(tmp_path, path)

        self._cleanup_old_cache(ticker)

        self._set_memory_cache(cache_key, df)

        return df