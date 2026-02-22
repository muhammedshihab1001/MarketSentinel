import time
import threading
import numpy as np
import pandas as pd
import logging
import os
import hashlib
import json
from datetime import timedelta

from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
    DTYPE
)
from core.market.universe import MarketUniverse

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    PIPELINE_FAILURES,
    CACHE_HITS,
    CACHE_MISSES,
    INFERENCE_IN_PROGRESS
)

logger = logging.getLogger("marketsentinel.pipeline")


class CircuitBreaker:
    def __init__(self, threshold=3, cooldown=120):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure = None
        self._lock = threading.Lock()

    def allow(self):
        with self._lock:
            if self.failures < self.threshold:
                return True

            if self.last_failure and (time.time() - self.last_failure) > self.cooldown:
                logger.warning("Circuit breaker reset.")
                self.failures = 0
                return True

            logger.critical("Inference blocked by circuit breaker.")
            return False

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure = time.time()

    def record_success(self):
        with self._lock:
            self.failures = 0


class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    TOP_K = 3
    BOTTOM_K = 3
    MIN_PROB_STD = 1e-6
    WEIGHT_TOLERANCE = 1e-6

    CACHE_TTL = int(os.getenv("INFERENCE_CACHE_TTL_SECONDS", "600"))
    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.models = ModelLoader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.breaker = CircuitBreaker()

        _ = self.models.xgb
        self._validate_models_loaded()

    ############################################################

    def _validate_models_loaded(self):

        container = self.models._xgb_container

        if container is None:
            raise RuntimeError("Model container missing.")

        if container.schema_signature != get_schema_signature():
            raise RuntimeError(
                "Schema signature mismatch between training and inference."
            )

        logger.info("Model + schema signature verified.")

    ############################################################

    def _validate_features_exist(self, df: pd.DataFrame):

        missing = set(MODEL_FEATURES) - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Missing required model features: {missing}"
            )

    ############################################################
    # DETERMINISTIC SORT GUARD
    ############################################################

    def _deterministic_sort(self, df):

        # enforce deterministic ranking (score, ticker)
        return df.sort_values(
            ["score", "ticker"],
            ascending=[True, True]
        )

    ############################################################

    def _construct_portfolio(self, latest_df):

        if "volatility" not in latest_df.columns:
            raise RuntimeError("Volatility feature missing for weighting.")

        n_assets = len(latest_df)

        if n_assets < 4:
            raise RuntimeError("Insufficient assets for portfolio construction.")

        latest_df = self._deterministic_sort(latest_df)

        k_long = min(self.TOP_K, n_assets // 2)
        k_short = min(self.BOTTOM_K, n_assets // 2)

        longs = latest_df.tail(k_long)
        shorts = latest_df.head(k_short)

        long_vol = longs["volatility"].replace(0, 1e-6)
        short_vol = shorts["volatility"].replace(0, 1e-6)

        long_weights = (1.0 / long_vol)
        short_weights = (1.0 / short_vol)

        long_weights /= long_weights.sum()
        short_weights /= short_weights.sum()

        long_weights *= self.TARGET_GROSS_EXPOSURE / 2
        short_weights *= self.TARGET_GROSS_EXPOSURE / 2

        weights = {}

        for t, w in zip(longs["ticker"], long_weights):
            weights[t] = float(w)

        for t, w in zip(shorts["ticker"], short_weights):
            weights[t] = -float(w)

        gross = sum(abs(v) for v in weights.values())

        if abs(gross - self.TARGET_GROSS_EXPOSURE) > self.WEIGHT_TOLERANCE:
            raise RuntimeError(
                f"Gross exposure mismatch: {gross}"
            )

        return weights

    ############################################################

    def _run_model_and_construct(self, latest_df, use_cache: bool):

        self._validate_features_exist(latest_df)

        feature_df = validate_feature_schema(
            latest_df.loc[:, MODEL_FEATURES],
            mode="inference"
        ).astype(DTYPE)

        t0 = time.time()
        probs = self.models.xgb.predict_proba(feature_df)[:, 1]
        latency = time.time() - t0

        MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
        MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)

        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        if np.std(probs) < self.MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected.")

        logger.info(
            "Inference prob stats | mean=%.4f std=%.4f",
            float(np.mean(probs)),
            float(np.std(probs))
        )

        latest_df = latest_df.copy()
        latest_df["score"] = probs
        latest_df["rank_pct"] = latest_df["score"].rank(pct=True)

        latest_df["signal"] = latest_df["rank_pct"].apply(
            lambda x: "LONG" if x >= 0.7 else (
                "SHORT" if x <= 0.3 else "NEUTRAL"
            )
        )

        weights = self._construct_portfolio(latest_df)

        portfolio_rows = []

        for _, row in latest_df.iterrows():
            portfolio_rows.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "score": float(row["score"]),
                "signal": row["signal"],
                "weight": float(weights.get(row["ticker"], 0.0))
            })

        # 🔐 Drift trigger (non-blocking)
        try:
            self.drift_detector.check_drift(feature_df)
        except Exception as e:
            logger.warning("Drift check failed (non-blocking): %s", str(e))

        return portfolio_rows