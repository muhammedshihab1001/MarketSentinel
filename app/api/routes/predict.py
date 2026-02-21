import time
import threading
import numpy as np
import pandas as pd
import logging
import os
import hashlib
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


############################################################
# CIRCUIT BREAKER
############################################################

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


############################################################
# INFERENCE PIPELINE
############################################################

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    MAX_NET_EXPOSURE = 0.2
    TOP_K = 3
    BOTTOM_K = 3

    LATENCY_GUARD_SECONDS = float(
        os.getenv("HARD_LATENCY_LIMIT_SECONDS", "5.0")
    )

    CACHE_TTL = int(
        os.getenv("INFERENCE_CACHE_TTL_SECONDS", "600")
    )

    INFERENCE_LOOKBACK_DAYS = int(
        os.getenv("INFERENCE_LOOKBACK_DAYS", "400")
    )

    MIN_PROB_STD = 1e-6

    ############################################################

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

    def _dataset_hash(self, df: pd.DataFrame) -> str:

        ordered = (
            df.sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        feature_block = validate_feature_schema(
            ordered.loc[:, MODEL_FEATURES],
            mode="inference"
        )

        payload = (
            feature_block
            .astype(float)
            .round(8)
            .to_csv(index=False)
            .encode()
        )

        return hashlib.sha256(payload).hexdigest()[:16]

    ############################################################

    def _construct_portfolio(self, latest_df):

        n_assets = len(latest_df)

        if n_assets < 4:
            raise RuntimeError("Insufficient assets for portfolio construction.")

        k_long = min(self.TOP_K, n_assets // 2)
        k_short = min(self.BOTTOM_K, n_assets // 2)

        ranked = latest_df.sort_values("score")

        longs = ranked.tail(k_long)
        shorts = ranked.head(k_short)

        long_vol = longs["volatility"].replace(0, 1e-6)
        short_vol = shorts["volatility"].replace(0, 1e-6)

        long_weights = 1.0 / long_vol
        short_weights = 1.0 / short_vol

        long_weights /= long_weights.sum()
        short_weights /= short_weights.sum()

        long_weights *= self.TARGET_GROSS_EXPOSURE / 2
        short_weights *= self.TARGET_GROSS_EXPOSURE / 2

        weights = {}

        for t, w in zip(longs["ticker"], long_weights):
            weights[t] = float(w)

        for t, w in zip(shorts["ticker"], short_weights):
            weights[t] = -float(w)

        return weights

    ############################################################
    # MAIN ENTRY
    ############################################################

    def run_batch(self, tickers: list[str]):

        start_pipeline = time.time()

        MarketUniverse.validate_subset(tickers)

        if not self.breaker.allow():
            PIPELINE_FAILURES.labels(stage="circuit_open").inc()
            raise RuntimeError("Inference circuit breaker open.")

        INFERENCE_IN_PROGRESS.inc()

        try:

            datasets = []

            end_date = pd.Timestamp.utcnow().date()
            start_date = end_date - timedelta(days=self.INFERENCE_LOOKBACK_DAYS)

            for ticker in tickers:

                price_df = self.market_data.get_price_data(
                    ticker=ticker,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )

                if price_df is None or price_df.empty:
                    continue

                dataset = self.feature_store.get_features(
                    price_df,
                    sentiment_df=None,
                    ticker=ticker,
                    training=False
                )

                if dataset is not None and not dataset.empty:
                    datasets.append(dataset)

            if not datasets:
                raise RuntimeError("No valid datasets built.")

            df = pd.concat(datasets, ignore_index=True)

            latest_date = df["date"].max()
            latest_df = df[df["date"] == latest_date].copy()

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference"
            )

            feature_df = feature_df.astype(DTYPE)

            dataset_hash = self._dataset_hash(latest_df)

            cache_key = self.cache.build_key({
                "model_version": self.models.xgb_version,
                "dataset_hash": dataset_hash,
                "schema": get_schema_signature()
            })

            cached = self.cache.get(cache_key)
            if cached:
                CACHE_HITS.inc()
                return cached

            CACHE_MISSES.inc()

            t0 = time.time()
            probs = self.models.xgb.predict_proba(feature_df)[:, 1]
            latency = time.time() - t0

            MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
            MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)

            probs = np.clip(probs, 1e-6, 1 - 1e-6)

            latest_df["score"] = probs

            weights = self._construct_portfolio(latest_df)

            # -------------------------------------------------
            # API-CONTRACT OUTPUT FORMAT
            # -------------------------------------------------

            portfolio_rows = []

            for ticker, weight in weights.items():

                score = float(
                    latest_df.loc[
                        latest_df["ticker"] == ticker, "score"
                    ].values[0]
                )

                portfolio_rows.append({
                    "ticker": ticker,
                    "weight": float(weight),
                    "score": score
                })

            self.cache.set(cache_key, portfolio_rows, ttl=self.CACHE_TTL)
            self.breaker.record_success()

            return portfolio_rows

        except Exception as e:

            self.breaker.record_failure()
            PIPELINE_FAILURES.labels(stage="inference").inc()
            logger.exception("Batch inference failure: %s", str(e))
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()