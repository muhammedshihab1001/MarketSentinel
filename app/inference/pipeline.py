import time
import threading
import numpy as np
import pandas as pd
import logging
import os

from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_store import FeatureStore
from core.signals.signal_engine import DecisionEngine
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    SIGNAL_DISTRIBUTION,
    CONFIDENCE_SCORE,
    MISSING_FEATURE_RATIO,
    PIPELINE_FAILURES,
    PREDICTION_CLASS_PROBABILITY,
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

            if (time.time() - self.last_failure) > self.cooldown:
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

    HARD_PIPELINE_TIMEOUT = float(
        os.getenv("PIPELINE_TIMEOUT_SECONDS", "12")
    )

    LATENCY_GUARD_SECONDS = float(
        os.getenv("HARD_LATENCY_LIMIT_SECONDS", "5.0")
    )

    MAX_NULL_RATIO = float(
        os.getenv("MAX_FEATURE_NULL_RATIO", "0.02")
    )

    FAIL_ON_DRIFT = os.getenv("FAIL_ON_DRIFT", "false").lower() == "true"

    DATA_FRESHNESS_HOURS = int(
        os.getenv("MAX_DATA_AGE_HOURS", "24")
    )

    def __init__(self):

        self.market_data = MarketDataService()
        self.news_fetcher = NewsFetcher()
        self.sentiment = SentimentAnalyzer()
        self.models = ModelLoader()

        self.decision_engine = DecisionEngine()
        self.feature_store = FeatureStore()

        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.breaker = CircuitBreaker()

        self._validate_models_loaded()

    def _validate_models_loaded(self):

        if self.models.xgb is None:
            raise RuntimeError("XGBoost model not loaded.")

        if not hasattr(self.models.xgb, "predict_proba"):
            raise RuntimeError("Invalid XGBoost artifact.")

    def _validate_data_freshness(self, df: pd.DataFrame):

        latest = pd.to_datetime(df["date"]).max()

        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")

        age_hours = (
            pd.Timestamp.utcnow() - latest
        ).total_seconds() / 3600

        if age_hours > self.DATA_FRESHNESS_HOURS:
            logger.warning(
                "Market data stale (%.1fh old). Continuing inference.",
                age_hours
            )

    def _extract_features(self, latest_row):

        missing = [
            f for f in MODEL_FEATURES
            if f not in latest_row
        ]

        if missing:
            raise RuntimeError(
                f"Feature mismatch detected. Missing: {missing}"
            )

        vector = latest_row.loc[list(MODEL_FEATURES)].astype("float32")

        null_ratio = float(np.isnan(vector).mean())

        MISSING_FEATURE_RATIO.set(null_ratio)

        if null_ratio > self.MAX_NULL_RATIO:
            logger.warning(
                "High null feature ratio detected: %.3f",
                null_ratio
            )

        return vector.values.reshape(1, -1)

    def _guard_latency(self, start, model):

        elapsed = time.time() - start

        if elapsed > self.LATENCY_GUARD_SECONDS:

            PIPELINE_FAILURES.labels(
                stage=f"{model}_latency"
            ).inc()

            logger.warning(
                "%s exceeded latency guard (%.2fs)",
                model,
                elapsed
            )

        return elapsed

    def run(self, ticker="AAPL"):

        if not self.breaker.allow():
            PIPELINE_FAILURES.labels(stage="circuit_open").inc()
            raise RuntimeError("Inference circuit breaker open.")

        start_pipeline = time.time()
        INFERENCE_IN_PROGRESS.inc()

        try:

            payload = {
                "ticker": ticker
            }

            cache_key = self.cache.build_key(payload)

            cached = self.cache.get(cache_key)

            if cached:
                CACHE_HITS.inc()
                return cached

            CACHE_MISSES.inc()

            lock = self.cache.get_lock(cache_key, timeout=10)

            with lock:

                price_df = self.market_data.get_price_data(
                    ticker=ticker,
                    start_date="2018-01-01",
                    end_date=pd.Timestamp.utcnow().date().isoformat()
                )

                self._validate_data_freshness(price_df)

                sentiment_df = self.sentiment.aggregate_daily_sentiment(
                    self.sentiment.analyze_dataframe(
                        self.news_fetcher.fetch(f"{ticker} stock")
                    )
                )

                dataset = self.feature_store.get_features(
                    price_df,
                    sentiment_df,
                    ticker=ticker
                )

                drift = self.drift_detector.detect(
                    dataset[MODEL_FEATURES]
                )

                if drift["drift_detected"] and self.FAIL_ON_DRIFT:
                    raise RuntimeError("Drift detected — inference halted.")

                latest = dataset.iloc[-1]
                features = self._extract_features(latest)

                t0 = time.time()

                prob_up = self.models.xgb.predict_proba(features)[0][1]

                latency = self._guard_latency(t0, "xgboost")

                MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
                MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)

                PREDICTION_CLASS_PROBABILITY.set(float(prob_up))

                predicted_return = prob_up - 0.5

                decision = self.decision_engine.generate(
                    predicted_return=predicted_return,
                    sentiment=latest.get("avg_sentiment", 0),
                    rsi=latest.get("rsi", 50),
                    prob_up=prob_up,
                    volatility=latest.get("volatility", 0),
                    lstm_prices=None,
                )

                SIGNAL_DISTRIBUTION.labels(
                    signal=decision["signal"]
                ).inc()

                CONFIDENCE_SCORE.set(float(decision["confidence"]))

                response = {
                    "ticker": ticker,
                    "signal_today": decision["signal"],
                    "confidence": float(decision["confidence"]),
                    "probability_up": float(prob_up),
                    "recommended_allocation": decision["allocation"],
                    "position_size_pct": decision["position_pct"]
                }

                self.cache.set(cache_key, response, ttl=900)

                if (time.time() - start_pipeline) > self.HARD_PIPELINE_TIMEOUT:
                    logger.warning("Pipeline exceeded hard timeout.")

                self.breaker.record_success()

                return response

        except Exception:

            self.breaker.record_failure()
            PIPELINE_FAILURES.labels(stage="inference").inc()

            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()
