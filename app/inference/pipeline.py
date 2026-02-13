import time
import threading
import numpy as np
import pandas as pd
import logging
import os
import hashlib

from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_store import FeatureStore
from core.signals.signal_engine import DecisionEngine
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema
)

from core.market.universe import MarketUniverse

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
            logger.critical("Circuit breaker failure count=%s", self.failures)

    def record_success(self):

        with self._lock:
            self.failures = 0


############################################################
# INFERENCE PIPELINE
############################################################

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

    CACHE_TTL = int(
        os.getenv("INFERENCE_CACHE_TTL_SECONDS", "900")
    )

    DATA_FRESHNESS_HOURS = int(
        os.getenv("MAX_DATA_AGE_HOURS", "24")
    )

    LOCK_TIMEOUT = 3

    ############################################################

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

    ############################################################

    def _deadline(self, start):
        remaining = self.HARD_PIPELINE_TIMEOUT - (time.time() - start)
        if remaining <= 0:
            raise RuntimeError("Pipeline timeout exceeded.")
        return remaining

    ############################################################

    def _validate_models_loaded(self):

        if self.models.xgb is None:
            raise RuntimeError("XGBoost model not loaded.")

        if not hasattr(self.models.xgb, "predict_proba"):
            raise RuntimeError("Invalid XGBoost artifact.")

        if self.models.sarimax is None:
            raise RuntimeError("SARIMAX model not loaded.")

    ############################################################
    # NEW — CANDLE CLOSE GUARD
    ############################################################

    def _assert_candle_closed(self, df: pd.DataFrame):

        last_ts = pd.to_datetime(df["date"].iloc[-1], utc=True)
        now = pd.Timestamp.utcnow()

        if (now - last_ts).total_seconds() < 3600:
            raise RuntimeError(
                "Latest candle may still be forming — refusing inference."
            )

    ############################################################

    def _validate_data_freshness(self, df: pd.DataFrame):

        latest = pd.to_datetime(df["date"], utc=True)

        age_hours = (
            pd.Timestamp.utcnow() - latest.max()
        ).total_seconds() / 3600

        if age_hours > self.DATA_FRESHNESS_HOURS:
            logger.warning(
                "Market data stale (%.1fh old). Continuing inference.",
                age_hours
            )

    ############################################################
    # HARD FEATURE EXTRACTION
    ############################################################

    def _extract_features(self, latest_row):

        df = pd.DataFrame(
            [latest_row.loc[list(MODEL_FEATURES)].values],
            columns=MODEL_FEATURES
        )

        df = validate_feature_schema(df)

        arr = df.to_numpy(dtype="float32")

        if arr.shape[1] != len(MODEL_FEATURES):
            raise RuntimeError("Feature width mismatch.")

        return arr

    ############################################################

    def _safe_probability(self, prob):

        if not np.isfinite(prob):
            raise RuntimeError("Model produced invalid probability.")

        prob = float(np.clip(prob, 0.0001, 0.9999))

        if prob < 0.01 or prob > 0.99:
            raise RuntimeError("Probability collapse detected.")

        return prob

    ############################################################

    def _safe_sentiment_fetch(self, ticker):

        try:

            news = self.news_fetcher.fetch(f"{ticker} stock")

            scored = self.sentiment.analyze_dataframe(news)

            return self.sentiment.aggregate_daily_sentiment(scored)

        except Exception:
            logger.exception(
                "Sentiment pipeline failed — using neutral fallback."
            )

            return pd.DataFrame()

    ############################################################
    # MAIN RUN
    ############################################################

    def run(self, ticker="AAPL"):

        MarketUniverse.validate_subset([ticker])

        if not self.breaker.allow():
            PIPELINE_FAILURES.labels(stage="circuit_open").inc()
            raise RuntimeError("Inference circuit breaker open.")

        start_pipeline = time.time()
        INFERENCE_IN_PROGRESS.inc()

        try:

            model_version = self.models.xgb_version

            ####################################################
            # DATA
            ####################################################

            price_df = self.market_data.get_price_data(
                ticker=ticker,
                start_date="2018-01-01",
                end_date=pd.Timestamp.utcnow().date().isoformat()
            )

            self._validate_data_freshness(price_df)

            sentiment_df = self._safe_sentiment_fetch(ticker)

            dataset = self.feature_store.get_features(
                price_df,
                sentiment_df,
                ticker=ticker
            )

            if dataset.empty:
                raise RuntimeError("Feature pipeline returned empty dataset.")

            validate_feature_schema(dataset.loc[:, MODEL_FEATURES])

            self._assert_candle_closed(dataset)

            ####################################################
            # DATASET HASH FOR CACHE LINEAGE
            ####################################################

            dataset_hash = hashlib.sha256(
                dataset.iloc[-1].to_json().encode()
            ).hexdigest()[:12]

            payload = {
                "ticker": ticker,
                "model_version": model_version,
                "dataset": dataset_hash
            }

            cache_key = self.cache.build_key(payload)

            cached = self.cache.get(cache_key)

            if cached:
                CACHE_HITS.inc()
                return cached

            CACHE_MISSES.inc()

            lock = self.cache.get_lock(
                cache_key,
                timeout=self.LOCK_TIMEOUT
            )

            if not lock.acquire(blocking=False):

                cached = self.cache.get(cache_key)

                if cached:
                    return cached

                raise RuntimeError("Inference contention detected.")

            try:

                ################################################
                # DRIFT
                ################################################

                try:

                    drift = self.drift_detector.detect(
                        dataset[MODEL_FEATURES]
                    )

                    if drift.get("drift_detected"):
                        logger.warning(
                            "Inference drift snapshot | ticker=%s | score=%.4f",
                            ticker,
                            drift.get("drift_score", -1)
                        )

                        if self.FAIL_ON_DRIFT:
                            raise RuntimeError(
                                "Drift detected — inference halted."
                            )

                except Exception:
                    logger.exception(
                        "Drift detector failure — continuing inference."
                    )

                ################################################
                # FEATURE VECTOR
                ################################################

                latest = dataset.iloc[-1].copy()
                features = self._extract_features(latest)

                ################################################
                # PREDICT
                ################################################

                t0 = time.time()

                preds = self.models.xgb.predict_proba(features)

                if preds.ndim != 2 or preds.shape[1] < 2:
                    raise RuntimeError("predict_proba returned invalid shape.")

                prob_up = self._safe_probability(preds[0][1])

                latency = time.time() - t0

                if latency > self.LATENCY_GUARD_SECONDS:
                    logger.warning("Inference latency breach: %.2fs", latency)

                MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
                MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)

                PREDICTION_CLASS_PROBABILITY.set(prob_up)

                ################################################
                # DECISION ENGINE
                ################################################

                predicted_return = prob_up - 0.5

                decision = self.decision_engine.generate(
                    predicted_return=predicted_return,
                    sentiment=latest.get("avg_sentiment", 0),
                    rsi=latest.get("rsi", 50),
                    prob_up=prob_up,
                    volatility=latest.get("volatility", 0),
                    lstm_prices=None,
                    prophet_trend=None,
                    regime=None
                )

                SIGNAL_DISTRIBUTION.labels(
                    signal=decision["signal"]
                ).inc()

                CONFIDENCE_SCORE.set(float(decision["confidence"]))

                response = {
                    "ticker": ticker,
                    "signal_today": decision["signal"],
                    "confidence": float(decision["confidence"]),
                    "probability_up": prob_up,
                    "recommended_allocation": decision["allocation"],
                    "position_size_pct": decision["position_pct"]
                }

                self.cache.set(cache_key, response, ttl=self.CACHE_TTL)

                self.breaker.record_success()

                return response

            finally:
                try:
                    lock.release()
                except Exception:
                    pass

        except RuntimeError:
            self.breaker.record_failure()
            PIPELINE_FAILURES.labels(stage="inference").inc()
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()
