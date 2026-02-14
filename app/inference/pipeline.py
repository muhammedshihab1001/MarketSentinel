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
    validate_feature_schema,
    get_schema_signature
)

from core.market.universe import MarketUniverse
from core.portfolio.distribution_forecaster import DistributionForecaster

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

    LATENCY_GUARD_SECONDS = float(
        os.getenv("HARD_LATENCY_LIMIT_SECONDS", "5.0")
    )

    CACHE_TTL = int(
        os.getenv("INFERENCE_CACHE_TTL_SECONDS", "900")
    )

    MAX_NULL_RATIO = float(
        os.getenv("MAX_FEATURE_NULL_RATIO", "0.02")
    )

    LOCK_TIMEOUT = 5

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

        self.distribution = DistributionForecaster()

        self.schema_sig = get_schema_signature()

        self._validate_models_loaded()

    ############################################################

    def _validate_models_loaded(self):

        model = self.models.xgb

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model is not a classifier.")

    ############################################################
    # DATA FRESHNESS (VERY IMPORTANT)
    ############################################################

    def _assert_data_freshness(self, price_df):

        latest_date = pd.to_datetime(price_df["date"].max(), utc=True)
        now = pd.Timestamp.utcnow()

        age_hours = (now - latest_date).total_seconds() / 3600

        if age_hours > 36:
            raise RuntimeError(
                f"Market data stale: {age_hours:.1f}h old."
            )

    ############################################################

    def _extract_features(self, latest_row):

        df = pd.DataFrame(
            [latest_row.loc[list(MODEL_FEATURES)].values],
            columns=MODEL_FEATURES
        )

        df = validate_feature_schema(df)

        null_ratio = df.isna().mean().mean()
        MISSING_FEATURE_RATIO.set(float(null_ratio))

        if null_ratio > self.MAX_NULL_RATIO:
            raise RuntimeError("Feature null ratio exceeded.")

        arr = df.to_numpy(dtype="float32")

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature vector detected.")

        return arr

    ############################################################

    def _safe_probability(self, prob):

        if not np.isfinite(prob):
            raise RuntimeError("Model produced invalid probability.")

        return float(np.clip(prob, 0.001, 0.999))

    ############################################################

    def _dataset_hash(self, latest_row):

        canonical = latest_row[list(MODEL_FEATURES)] \
            .round(10) \
            .to_json()

        return hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:12]

    ############################################################
    # MAIN RUN
    ############################################################

    def run(self, ticker="AAPL"):

        MarketUniverse.validate_subset([ticker])

        if not self.breaker.allow():
            PIPELINE_FAILURES.labels(stage="circuit_open").inc()
            raise RuntimeError("Inference circuit breaker open.")

        INFERENCE_IN_PROGRESS.inc()

        try:

            ####################################################
            # LOAD DATA
            ####################################################

            price_df = self.market_data.get_price_data(
                ticker=ticker,
                start_date="2018-01-01",
                end_date=pd.Timestamp.utcnow().date().isoformat()
            )

            if price_df is None or price_df.empty:
                raise RuntimeError("Market data unavailable.")

            self._assert_data_freshness(price_df)

            ####################################################
            # FEATURES
            ####################################################

            sentiment_df = None
            try:
                news = self.news_fetcher.fetch(f"{ticker} stock")
                scored = self.sentiment.analyze_dataframe(news)
                sentiment_df = self.sentiment.aggregate_daily_sentiment(scored)
            except Exception:
                logger.warning("Sentiment fallback used.")

            dataset = self.feature_store.get_features(
                price_df,
                sentiment_df,
                ticker=ticker,
                training=False
            )

            if dataset.empty:
                raise RuntimeError("Feature pipeline returned empty dataset.")

            latest = dataset.iloc[-1]
            features = self._extract_features(latest)

            ####################################################
            # CACHE (stampede-safe)
            ####################################################

            model_version = self.models.xgb_version
            dataset_hash = self._dataset_hash(latest)

            cache_key = self.cache.build_key({
                "ticker": ticker,
                "model_version": model_version,
                "dataset": dataset_hash,
                "schema": self.schema_sig
            })

            cached = self.cache.get(cache_key)

            if cached:
                CACHE_HITS.inc()
                return cached

            lock = self.cache.get_lock(
                cache_key,
                timeout=self.LOCK_TIMEOUT
            )

            if not lock.acquire(blocking=False):
                time.sleep(0.15)
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
                raise RuntimeError("Inference contention detected.")

            try:

                ####################################################
                # DRIFT CHECK
                ####################################################

                try:
                    drift = self.drift_detector.detect(
                        dataset[MODEL_FEATURES]
                    )

                    if drift.get("drift_detected"):
                        logger.warning(
                            "Drift detected | score=%.4f",
                            drift.get("drift_score", -1)
                        )

                except Exception:
                    logger.exception("Drift detector failure.")

                ####################################################
                # MODEL
                ####################################################

                t0 = time.time()

                preds = self.models.xgb.predict_proba(features)
                prob_up = self._safe_probability(preds[0][1])

                latency = time.time() - t0

                if latency > self.LATENCY_GUARD_SECONDS:
                    raise RuntimeError("Inference latency breach.")

                MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
                MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)
                PREDICTION_CLASS_PROBABILITY.set(prob_up)

                ####################################################
                # DISTRIBUTION
                ####################################################

                current_price = float(latest["close"])
                volatility = float(latest.get("volatility", 0.02))

                try:

                    recent_prices = price_df["close"].tail(60).tolist()
                    lstm_forecast = self.models.lstm_forecast(recent_prices)

                    surface = self.distribution.from_lstm(
                        current_price,
                        lstm_forecast
                    )

                except Exception:
                    logger.warning("Falling back to volatility distribution.")

                    surface = self.distribution.from_volatility(
                        current_price,
                        volatility,
                        prob_up
                    )

                ####################################################
                # DECISION — PASS FULL PRICE DF
                ####################################################

                decision = self.decision_engine.generate(
                    ticker=ticker,
                    price_df=price_df,
                    current_price=current_price,
                    forecast_up=surface["p90"],
                    forecast_down=surface["p10"],
                    prob_up=prob_up,
                    volatility=volatility,
                    regime=None
                )

                SIGNAL_DISTRIBUTION.labels(
                    signal=decision["signal"]
                ).inc()

                CONFIDENCE_SCORE.set(float(decision["confidence"]))

                response = {
                    "ticker": ticker,
                    "signal_today": decision["signal"],
                    "confidence": decision["confidence"],
                    "probability_up": prob_up,
                    "expected_value": decision.get("expected_value"),
                    "risk_score": decision.get("risk_score"),
                    "allocation": decision["allocation"],
                    "position_pct": decision["position_pct"]
                }

                self.cache.set(cache_key, response, ttl=self.CACHE_TTL)

                self.breaker.record_success()

                return response

            finally:
                try:
                    lock.release()
                except Exception:
                    pass

        except Exception as e:

            self.breaker.record_failure()
            PIPELINE_FAILURES.labels(stage="inference").inc()

            logger.exception("Inference failure: %s", str(e))
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()
