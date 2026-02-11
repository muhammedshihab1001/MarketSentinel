import datetime
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
from core.scenario.scenario_engine import ScenarioEngine
from core.explainability.decision_explainer import DecisionExplainer
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    SIGNAL_DISTRIBUTION,
    FORECAST_HORIZON,
    CONFIDENCE_SCORE,
    MISSING_FEATURE_RATIO,
    PIPELINE_FAILURES,
    PREDICTION_CLASS_PROBABILITY,
    CACHE_HITS,
    CACHE_MISSES,
    INFERENCE_IN_PROGRESS
)


logger = logging.getLogger("marketsentinel.pipeline")


# =====================================================
# CIRCUIT BREAKER
# =====================================================

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


# =====================================================

class InferencePipeline:

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
        self.scenario_engine = ScenarioEngine()
        self.explainer = DecisionExplainer()
        self.feature_store = FeatureStore()

        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.breaker = CircuitBreaker()

        self._validate_models_loaded()

    # ---------------------------------------------------

    def _validate_models_loaded(self):

        if self.models.xgb is None:
            raise RuntimeError("XGBoost model not loaded.")

        if not hasattr(self.models.xgb, "predict_proba"):
            raise RuntimeError("Invalid XGBoost artifact.")

    # ---------------------------------------------------
    # DATA FRESHNESS GUARD
    # ---------------------------------------------------

    def _validate_data_freshness(self, df: pd.DataFrame):

        if "date" not in df.columns:
            return

        latest = pd.to_datetime(df["date"]).max()

        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")

        now = pd.Timestamp.utcnow()

        age_hours = (now - latest).total_seconds() / 3600

        if age_hours > self.DATA_FRESHNESS_HOURS:
            raise RuntimeError(
                f"Market data stale ({age_hours:.1f}h old)."
            )

    # ---------------------------------------------------

    def _safe_sentiment(self, ticker):

        try:

            news_df = self.news_fetcher.fetch(
                f"{ticker} stock",
                max_items=50
            )

            scored = self.sentiment.analyze_dataframe(news_df)

            return self.sentiment.aggregate_daily_sentiment(scored)

        except Exception:

            PIPELINE_FAILURES.labels(stage="sentiment").inc()

            return pd.DataFrame({
                "date": [],
                "avg_sentiment": [],
                "news_count": [],
                "sentiment_std": []
            })

    # ---------------------------------------------------

    def _extract_features(self, latest_row):

        missing = [
            f for f in MODEL_FEATURES
            if f not in latest_row
        ]

        if missing:
            raise RuntimeError(
                f"Feature mismatch detected. Missing: {missing}"
            )

        return latest_row[MODEL_FEATURES].values.reshape(1, -1)

    # ---------------------------------------------------

    def _guard_latency(self, start, model):

        elapsed = time.time() - start

        if elapsed > self.LATENCY_GUARD_SECONDS:

            logger.critical(f"{model} latency breach.")

            PIPELINE_FAILURES.labels(
                stage=f"{model}_latency"
            ).inc()

            raise RuntimeError(
                f"{model} exceeded latency guard."
            )

        return elapsed

    # ---------------------------------------------------

    def run(
        self,
        ticker="AAPL",
        start_date=None,
        end_date=None,
        forecast_days=30
    ):

        if not self.breaker.allow():
            PIPELINE_FAILURES.labels(stage="circuit_open").inc()
            raise RuntimeError("Inference circuit breaker open.")

        INFERENCE_IN_PROGRESS.inc()

        try:

            today = pd.Timestamp.utcnow().date()

            start_date = start_date or today
            end_date = end_date or (
                start_date + datetime.timedelta(days=forecast_days)
            )

            payload = {
                "ticker": ticker,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "forecast_days": forecast_days
            }

            cache_key = self.cache.build_key(payload)

            cached = self.cache.get(cache_key)

            if cached:
                CACHE_HITS.inc()
                return cached

            CACHE_MISSES.inc()

            lock = self.cache.get_lock(cache_key)

            with lock:

                cached = self.cache.get(cache_key)
                if cached:
                    CACHE_HITS.inc()
                    return cached

                # ---------------- DATA ----------------

                price_df = self.market_data.get_price_data(
                    ticker=ticker,
                    start_date="2018-01-01",
                    end_date=today.isoformat()
                )

                self._validate_data_freshness(price_df)

                sentiment_df = self._safe_sentiment(ticker)

                dataset = self.feature_store.get_features(
                    price_df,
                    sentiment_df,
                    ticker=ticker
                )

                if dataset.empty:
                    raise RuntimeError(
                        "Feature pipeline returned empty dataset."
                    )

                null_ratio = float(dataset.isnull().mean().mean())

                MISSING_FEATURE_RATIO.set(null_ratio)

                if null_ratio > self.MAX_NULL_RATIO:
                    raise RuntimeError(
                        f"Feature null ratio unsafe: {null_ratio:.3f}"
                    )

                drift = self.drift_detector.detect(dataset)

                if drift["drift_detected"]:

                    logger.critical("Feature drift detected.")

                    if self.FAIL_ON_DRIFT:
                        raise RuntimeError(
                            "Drift detected — inference halted."
                        )

                latest = dataset.iloc[-1]
                features = self._extract_features(latest)

                # ---------------- XGBOOST ----------------

                t0 = time.time()

                prob_up = self.models.xgb.predict_proba(features)[0][1]

                latency = self._guard_latency(t0, "xgboost")

                MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
                MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)

                PREDICTION_CLASS_PROBABILITY.labels(
                    model="xgboost"
                ).observe(prob_up)

                # ---------------- SHADOW ----------------

                if self.models.shadow_xgb is not None:
                    try:

                        shadow_prob = self.models.shadow_xgb.predict_proba(features)[0][1]

                        delta = abs(prob_up - shadow_prob)

                        if delta > 0.15:
                            logger.warning(
                                f"Shadow divergence detected | delta={delta:.2f}"
                            )

                    except Exception:
                        logger.exception("Shadow inference failed.")

                predicted_return = prob_up - 0.5

                # ---------------- DECISION ----------------

                try:

                    decision = self.decision_engine.generate(
                        predicted_return=predicted_return,
                        sentiment=latest.get("avg_sentiment", 0),
                        rsi=latest.get("rsi", 50),
                        prob_up=prob_up,
                        volatility=latest.get("volatility", 0),
                        lstm_prices=None,
                        prophet_trend=None
                    )

                except Exception:

                    logger.exception("Decision engine failure.")
                    PIPELINE_FAILURES.labels(stage="decision_engine").inc()

                    decision = {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "allocation": 0.0,
                        "position_pct": 0.0
                    }

                SIGNAL_DISTRIBUTION.labels(
                    signal=decision["signal"]
                ).inc()

                CONFIDENCE_SCORE.set(float(decision["confidence"]))
                FORECAST_HORIZON.set(0)

                response = {
                    "ticker": ticker,
                    "signal_today": decision["signal"],
                    "confidence": float(decision["confidence"]),
                    "probability_up": float(prob_up),
                    "recommended_allocation": decision["allocation"],
                    "position_size_pct": decision["position_pct"]
                }

                self.cache.set(cache_key, response, ttl=900)

                return response

        except Exception:

            self.breaker.record_failure()
            PIPELINE_FAILURES.labels(stage="inference").inc()

            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()
