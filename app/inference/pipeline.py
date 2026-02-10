import datetime
import time
import threading
import numpy as np
import pandas as pd
import logging

from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_store import FeatureStore
from core.signals.signal_engine import DecisionEngine
from core.scenario.scenario_engine import ScenarioEngine
from core.explainability.decision_explainer import DecisionExplainer
from core.monitoring.drift_detector import DriftDetector

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
# THREAD SAFE CIRCUIT BREAKER
# =====================================================

class CircuitBreaker:

    def __init__(self, threshold=5, cooldown=60):
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
                self.failures = 0
                return True

            return False

    def record_failure(self):

        with self._lock:
            self.failures += 1
            self.last_failure = time.time()


# =====================================================

class InferencePipeline:

    LATENCY_GUARD_SECONDS = 5.0

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

        required = getattr(
            self.models.xgb,
            "feature_names_in_",
            list(latest_row.index)
        )

        missing = [f for f in required if f not in latest_row]

        if missing:
            raise RuntimeError(
                f"Feature mismatch detected. Missing: {missing}"
            )

        return latest_row[required].values.reshape(1, -1)

    # ---------------------------------------------------

    def _guard_latency(self, start, model):

        elapsed = time.time() - start

        if elapsed > self.LATENCY_GUARD_SECONDS:
            logger.warning(f"{model} exceeded latency guard.")
            PIPELINE_FAILURES.labels(stage=f"{model}_latency").inc()

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

            today = datetime.date.today()

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

            # 🔥 FIXED — async lock removed from cache earlier
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

                sentiment_df = self._safe_sentiment(ticker)

                dataset = self.feature_store.get_features(
                    price_df,
                    sentiment_df,
                    ticker=ticker
                )

                if dataset.empty:
                    raise RuntimeError("Feature pipeline returned empty dataset.")

                MISSING_FEATURE_RATIO.set(
                    float(dataset.isnull().mean().mean())
                )

                drift = self.drift_detector.detect(dataset)

                if drift["drift_detected"]:
                    logger.warning("Feature drift detected.")

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

                # 🔥 SHADOW EXECUTION
                shadow_prob = None

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
