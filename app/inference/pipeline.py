import datetime
import time
import numpy as np
import pandas as pd

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
    PREDICTION_CLASS_PROBABILITY
)


class CircuitBreaker:

    def __init__(self, threshold=5, cooldown=60):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure = None

    def allow(self):

        if self.failures < self.threshold:
            return True

        if (time.time() - self.last_failure) > self.cooldown:
            self.failures = 0
            return True

        return False

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()


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

        # production safety
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

            # controlled fallback → neutral sentiment
            return pd.DataFrame({
                "date": [],
                "avg_sentiment": [],
                "news_count": [],
                "sentiment_std": []
            })

    # ---------------------------------------------------

    def _extract_features(self, latest_row):

        required = self.models.xgb.feature_names_in_

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
            PIPELINE_FAILURES.labels(stage=f"{model}_latency").inc()
            raise RuntimeError(f"{model} exceeded latency guard.")

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
                return cached

            lock = self.cache.get_lock(cache_key)

            with lock:

                cached = self.cache.get(cache_key)
                if cached:
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

                missing_ratio = dataset.isnull().mean().mean()
                MISSING_FEATURE_RATIO.set(float(missing_ratio))

                # centralized drift system
                drift = self.drift_detector.detect(dataset)

                if drift["drift_detected"]:
                    PIPELINE_FAILURES.labels(stage="drift_detected").inc()

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

                predicted_return = prob_up - 0.5

                # ---------------- LSTM (fallback safe) ----------------

                lstm_prices = None

                try:

                    recent_prices = price_df[["close"]].tail(60).values

                    t0 = time.time()

                    lstm_prices = self.models.lstm_forecast(recent_prices)
                    lstm_prices = lstm_prices[:forecast_days]

                    latency = self._guard_latency(t0, "lstm")

                    MODEL_INFERENCE_COUNT.labels(model="lstm").inc()
                    MODEL_INFERENCE_LATENCY.labels(model="lstm").observe(latency)

                except Exception:
                    PIPELINE_FAILURES.labels(stage="lstm_failure").inc()

                # ---------------- PROPHET (fallback safe) ----------------

                prophet_trend = None

                try:

                    t0 = time.time()

                    prophet_out = self.models.prophet_forecast()
                    prophet_trend = prophet_out["trend"]

                    latency = self._guard_latency(t0, "prophet")

                    MODEL_INFERENCE_COUNT.labels(model="prophet").inc()
                    MODEL_INFERENCE_LATENCY.labels(model="prophet").observe(latency)

                except Exception:
                    PIPELINE_FAILURES.labels(stage="prophet_failure").inc()

                # ---------------- DECISION ----------------

                decision = self.decision_engine.generate(
                    predicted_return=predicted_return,
                    sentiment=latest.get("avg_sentiment", 0),
                    rsi=latest.get("rsi", 50),
                    prob_up=prob_up,
                    volatility=latest.get("volatility", 0),
                    lstm_prices=lstm_prices,
                    prophet_trend=prophet_trend
                )

                SIGNAL_DISTRIBUTION.labels(
                    signal=decision["signal"]
                ).inc()

                CONFIDENCE_SCORE.set(float(decision["confidence"]))
                FORECAST_HORIZON.set(
                    len(lstm_prices) if lstm_prices is not None else 0
                )

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

        except Exception as e:

            self.breaker.record_failure()

            PIPELINE_FAILURES.labels(stage="inference").inc()

            raise
