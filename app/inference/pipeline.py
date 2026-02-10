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

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    SIGNAL_DISTRIBUTION,
    FORECAST_HORIZON,
    CONFIDENCE_SCORE,
    MISSING_FEATURE_RATIO,
    FEATURE_MEAN,
    FEATURE_STD,
    FEATURE_MAX,
    FEATURE_MIN,
    PIPELINE_FAILURES,
    PREDICTION_CLASS_PROBABILITY
)


class InferencePipeline:

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

        self.feature_baselines = self._load_feature_baselines()

    # ---------------------------------------------------

    def _load_feature_baselines(self):

        try:

            latest_dir, _ = self.models._resolve_latest_dir(
                "artifacts/xgboost"
            )

            metadata_path = f"{latest_dir}/metadata.json"

            import json
            with open(metadata_path) as f:
                metadata = json.load(f)

            return metadata.get("feature_baselines", {})

        except Exception:
            return {}

    # ---------------------------------------------------

    def _detect_feature_drift(self, dataset):

        if not self.feature_baselines:
            return

        numeric_cols = dataset.select_dtypes(include="number").columns

        for col in numeric_cols:

            if col not in self.feature_baselines:
                continue

            baseline = self.feature_baselines[col]

            live_series = dataset[col].dropna()

            if len(live_series) == 0:
                continue

            live_mean = float(live_series.mean())
            live_std = float(live_series.std())
            live_max = float(live_series.max())
            live_min = float(live_series.min())

            FEATURE_MEAN.labels(feature=col).set(live_mean)
            FEATURE_STD.labels(feature=col).set(live_std)
            FEATURE_MAX.labels(feature=col).set(live_max)
            FEATURE_MIN.labels(feature=col).set(live_min)

            baseline_mean = baseline["mean"]
            baseline_std = baseline["std"] or 1e-6

            z_score = abs(live_mean - baseline_mean) / baseline_std

            if z_score > 3:
                PIPELINE_FAILURES.labels(stage="feature_drift").inc()

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
        """
        Prevent silent schema mismatch.
        """

        required = self.models.xgb.feature_names_in_

        missing = [f for f in required if f not in latest_row]

        if missing:
            raise RuntimeError(
                f"Feature mismatch detected. Missing: {missing}"
            )

        return latest_row[required].values.reshape(1, -1)

    # ---------------------------------------------------

    def run(
        self,
        ticker="AAPL",
        start_date=None,
        end_date=None,
        forecast_days=30
    ):

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

                # -----------------------------------
                # DATA
                # -----------------------------------

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
                    raise ValueError("Feature pipeline empty")

                missing_ratio = dataset.isnull().mean().mean()
                MISSING_FEATURE_RATIO.set(float(missing_ratio))

                self._detect_feature_drift(dataset)

                latest = dataset.iloc[-1]

                features = self._extract_features(latest)

                # -----------------------------------
                # XGBOOST
                # -----------------------------------

                t0 = time.time()

                prediction = self.models.xgb.predict(features)[0]
                prob_up = self.models.xgb.predict_proba(features)[0][1]

                MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
                MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(
                    time.time() - t0
                )

                # 🔥 probability histogram
                PREDICTION_CLASS_PROBABILITY.labels(
                    model="xgboost"
                ).observe(prob_up)

                predicted_return = prob_up - 0.5

                # -----------------------------------
                # LSTM
                # -----------------------------------

                recent_prices = price_df[["close"]].tail(60).values

                t0 = time.time()

                lstm_prices = self.models.lstm_forecast(recent_prices)
                lstm_prices = lstm_prices[:forecast_days]

                MODEL_INFERENCE_COUNT.labels(model="lstm").inc()
                MODEL_INFERENCE_LATENCY.labels(model="lstm").observe(
                    time.time() - t0
                )

                # -----------------------------------
                # PROPHET
                # -----------------------------------

                t0 = time.time()

                prophet_out = self.models.prophet_forecast()

                MODEL_INFERENCE_COUNT.labels(model="prophet").inc()
                MODEL_INFERENCE_LATENCY.labels(model="prophet").observe(
                    time.time() - t0
                )

                # -----------------------------------
                # DECISION (NEW STRUCTURED OUTPUT)
                # -----------------------------------

                decision = self.decision_engine.generate(
                    predicted_return=predicted_return,
                    sentiment=latest["avg_sentiment"],
                    rsi=latest["rsi"],
                    prob_up=prob_up,
                    volatility=latest["volatility"],
                    lstm_prices=lstm_prices,
                    prophet_trend=prophet_out["trend"]
                )

                SIGNAL_DISTRIBUTION.labels(
                    signal=decision["signal"]
                ).inc()

                CONFIDENCE_SCORE.set(float(decision["confidence"]))
                FORECAST_HORIZON.set(len(lstm_prices))

                response = {
                    "ticker": ticker,
                    "signal_today": decision["signal"],
                    "confidence": float(decision["confidence"]),
                    "probability_up": float(prob_up),

                    # 🔥 BIG upgrade
                    "recommended_allocation": decision["allocation"],
                    "position_size_pct": decision["position_pct"]
                }

                self.cache.set(cache_key, response, ttl=900)

                return response

        except Exception:

            PIPELINE_FAILURES.labels(stage="inference").inc()
            raise
