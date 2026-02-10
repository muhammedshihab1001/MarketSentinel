import datetime
import time
import numpy as np
import pandas as pd

from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.signals.signal_engine import DecisionEngine
from core.scenario.scenario_engine import ScenarioEngine
from core.explainability.decision_explainer import DecisionExplainer
from core.features.feature_store import FeatureStore

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    SIGNAL_DISTRIBUTION,
    FORECAST_HORIZON,
    CONFIDENCE_SCORE,
    MISSING_FEATURE_RATIO
)


class InferencePipeline:
    """
    Institutional-grade inference control plane.
    """

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

    # ---------------------------------------------------

    def run(
        self,
        ticker="AAPL",
        start_date=None,
        end_date=None,
        forecast_days=30
    ):

        today = datetime.date.today()

        # ✅ Normalize BEFORE cache
        start_date = start_date or today
        end_date = end_date or (
            start_date + datetime.timedelta(days=forecast_days)
        )

        # ==========================================
        # SAFE CACHE KEY
        # ==========================================

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

        # -----------------------------------
        # FETCH DATA
        # -----------------------------------

        price_df = self.market_data.get_price_data(
            ticker=ticker,
            start_date="2018-01-01",
            end_date=today.isoformat()
        )

        news_df = self.news_fetcher.fetch(
            f"{ticker} stock",
            max_items=50
        )

        scored = self.sentiment.analyze_dataframe(news_df)
        sentiment_df = self.sentiment.aggregate_daily_sentiment(scored)

        dataset = self.feature_store.get_features(
            price_df,
            sentiment_df
        )


        # ✅ Safety check
        if dataset.empty:
            raise ValueError(
                "Feature pipeline returned empty dataset."
            )

        # -----------------------------------
        # DATA QUALITY
        # -----------------------------------

        missing_ratio = dataset.isnull().mean().mean()
        MISSING_FEATURE_RATIO.set(float(missing_ratio))

        latest = dataset.iloc[-1]

        features = latest[
            self.models.xgb.feature_names_in_
        ].values.reshape(1, -1)

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
        # TODAY DECISION
        # -----------------------------------

        signal_today, confidence = self.decision_engine.generate(
            predicted_return=predicted_return,
            sentiment=latest["avg_sentiment"],
            rsi=latest["rsi"],
            prob_up=prob_up,
            volatility=latest["volatility"],
            lstm_prices=lstm_prices,
            prophet_trend=prophet_out["trend"]
        )

        SIGNAL_DISTRIBUTION.labels(signal=signal_today).inc()
        CONFIDENCE_SCORE.set(float(confidence))
        FORECAST_HORIZON.set(len(lstm_prices))

        # ==========================================
        # TIMELINE (Historical + Forecast)
        # ==========================================

        timeline = []

        hist_df = price_df.tail(120)

        for _, row in hist_df.iterrows():
            timeline.append({
                "date": str(pd.to_datetime(row["date"]).date()),
                "price": float(row["close"]),
                "type": "historical"
            })

        last_date = pd.to_datetime(price_df["date"].iloc[-1])

        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(lstm_prices)
        )

        last_close = price_df["close"].iloc[-1]

        forecast_series = []

        for date, price in zip(future_dates, lstm_prices):

            step_return = (price - last_close) / last_close

            step_signal, _ = self.decision_engine.generate(
                predicted_return=step_return,
                sentiment=latest["avg_sentiment"],
                rsi=latest["rsi"],
                prob_up=prob_up,
                volatility=latest["volatility"],
                lstm_prices=lstm_prices,
                prophet_trend=prophet_out["trend"]
            )

            point = {
                "date": str(date.date()),
                "price": float(price),
                "type": "forecast",
                "signal": step_signal,
                "expected_return_pct": round(step_return * 100, 2)
            }

            timeline.append(point)
            forecast_series.append(point)

        # -----------------------------------
        # SCENARIOS
        # -----------------------------------

        scenarios = self.scenario_engine.generate({
            "mean_forecast": float(np.mean(lstm_prices)),
            "std_dev": float(np.std(lstm_prices))
        })

        # -----------------------------------
        # EXPLANATION
        # -----------------------------------

        explanation = self.explainer.explain(
            prediction=int(prediction),
            prob_up=float(prob_up),
            sentiment=float(latest["avg_sentiment"]),
            volatility=float(latest["volatility"]),
            rsi=float(latest["rsi"])
        )

        response = {
            "ticker": ticker,
            "signal_today": signal_today,
            "confidence": float(confidence),
            "probability_up": float(prob_up),
            "timeline": timeline,
            "forecast_start": str(future_dates[0].date()),
            "forecast_end": str(future_dates[-1].date()),
            "horizon_days": len(forecast_series),
            "scenarios": scenarios,
            "explanation": explanation
        }

        # ==========================================
        # CACHE WRITE
        # ==========================================

        self.cache.set(cache_key, response, ttl=900)

        return response
