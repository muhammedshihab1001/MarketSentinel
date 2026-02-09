import datetime
import time
import numpy as np
import pandas as pd

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.signals.signal_engine import DecisionEngine
from core.scenario.scenario_engine import ScenarioEngine
from core.explainability.decision_explainer import DecisionExplainer

from app.inference.model_loader import ModelLoader

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

        self.fetcher = StockPriceFetcher()
        self.news_fetcher = NewsFetcher()
        self.sentiment = SentimentAnalyzer()
        self.models = ModelLoader()

        self.decision_engine = DecisionEngine()
        self.scenario_engine = ScenarioEngine()
        self.explainer = DecisionExplainer()

    # ---------------------------------------------------

    def run(self, ticker="AAPL"):

        end_date = datetime.date.today().isoformat()

        # -----------------------------------
        # FETCH DATA
        # -----------------------------------

        price_df = self.fetcher.fetch(
            ticker=ticker,
            start_date="2018-01-01",
            end_date=end_date
        )

        news_df = self.news_fetcher.fetch(
            f"{ticker} stock",
            max_items=50
        )

        scored = self.sentiment.analyze_dataframe(news_df)
        sentiment_df = self.sentiment.aggregate_daily_sentiment(scored)

        dataset = FeatureEngineer.build_feature_pipeline(
            price_df,
            sentiment_df
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

        start = time.time()

        prediction = self.models.xgb.predict(features)[0]
        prob_up = self.models.xgb.predict_proba(features)[0][1]

        MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
        MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(
            time.time() - start
        )

        predicted_return = prob_up - 0.5

        # -----------------------------------
        # LSTM
        # -----------------------------------

        recent_prices = price_df[["close"]].tail(60).values

        start = time.time()

        lstm_prices = self.models.lstm_forecast(recent_prices)

        MODEL_INFERENCE_COUNT.labels(model="lstm").inc()
        MODEL_INFERENCE_LATENCY.labels(model="lstm").observe(
            time.time() - start
        )

        # -----------------------------------
        # PROPHET
        # -----------------------------------

        start = time.time()

        prophet_out = self.models.prophet_forecast()

        MODEL_INFERENCE_COUNT.labels(model="prophet").inc()
        MODEL_INFERENCE_LATENCY.labels(model="prophet").observe(
            time.time() - start
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

        # -----------------------------------
        # BUILD FORECAST CURVE
        # -----------------------------------

        last_date = pd.to_datetime(price_df["date"].iloc[-1])

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(lstm_prices),
            freq="B"  # business days only
        )

        forecast_series = []

        last_close = price_df["close"].iloc[-1]

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

            forecast_series.append({
                "date": str(date.date()),
                "expected_price": float(price),
                "signal": step_signal
            })

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

        # -----------------------------------
        # RESPONSE
        # -----------------------------------

        return {

            "ticker": ticker,

            "signal_today": signal_today,
            "confidence": float(confidence),
            "probability_up": float(prob_up),

            "forecast": forecast_series,

            "forecast_start": str(future_dates[0].date()),
            "forecast_end": str(future_dates[-1].date()),
            "horizon_days": len(forecast_series),

            "scenarios": scenarios,
            "explanation": explanation
        }
