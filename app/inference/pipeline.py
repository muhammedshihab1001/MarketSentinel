import datetime
import numpy as np

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.signals.signal_engine import DecisionEngine
from core.scenario.scenario_engine import ScenarioEngine
from core.explainability.decision_explainer import DecisionExplainer

from app.inference.model_loader import ModelLoader


class InferencePipeline:
    """
    Institutional-style inference control plane.
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

        price_df = self.fetcher.fetch(
            ticker=ticker,
            start_date="2018-01-01",
            end_date=end_date
        )

        news_df = self.news_fetcher.fetch(f"{ticker} stock", max_items=50)

        scored = self.sentiment.analyze_dataframe(news_df)
        sentiment_df = self.sentiment.aggregate_daily_sentiment(scored)

        dataset = FeatureEngineer.build_feature_pipeline(
            price_df,
            sentiment_df
        )

        latest = dataset.iloc[-1]

        features = latest[self.models.xgb.feature_names_in_].values.reshape(1, -1)

        # -----------------------------
        # XGBoost Direction
        # -----------------------------

        prediction = self.models.xgb.predict(features)[0]
        prob_up = self.models.xgb.predict_proba(features)[0][1]

        predicted_return = prob_up - 0.5

        # -----------------------------
        # LSTM Forecast
        # -----------------------------

        recent_prices = price_df[["close"]].tail(60).values

        lstm_prices = self.models.lstm_forecast(recent_prices)

        # -----------------------------
        # Prophet
        # -----------------------------

        prophet_out = self.models.prophet_forecast()

        # -----------------------------
        # Decision
        # -----------------------------

        signal, confidence = self.decision_engine.generate(
            predicted_return=predicted_return,
            sentiment=latest["avg_sentiment"],
            rsi=latest["rsi"],
            prob_up=prob_up,
            volatility=latest["volatility"],
            lstm_prices=lstm_prices,
            prophet_trend=prophet_out["trend"]
        )

        # -----------------------------
        # Scenario Analysis
        # -----------------------------

        scenarios = self.scenario_engine.generate({
            "mean_forecast": np.mean(lstm_prices),
            "std_dev": np.std(lstm_prices)
        })

        # -----------------------------
        # Explanation
        # -----------------------------

        explanation = self.explainer.explain(
            prediction=int(prediction),
            prob_up=float(prob_up),
            sentiment=float(latest["avg_sentiment"]),
            volatility=float(latest["volatility"]),
            rsi=float(latest["rsi"])
        )

        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": float(confidence),
            "probability_up": float(prob_up),
            "scenarios": scenarios,
            "explanation": explanation
        }
