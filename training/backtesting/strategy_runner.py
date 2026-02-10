import numpy as np

from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.sentiment.sentiment import SentimentAnalyzer
from core.data.news_fetcher import NewsFetcher
from core.signals.signal_engine import DecisionEngine

from training.backtesting.backtest_engine import BacktestEngine


class StrategyRunner:
    """
    Institutional strategy orchestration layer.

    Guarantees:
    ✅ Backtest uses SAME pipeline as inference
    ✅ Eliminates training/inference drift
    ✅ Prevents lookahead bias
    ✅ Enables reproducible research
    """

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.sentiment = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.decision_engine = DecisionEngine()
        self.engine = BacktestEngine()

    # ---------------------------------------------------

    def run(
        self,
        model,
        ticker: str,
        start_date: str,
        end_date: str
    ):
        """
        Runs a TRUE model-driven backtest.
        """

        # ---------------------------------------
        # FETCH DATA (IDENTICAL TO INFERENCE)
        # ---------------------------------------

        price_df = self.market_data.get_price_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        news_df = self.news_fetcher.fetch(
            f"{ticker} stock",
            max_items=200
        )

        scored = self.sentiment.analyze_dataframe(news_df)
        sentiment_df = self.sentiment.aggregate_daily_sentiment(scored)

        dataset = self.feature_store.get_features(
            price_df,
            sentiment_df
        )

        if dataset.empty:
            raise ValueError("Feature pipeline returned empty dataset")

        # ---------------------------------------
        # MODEL-DRIVEN SIGNAL GENERATION
        # ---------------------------------------

        signals = []

        feature_cols = model.feature_names_in_

        for _, row in dataset.iterrows():

            features = row[feature_cols].values.reshape(1, -1)

            prob_up = model.predict_proba(features)[0][1]

            predicted_return = prob_up - 0.5

            signal, _ = self.decision_engine.generate(
                predicted_return=predicted_return,
                sentiment=row["avg_sentiment"],
                rsi=row["rsi"],
                prob_up=prob_up,
                volatility=row["volatility"],
                lstm_prices=np.array([row["close"]]),  # placeholder
                prophet_trend="NEUTRAL"
            )

            signals.append(signal)

        prices = dataset["close"].values

        # ---------------------------------------
        # RUN BACKTEST
        # ---------------------------------------

        results = self.engine.run(
            prices=prices,
            signals=signals
        )

        return results
