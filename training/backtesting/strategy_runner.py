import numpy as np

from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.sentiment.sentiment import SentimentAnalyzer
from core.data.news_fetcher import NewsFetcher
from core.signals.signal_engine import DecisionEngine

from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)

from training.backtesting.backtest_engine import BacktestEngine


class StrategyRunner:
    """
    Institutional strategy orchestration layer.

    Guarantees:
    ✔ zero lookahead
    ✔ schema-locked
    ✔ neutral sentiment fallback
    ✔ deterministic signals
    ✔ no engine state carryover
    """

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.sentiment = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher()

    # ---------------------------------------------------

    def run(
        self,
        model,
        ticker: str,
        start_date: str,
        end_date: str
    ):

        ############################################
        # FETCH PRICE
        ############################################

        price_df = self.market_data.get_price_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        ############################################
        # NEWS — NEVER BLOCK
        ############################################

        sentiment_df = None

        try:

            news_df = self.news_fetcher.fetch(
                f"{ticker} stock",
                max_items=200
            )

            if news_df is not None and not news_df.empty:

                scored = self.sentiment.analyze_dataframe(news_df)
                sentiment_df = self.sentiment.aggregate_daily_sentiment(scored)

        except Exception:
            pass  # FeatureEngineer handles neutral fallback

        ############################################
        # FEATURES
        ############################################

        dataset = self.feature_store.get_features(
            price_df,
            sentiment_df,
            ticker=ticker,
            training=True
        )

        if dataset.empty:
            raise RuntimeError("Feature pipeline returned empty dataset")

        ############################################
        # SCHEMA LOCK
        ############################################

        validate_feature_schema(dataset.loc[:, MODEL_FEATURES])

        ############################################
        # STRICT NO LOOKAHEAD
        ############################################

        dataset = dataset.sort_values("date").reset_index(drop=True)

        features = dataset.loc[:, MODEL_FEATURES].to_numpy(dtype=np.float32)

        probs = model.predict_proba(features)[:, 1]

        # SHIFT SIGNALS → trade next bar
        probs = np.roll(probs, 1)
        probs[0] = 0.5

        ############################################
        # FRESH ENGINES (NO STATE LEAK)
        ############################################

        decision_engine = DecisionEngine()
        backtest_engine = BacktestEngine()

        signals = []

        for i in range(len(dataset)):

            row = dataset.iloc[i]

            prob_up = float(probs[i])

            predicted_return = prob_up - 0.5

            signal_dict = decision_engine.generate(
                predicted_return=predicted_return,
                sentiment=float(row["avg_sentiment"]),
                rsi=float(row["rsi"]),
                prob_up=prob_up,
                volatility=float(row["volatility"]),
                lstm_prices=np.array([row["close"]]),
                macro_trend="NEUTRAL",
                regime=None
            )

            signals.append(signal_dict["signal"])

        prices = dataset["close"].to_numpy(dtype=float)

        ############################################
        # BACKTEST
        ############################################

        results = backtest_engine.run(
            prices=prices,
            signals=signals
        )

        return results
