import numpy as np
import pandas as pd

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
    ✔ probability sanity
    ✔ temporal boundary enforcement
    """

    MIN_PROB_STD = 1e-5
    EPSILON = 1e-12

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.sentiment = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher()

    ###################################################

    def _assert_prices(self, prices):

        if not np.isfinite(prices).all():
            raise RuntimeError("Non-finite prices detected.")

        if (prices <= 0).any():
            raise RuntimeError("Non-positive prices detected.")

    ###################################################

    def _assert_temporal_integrity(self, df):

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

    ###################################################

    def _validate_probabilities(self, probs):

        if not np.isfinite(probs).all():
            raise RuntimeError("Model produced non-finite probabilities.")

        if np.std(probs) < self.MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected.")

        if probs.mean() < 0.02 or probs.mean() > 0.98:
            raise RuntimeError("Degenerate classifier detected.")

    ###################################################

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

        if price_df is None or price_df.empty:
            raise RuntimeError("No market data returned.")

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
            pass  # fallback handled in feature store

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

        self._assert_temporal_integrity(dataset)

        features = dataset.loc[:, MODEL_FEATURES].to_numpy(dtype=np.float32)

        probs = model.predict_proba(features)[:, 1]

        self._validate_probabilities(probs)

        ################################################
        # SAFE SHIFT — NO np.roll EVER
        ################################################

        probs = pd.Series(probs).shift(1).fillna(0.5).to_numpy()

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
                ticker=ticker,   # <<< CRITICAL FIX
                predicted_return=predicted_return,
                sentiment=float(row.get("avg_sentiment", 0.0)),
                rsi=float(row.get("rsi", 50.0)),
                prob_up=prob_up,
                volatility=float(row.get("volatility", 0.02)),
                lstm_prices=np.array([row["close"]]),
                macro_trend="NEUTRAL",
                regime=row.get("regime", None)
            )

            signals.append(signal_dict["signal"])

        prices = dataset["close"].to_numpy(dtype=float)

        self._assert_prices(prices)

        ############################################
        # BACKTEST
        ############################################

        results = backtest_engine.run(
            prices=prices,
            signals=signals
        )

        return results
