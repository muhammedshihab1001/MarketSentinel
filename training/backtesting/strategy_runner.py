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

from core.market.universe import MarketUniverse
from training.backtesting.backtest_engine import BacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class StrategyRunner:

    MIN_PROB_STD = 1e-5
    EPSILON = 1e-12

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.sentiment = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.regime_detector = MarketRegimeDetector()

    ###################################################
    # SAFETY
    ###################################################

    def _assert_prices(self, prices):

        if not np.isfinite(prices).all():
            raise RuntimeError("Non-finite prices detected.")

        if (prices <= 0).any():
            raise RuntimeError("Non-positive prices detected.")

    def _assert_temporal_integrity(self, df):

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

    def _validate_probabilities(self, probs):

        if not np.isfinite(probs).all():
            raise RuntimeError("Model produced non-finite probabilities.")

        if np.std(probs) < self.MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected.")

        if probs.mean() < 0.02 or probs.mean() > 0.98:
            raise RuntimeError("Degenerate classifier detected.")

    ###################################################
    # CROSS-SECTION NORMALIZATION (MATCH TRAINING)
    ###################################################

    def _cross_sectional_normalize(self, df):

        df = df.sort_values(["date", "ticker"]).copy()
        grouped = df.groupby("date")

        for col in MODEL_FEATURES:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std").fillna(1).clip(lower=1e-6)
            df[col] = (df[col] - mean) / std

        return df

    ###################################################
    # LOAD FULL UNIVERSE FEATURES (CRITICAL FIX)
    ###################################################

    def _build_cross_section_dataset(
        self,
        start_date,
        end_date
    ):

        universe = MarketUniverse.get_universe()

        datasets = []

        for ticker in universe:

            try:

                price_df = self.market_data.get_price_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )

                dataset = self.feature_store.get_features(
                    price_df,
                    sentiment_df=None,
                    ticker=ticker,
                    training=False
                )

                datasets.append(dataset)

            except Exception:
                continue

        if not datasets:
            raise RuntimeError("Universe feature build failed.")

        df = pd.concat(datasets, ignore_index=True)

        df = self._cross_sectional_normalize(df)

        return df

    ###################################################
    # MAIN
    ###################################################

    def run(
        self,
        model,
        ticker: str,
        start_date: str,
        end_date: str
    ):

        ################################################
        # BUILD FULL CROSS-SECTION DATASET
        ################################################

        full_df = self._build_cross_section_dataset(
            start_date,
            end_date
        )

        ################################################
        # EXTRACT TARGET TICKER
        ################################################

        dataset = full_df[full_df["ticker"] == ticker].copy()

        if dataset.empty:
            raise RuntimeError("Ticker missing after normalization.")

        ################################################
        # REGIME DETECTION
        ################################################

        dataset = self.regime_detector.detect(dataset)

        dataset = dataset.sort_values("date").reset_index(drop=True)

        self._assert_temporal_integrity(dataset)

        ################################################
        # SCHEMA VALIDATION
        ################################################

        validated = validate_feature_schema(
            dataset.loc[:, MODEL_FEATURES]
        )

        dataset.loc[:, MODEL_FEATURES] = validated

        ################################################
        # MODEL INFERENCE
        ################################################

        features = dataset.loc[:, MODEL_FEATURES].to_numpy(dtype=np.float32)

        probs = model.predict_proba(features)[:, 1]

        self._validate_probabilities(probs)

        probs = (
            pd.Series(probs)
            .shift(1)
            .fillna(0.5)
            .to_numpy()
        )

        ################################################
        # SIGNAL GENERATION
        ################################################

        decision_engine = DecisionEngine()
        backtest_engine = BacktestEngine()

        signals = []

        for i in range(len(dataset)):

            row = dataset.iloc[i]

            prob_up = float(probs[i])
            predicted_return = prob_up - 0.5

            signal_dict = decision_engine.generate(
                ticker=ticker,
                predicted_return=predicted_return,
                sentiment=float(row.get("avg_sentiment", 0.0)),
                rsi=float(row.get("rsi", 50.0)),
                prob_up=prob_up,
                volatility=float(row.get("volatility", 0.02)),
                lstm_prices=np.array([row["close"]]),
                macro_trend="NEUTRAL",
                regime=row.get("regime", "SIDEWAYS")
            )

            signals.append(signal_dict["signal"])

        ################################################
        # BACKTEST
        ################################################

        prices = dataset["close"].to_numpy(dtype=float)

        self._assert_prices(prices)

        results = backtest_engine.run(
            prices=prices,
            signals=signals
        )

        return results
