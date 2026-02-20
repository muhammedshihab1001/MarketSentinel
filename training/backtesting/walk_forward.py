import numpy as np
import pandas as pd
import logging

from core.schema.feature_schema import (
    MODEL_FEATURES,
    DTYPE,
    FEATURE_LIMITS
)

from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector
from core.signals.signal_engine import DecisionEngine

logger = logging.getLogger("marketsentinel.walkforward")


class WalkForwardValidator:

    MIN_TRADES_PER_WINDOW = 5
    MIN_WINDOWS = 6
    MIN_TEST_DAYS = 20

    MIN_ASSETS_PER_DAY = 3
    MIN_FEATURE_VARIANCE = 1e-8

    DRIFT_WARN_Z = 10.0
    MIN_PROB_STD = 1e-5

    ########################################################

    def __init__(
        self,
        model_trainer,
        window_size=252,
        step_size=63,
        embargo_days=14
    ):
        self.model_trainer = model_trainer

        self.window_size = window_size
        self.step_size = step_size
        self.embargo_days = embargo_days

        self.engine = PortfolioBacktestEngine()
        self.regime_detector = MarketRegimeDetector()
        self.decision_engine = DecisionEngine()

    ########################################################

    def _apply_embargo(self, train_df, test_start):

        embargo_cut = pd.Timestamp(test_start) - pd.Timedelta(days=self.embargo_days)
        train_df = train_df[train_df["date"] < embargo_cut]

        if len(train_df) < int(self.window_size * 0.70):
            raise RuntimeError("Embargo removed too much training data.")

        return train_df

    ########################################################

    def _validate_training_frame(self, df):

        features = df.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=float)

        if not np.isfinite(features).all():
            raise RuntimeError("Non-finite values detected in training features.")

        variances = np.var(features, axis=0)

        if np.min(variances) < self.MIN_FEATURE_VARIANCE:
            raise RuntimeError("Feature variance collapsed.")

        if df["target"].nunique() < 2:
            raise RuntimeError("Training labels collapsed.")

    ########################################################

    def _validate_inference_slice(self, df_slice: pd.DataFrame) -> np.ndarray:

        if df_slice.empty:
            raise RuntimeError("Empty inference slice.")

        if set(df_slice.columns) != set(MODEL_FEATURES):
            raise RuntimeError("Inference slice feature mismatch.")

        df_slice = df_slice.loc[:, list(MODEL_FEATURES)].astype(DTYPE, copy=True)

        df_slice.replace([np.inf, -np.inf], np.nan, inplace=True)

        if df_slice.isna().any().any():
            raise RuntimeError("NaN detected in inference slice.")

        for col in MODEL_FEATURES:
            if col in FEATURE_LIMITS:
                lo, hi = FEATURE_LIMITS[col]
                if (df_slice[col] < lo).any() or (df_slice[col] > hi).any():
                    raise RuntimeError(f"Inference feature limit breach: {col}")

        return df_slice.to_numpy(dtype=DTYPE)

    ########################################################

    def _validate_probabilities(self, probs):

        if not np.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities.")

        if np.std(probs) < self.MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected.")

    ########################################################

    def _distribution_guard(self, train_df, test_df):

        train_mu = train_df[list(MODEL_FEATURES)].mean()
        train_std = train_df[list(MODEL_FEATURES)].std(ddof=0) + 1e-9

        z = np.abs((test_df[list(MODEL_FEATURES)] - train_mu) / train_std)
        max_z = float(np.nanmax(z.to_numpy()))

        if max_z > self.DRIFT_WARN_Z:
            logger.warning("Feature drift detected | max_z=%.2f", max_z)

    ########################################################

    def run(self, df: pd.DataFrame):

        logger.info("Walk-forward validation started.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        df = self.regime_detector.detect(df)

        unique_dates = pd.to_datetime(
            df["date"].drop_duplicates()
        ).sort_values()

        results = []
        equity_curve = []

        initial_capital = 10_000.0
        start_idx = self.window_size
        window_id = 1

        while start_idx < len(unique_dates) - 1:

            logger.info("Running WF window #%s", window_id)

            train_dates = unique_dates.iloc[start_idx - self.window_size:start_idx]
            test_dates = unique_dates.iloc[start_idx:start_idx + self.step_size]

            if len(test_dates) < self.MIN_TEST_DAYS:
                break

            train_df = df[
                (df["date"] >= train_dates.iloc[0]) &
                (df["date"] <= train_dates.iloc[-1])
            ].copy()

            train_df = self._apply_embargo(train_df, test_dates.iloc[0])

            test_df = df[
                (df["date"] >= test_dates.iloc[0]) &
                (df["date"] <= test_dates.iloc[-1])
            ].copy()

            self._validate_training_frame(train_df)
            self._distribution_guard(train_df, test_df)

            model = self.model_trainer(train_df)

            grouped_prices = {}
            grouped_signals = {}
            trade_counter = 0

            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(len(test_dates_sorted) - 1):

                signal_date = test_dates_sorted[i]
                execution_date = test_dates_sorted[i + 1]

                signal_slice = test_df[test_df["date"] == signal_date]
                exec_slice = test_df[test_df["date"] == execution_date]

                if signal_slice.empty or exec_slice.empty:
                    continue

                features = self._validate_inference_slice(
                    signal_slice.loc[:, list(MODEL_FEATURES)]
                )

                probs = model.predict_proba(features)[:, 1]

                self._validate_probabilities(probs)

                daily_signals = {}

                for row, prob in zip(
                    signal_slice.itertuples(index=False),
                    probs
                ):

                    decision = self.decision_engine.generate(
                        ticker=row.ticker,
                        predicted_return=float(prob - 0.5),
                        prob_up=float(prob),
                        volatility=float(getattr(row, "volatility", 0.02)),
                        regime=getattr(row, "regime", "SIDEWAYS"),
                        price_df=test_df[test_df["ticker"] == row.ticker]
                    )

                    if decision["signal"] in {"BUY", "SELL"}:
                        daily_signals[row.ticker] = decision["signal"]

                if len(daily_signals) < self.MIN_ASSETS_PER_DAY:
                    continue

                prices = {
                    t: float(p)
                    for t, p in zip(exec_slice["ticker"], exec_slice["close"])
                    if pd.notna(p) and np.isfinite(p) and p > 0
                }

                filtered = {
                    t: daily_signals[t]
                    for t in daily_signals
                    if t in prices
                }

                if not filtered:
                    continue

                grouped_prices[execution_date] = {
                    t: prices[t] for t in filtered
                }

                grouped_signals[execution_date] = filtered

                trade_counter += len(filtered)

            if trade_counter < self.MIN_TRADES_PER_WINDOW:
                start_idx += self.step_size
                window_id += 1
                continue

            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=initial_capital
            )

            curve = np.array(metrics["equity_curve"], dtype=float)

            if equity_curve:
                scale = equity_curve[-1] / curve[0]
                equity_curve.extend((curve * scale)[1:].tolist())
            else:
                equity_curve.extend(curve.tolist())

            results.append(metrics)

            start_idx += self.step_size
            window_id += 1

        if len(results) < self.MIN_WINDOWS:
            raise RuntimeError("Walk-forward produced insufficient windows.")

        logger.info("Walk-forward completed successfully.")

        return self.aggregate_results(results, equity_curve)