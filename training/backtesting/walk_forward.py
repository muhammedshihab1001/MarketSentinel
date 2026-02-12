import numpy as np
import pandas as pd

from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema
)

from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class WalkForwardValidator:

    EMBARGO_DAYS = 52
    MIN_TRADES_PER_WINDOW = 5
    MIN_TRAIN_ROWS = 500

    def __init__(
        self,
        model_trainer,
        signal_generator,
        window_size=504,
        step_size=63
    ):
        self.model_trainer = model_trainer
        self.signal_generator = signal_generator
        self.window_size = window_size
        self.step_size = step_size

        self.engine = PortfolioBacktestEngine()
        self.regime_detector = MarketRegimeDetector()

    ############################################
    # VALIDATION
    ############################################

    def _validate_training_frame(self, df):

        if df["ticker"].nunique() < 3:
            raise RuntimeError(
                "Training universe too small. Cross-sectional learning unsafe."
            )

        expected = df.sort_values(["date", "ticker"]).index

        if not expected.equals(df.index):
            raise RuntimeError(
                "Training dataframe not sorted by ['date','ticker']."
            )

        feature_slice = df.loc[:, MODEL_FEATURES]
        validate_feature_schema(feature_slice)

        if df["target"].nunique() < 2:
            raise RuntimeError(
                "Training labels collapsed. No class diversity."
            )

        if len(df) < self.MIN_TRAIN_ROWS:
            raise RuntimeError(
                "Training window too small after embargo."
            )

    ############################################
    # MODEL CHECKS
    ############################################

    def _validate_classifier_contract(self, model):

        if not hasattr(model, "predict_proba"):
            raise RuntimeError(
                "Model lacks predict_proba. Classifier required."
            )

    def _sanity_check_model(self, model, sample_df):

        self._validate_classifier_contract(model)

        X = sample_df.loc[:, MODEL_FEATURES].iloc[:50]

        preds = model.predict_proba(X)[:, 1]

        if not np.isfinite(preds).all():
            raise RuntimeError("Model produced non-finite predictions.")

        if np.std(preds) < 5e-5:
            raise RuntimeError(
                "Model collapsed. Predictions nearly constant."
            )

    def _stability_check(self, model, test_df):

        X = test_df.loc[:, MODEL_FEATURES]

        preds = model.predict_proba(X)[:, 1]

        if not np.isfinite(preds).all():
            raise RuntimeError("Model unstable — non-finite outputs.")

        if np.std(preds) < 1e-4:
            raise RuntimeError(
                "Model unstable on unseen data."
            )

    ############################################
    # PRICE VALIDATION
    ############################################

    def _validate_prices(self, prices):

        arr = np.array(list(prices.values()), dtype=float)

        if not np.isfinite(arr).all():
            raise RuntimeError("Invalid prices detected.")

        if (arr <= 0).any():
            raise RuntimeError("Non-positive prices detected.")

    ############################################
    # RUN
    ############################################

    def run(self, df: pd.DataFrame):

        if "date" not in df.columns:
            raise RuntimeError("WalkForward requires 'date' column.")

        if "ticker" not in df.columns:
            raise RuntimeError(
                "Portfolio walk-forward requires 'ticker' column."
            )

        df = df.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        original_rows = len(df)

        df = self.regime_detector.detect(df)

        if df.empty:
            raise RuntimeError(
                "All rows removed during regime detection."
            )

        if len(df) < original_rows * 0.6:
            raise RuntimeError(
                "Excessive data loss during regime detection."
            )

        unique_dates = pd.to_datetime(
            df["date"].drop_duplicates()
        ).sort_values()

        if len(unique_dates) < self.window_size + self.step_size:
            raise RuntimeError(
                "Dataset too small for walk-forward validation."
            )

        results = []
        equity_curve = []

        regime_buckets = {
            "BULL": [],
            "BEAR": [],
            "SIDEWAYS": []
        }

        capital = 10_000
        start_idx = self.window_size

        while start_idx < len(unique_dates):

            train_end_date = unique_dates.iloc[start_idx]

            embargo_cutoff = train_end_date - pd.Timedelta(
                days=self.EMBARGO_DAYS
            )

            train_dates = unique_dates[
                unique_dates < embargo_cutoff
            ].tail(self.window_size)

            if len(train_dates) < self.MIN_TRAIN_ROWS:
                raise RuntimeError(
                    "Embargo consumed too much training history."
                )

            test_dates = unique_dates[
                (unique_dates >= train_end_date) &
                (unique_dates < train_end_date + pd.Timedelta(days=365))
            ][:self.step_size]

            if len(test_dates) < 2:
                break

            train_df = df.loc[
                df["date"].between(
                    train_dates.iloc[0],
                    train_dates.iloc[-1]
                )
            ].copy()

            test_df = df.loc[
                df["date"].between(
                    test_dates.iloc[0],
                    test_dates.iloc[-1]
                )
            ].copy()

            self._validate_training_frame(train_df)

            model = self.model_trainer(train_df)

            self._sanity_check_model(model, train_df)
            self._stability_check(model, test_df)

            grouped_prices = {}
            grouped_signals = {}

            trade_counter = 0

            for date, slice_df in test_df.groupby("date"):

                prices = dict(
                    zip(slice_df["ticker"], slice_df["close"])
                )

                self._validate_prices(prices)

                signals_list = self.signal_generator(
                    model,
                    slice_df
                )

                if len(signals_list) != len(slice_df):
                    raise RuntimeError(
                        "Signal generator returned mismatched length."
                    )

                trade_counter += sum(
                    1 for s in signals_list
                    if s != "HOLD"
                )

                signals = dict(
                    zip(slice_df["ticker"], signals_list)
                )

                grouped_prices[date] = prices
                grouped_signals[date] = signals

            if trade_counter < self.MIN_TRADES_PER_WINDOW:
                raise RuntimeError(
                    "Strategy produced insufficient trades."
                )

            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=capital
            )

            if not np.isfinite(metrics["final_portfolio"]):
                raise RuntimeError(
                    "Portfolio produced invalid capital value."
                )

            if metrics["final_portfolio"] <= 0:
                raise RuntimeError(
                    "Strategy collapsed. Capital depleted."
                )

            curve = np.array(metrics["equity_curve"], dtype=float)

            if not np.isfinite(curve).all():
                raise RuntimeError(
                    "Equity curve contains invalid values."
                )

            if (curve <= 0).any():
                raise RuntimeError(
                    "Negative equity encountered."
                )

            capital = float(metrics["final_portfolio"])

            if equity_curve:
                equity_curve.extend(curve[1:].tolist())
            else:
                equity_curve.extend(curve.tolist())

            results.append(metrics)

            dominant_regime = (
                test_df["regime"]
                .value_counts()
                .idxmax()
            )

            if dominant_regime in regime_buckets:
                regime_buckets[dominant_regime].append(metrics)

            start_idx += self.step_size

        if len(results) < 8:
            raise RuntimeError(
                "Walk-forward produced insufficient windows."
            )

        return self.aggregate_results(
            results,
            equity_curve,
            regime_buckets
        )

    ############################################
    # AGGREGATION
    ############################################

    def aggregate_results(
        self,
        results,
        equity_curve,
        regime_buckets
    ):

        df = pd.DataFrame(results)
        curve = np.array(equity_curve, dtype=float)

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak
        max_drawdown = float(drawdowns.min())

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(
            df[df["strategy_return"] < 0]["strategy_return"].sum()
        ) or 1e-6

        profit_factor = float(gains / losses)

        regime_summary = {}

        for regime, bucket in regime_buckets.items():

            if not bucket:
                continue

            bucket_df = pd.DataFrame(bucket)

            regime_summary[regime] = {
                "avg_return": float(bucket_df["strategy_return"].mean()),
                "avg_sharpe": float(bucket_df["sharpe_ratio"].mean()),
                "windows": len(bucket_df)
            }

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "return_volatility": float(df["strategy_return"].std()),
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),
            "regime_performance": regime_summary,
            "num_windows": len(df)
        }
