import numpy as np
import pandas as pd

from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema
)

from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class WalkForwardValidator:

    MIN_TRADES_PER_WINDOW = 5
    MIN_TRAIN_RATIO = 0.75
    MIN_WINDOWS = 6
    MIN_ASSETS_PER_DAY = 3

    def __init__(
        self,
        model_trainer,
        signal_generator,
        window_size=252,
        step_size=63,
        embargo_days=None
    ):
        self.model_trainer = model_trainer
        self.signal_generator = signal_generator
        self.window_size = window_size
        self.step_size = step_size

        self.EMBARGO_DAYS = embargo_days or max(10, window_size // 12)

        self.engine = PortfolioBacktestEngine()
        self.regime_detector = MarketRegimeDetector()

    ############################################
    # TRAIN VALIDATION
    ############################################

    def _validate_training_frame(self, df):

        if df["ticker"].nunique() < 3:
            raise RuntimeError("Training universe too small.")

        if not df.index.equals(
            df.sort_values(["date", "ticker"]).index
        ):
            raise RuntimeError("Training dataframe not sorted.")

        validate_feature_schema(df.loc[:, MODEL_FEATURES])

        if df["target"].nunique() < 2:
            raise RuntimeError("Training labels collapsed.")

        if len(df) < self.window_size * self.MIN_TRAIN_RATIO:
            raise RuntimeError(
                "Training window too small after embargo."
            )

    ############################################
    # MODEL CHECK
    ############################################

    def _sanity_check_model(self, model, sample_df):

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Classifier required.")

        X = sample_df.loc[:, MODEL_FEATURES].iloc[:50]
        preds = model.predict_proba(X)[:, 1]

        if not np.isfinite(preds).all():
            raise RuntimeError("Non-finite predictions.")

        if np.std(preds) < 5e-5:
            raise RuntimeError("Model collapsed.")

    ############################################
    # SAFE PRICE BUILDER
    ############################################

    def _build_price_dict(self, slice_df):

        prices = {
            t: float(p)
            for t, p in zip(slice_df["ticker"], slice_df["close"])
            if pd.notna(p) and np.isfinite(p) and p > 0
        }

        return prices

    ############################################
    # CRITICAL FIX — UNIVERSE ALIGNMENT
    ############################################

    def _align_universe(self, prices, signals):

        shared = set(prices.keys()) & set(signals.keys())

        if len(shared) < self.MIN_ASSETS_PER_DAY:
            return None, None

        prices = {t: prices[t] for t in shared}
        signals = {t: signals[t] for t in shared}

        return prices, signals

    ############################################
    # RUN
    ############################################

    def run(self, df: pd.DataFrame):

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        df = self.regime_detector.detect(df)

        unique_dates = pd.to_datetime(
            df["date"].drop_duplicates()
        ).sort_values()

        if len(unique_dates) < self.window_size + self.step_size:
            raise RuntimeError("Dataset too small.")

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

            if len(train_dates) < self.window_size * self.MIN_TRAIN_RATIO:
                start_idx += self.step_size
                continue

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

            grouped_prices = {}
            grouped_signals = {}

            trade_counter = 0

            ############################################
            # DAILY LOOP
            ############################################

            for date, slice_df in test_df.groupby("date"):

                prices = self._build_price_dict(slice_df)

                signals_list = self.signal_generator(
                    model,
                    slice_df
                )

                raw_signals = dict(
                    zip(slice_df["ticker"], signals_list)
                )

                prices, signals = self._align_universe(
                    prices,
                    raw_signals
                )

                if prices is None:
                    continue

                trade_counter += sum(
                    1 for s in signals.values()
                    if s != "HOLD"
                )

                grouped_prices[date] = prices
                grouped_signals[date] = signals

            if trade_counter < self.MIN_TRADES_PER_WINDOW:
                start_idx += self.step_size
                continue

            ############################################
            # PORTFOLIO
            ############################################

            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=capital
            )

            capital = float(metrics["final_portfolio"])

            curve = np.array(metrics["equity_curve"], dtype=float)

            if not np.isfinite(curve).all():
                raise RuntimeError("Invalid equity curve.")

            if equity_curve:
                equity_curve.extend(curve[1:].tolist())
            else:
                equity_curve.extend(curve.tolist())

            results.append(metrics)

            dominant_regime = (
                test_df["regime"].value_counts().idxmax()
            )

            if dominant_regime in regime_buckets:
                regime_buckets[dominant_regime].append(metrics)

            start_idx += self.step_size

        if len(results) < self.MIN_WINDOWS:
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

    def aggregate_results(self, results, equity_curve, regime_buckets):

        df = pd.DataFrame(results)
        curve = np.array(equity_curve, dtype=float)

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(
            df[df["strategy_return"] < 0]["strategy_return"].sum()
        ) or 1e-6

        profit_factor = float(gains / losses)

        regime_summary = {}

        for regime, bucket in regime_buckets.items():

            if bucket:
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
            "max_drawdown": float(drawdowns.min()),
            "return_volatility": float(df["strategy_return"].std()),
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),
            "regime_performance": regime_summary,
            "num_windows": len(df)
        }
