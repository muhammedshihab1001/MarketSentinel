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
    MIN_FEATURE_VARIANCE = 1e-8
    MAX_TURNOVER = 0.65
    MAX_CAPITAL_MULTIPLE = 50

    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

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

    def _assert_monotonic(self, df):

        grouped = df.sort_values("date").groupby("ticker")

        for ticker, g in grouped:
            if not g["date"].is_monotonic_increasing:
                raise RuntimeError(
                    f"Non-monotonic dates detected for {ticker}"
                )

    ############################################

    def _validate_training_frame(self, df):

        if df["ticker"].nunique() < 3:
            raise RuntimeError("Training universe too small.")

        validate_feature_schema(df.loc[:, MODEL_FEATURES])

        features = df.loc[:, MODEL_FEATURES].to_numpy(dtype=float)

        if not np.isfinite(features).all():
            raise RuntimeError("Non-finite feature values detected.")

        if np.min(np.var(features, axis=0)) < self.MIN_FEATURE_VARIANCE:
            raise RuntimeError("Feature variance collapsed.")

        if df["target"].nunique() < 2:
            raise RuntimeError("Training labels collapsed.")

        if len(df) < self.window_size * self.MIN_TRAIN_RATIO:
            raise RuntimeError(
                "Training window too small after embargo."
            )

    ############################################

    def _sanity_check_model(self, model, sample_df):

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Classifier required.")

        X = sample_df.loc[:, MODEL_FEATURES].iloc[:50]

        preds = model.predict_proba(X)

        if preds.ndim != 2 or preds.shape[1] < 2:
            raise RuntimeError("Invalid predict_proba shape.")

        probs = preds[:, 1]

        if not np.isfinite(probs).all():
            raise RuntimeError("Non-finite predictions.")

        # 🔴 NEW — probability collapse guard
        if np.std(probs) < 5e-5:
            raise RuntimeError("Model collapsed.")

        if np.mean(probs) < 0.02 or np.mean(probs) > 0.98:
            raise RuntimeError("Probability collapse detected.")

    ############################################

    def _validate_signals(self, signals):

        invalid = set(signals.values()) - self.VALID_SIGNALS

        if invalid:
            raise RuntimeError(
                f"Invalid signal values detected: {invalid}"
            )

        if all(s == "HOLD" for s in signals.values()):
            raise RuntimeError(
                "Degenerate strategy — all HOLD signals."
            )

    ############################################

    def _calculate_turnover(self, grouped_signals):

        previous = None
        flips = 0
        total = 0

        for date in sorted(grouped_signals):

            current = grouped_signals[date]

            if previous is not None:

                shared = set(previous) & set(current)

                for t in shared:
                    if previous[t] != current[t]:
                        flips += 1

                total += len(shared)

            previous = current

        if total == 0:
            return 0.0

        return flips / total

    ############################################

    def run(self, df: pd.DataFrame):

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        self._assert_monotonic(df)

        unique_dates = pd.to_datetime(
            df["date"].drop_duplicates()
        ).sort_values()

        if len(unique_dates) < self.window_size + self.step_size + 1:
            raise RuntimeError("Dataset too small.")

        results = []
        equity_curve = []
        sharpe_series = []

        initial_capital = 10_000.0
        start_idx = self.window_size

        while start_idx < len(unique_dates) - 1:

            # 🔴 Force window independence
            model = None

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
                (unique_dates >= train_end_date)
            ][:self.step_size + 1]

            if len(test_dates) < 2:
                break

            train_df = df.loc[
                (df["date"] >= train_dates.iloc[0]) &
                (df["date"] <= train_dates.iloc[-1])
            ].copy()

            test_df = df.loc[
                (df["date"] >= test_dates.iloc[0]) &
                (df["date"] <= test_dates.iloc[-1])
            ].copy()

            # 🔴 CRITICAL FIX — compute regime AFTER split (no leakage)
            train_df = self.regime_detector.detect(train_df)
            test_df = self.regime_detector.detect(test_df)

            # 🔴 Schema re-validation after regime injection
            validate_feature_schema(train_df.loc[:, MODEL_FEATURES])
            validate_feature_schema(test_df.loc[:, MODEL_FEATURES])

            self._validate_training_frame(train_df)

            model = self.model_trainer(train_df)
            self._sanity_check_model(model, train_df)

            grouped_prices = {}
            grouped_signals = {}
            trade_counter = 0

            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(len(test_dates_sorted) - 1):

                signal_date = test_dates_sorted[i]
                execution_date = test_dates_sorted[i + 1]

                signal_slice = test_df[test_df["date"] == signal_date]
                exec_slice = test_df[test_df["date"] == execution_date]

                prices = {
                    t: float(p)
                    for t, p in zip(exec_slice["ticker"], exec_slice["close"])
                    if pd.notna(p) and np.isfinite(p) and p > 0
                }

                signals_list = self.signal_generator(
                    model,
                    signal_slice
                )

                signals = dict(
                    zip(signal_slice["ticker"], signals_list)
                )

                self._validate_signals(signals)

                shared = set(prices) & set(signals)

                if len(shared) < self.MIN_ASSETS_PER_DAY:
                    continue

                prices = {t: prices[t] for t in shared}
                signals = {t: signals[t] for t in shared}

                trade_counter += sum(
                    1 for s in signals.values()
                    if s != "HOLD"
                )

                grouped_prices[execution_date] = prices
                grouped_signals[execution_date] = signals

            if trade_counter < self.MIN_TRADES_PER_WINDOW:
                start_idx += self.step_size
                continue

            turnover = self._calculate_turnover(grouped_signals)

            if turnover > self.MAX_TURNOVER:
                start_idx += self.step_size
                continue

            # 🔴 FIX — remove path-dependent capital bias
            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=initial_capital
            )

            curve = np.array(metrics["equity_curve"], dtype=float)

            if not np.isfinite(curve).all():
                raise RuntimeError("Equity curve corrupted.")

            if equity_curve:
                equity_curve.extend(curve[1:].tolist())
            else:
                equity_curve.extend(curve.tolist())

            results.append(metrics)
            sharpe_series.append(metrics["sharpe_ratio"])

            start_idx += self.step_size

        if len(results) < self.MIN_WINDOWS:
            raise RuntimeError(
                "Walk-forward produced insufficient windows."
            )

        sharpe_std = np.std(sharpe_series)

        if sharpe_std > 1.2:
            raise RuntimeError(
                "Sharpe instability detected — strategy unreliable."
            )

        return self.aggregate_results(results, equity_curve)

    ############################################

    def aggregate_results(self, results, equity_curve):

        df = pd.DataFrame(results)
        curve = np.array(equity_curve, dtype=float)

        if not np.isfinite(curve).all():
            raise RuntimeError("Equity curve corrupted.")

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(
            df[df["strategy_return"] < 0]["strategy_return"].sum()
        ) or 1e-6

        profit_factor = float(gains / losses)

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "profit_factor": profit_factor,
            "max_drawdown": float(drawdowns.min()),
            "return_volatility": float(df["strategy_return"].std()),
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),
            "num_windows": len(df)
        }
