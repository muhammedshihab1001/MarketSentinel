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
    MAX_SHARPE_REALITY = 4.0
    MIN_EQUITY_STD = 1e-4
    MIN_TEST_DAYS = 20

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

    def _assert_no_future_leak(self, train_df, test_df):

        if train_df["date"].max() >= test_df["date"].min():
            raise RuntimeError(
                "Temporal boundary violated — future leakage detected."
            )

        if set(train_df.index) & set(test_df.index):
            raise RuntimeError(
                "Train/Test index overlap — leakage."
            )

    ############################################

    def _assert_monotonic(self, df):

        grouped = df.sort_values("date").groupby("ticker")

        for ticker, g in grouped:
            if not g["date"].is_monotonic_increasing:
                raise RuntimeError(
                    f"Non-monotonic dates detected for {ticker}"
                )

    ############################################

    def _validate_survivorship(self, df):

        coverage = df.groupby("ticker").size()

        if (coverage < self.window_size * 0.60).any():
            raise RuntimeError(
                "Survivorship bias risk — assets missing large history."
            )

    ############################################

    def _distribution_guard(self, train_df, test_df):

        train_mu = train_df[MODEL_FEATURES].mean()
        train_std = train_df[MODEL_FEATURES].std(ddof=0) + 1e-9

        z = np.abs(
            (test_df[MODEL_FEATURES] - train_mu) / train_std
        )

        if (z > 8).any().any():
            raise RuntimeError(
                "Severe feature distribution shift detected."
            )

    ############################################

    def _validate_training_frame(self, df):

        if df["ticker"].nunique() < 3:
            raise RuntimeError("Training universe too small.")

        validate_feature_schema(df.loc[:, MODEL_FEATURES])

        features = df.loc[:, MODEL_FEATURES].to_numpy(dtype=float)

        if features.shape[1] != len(MODEL_FEATURES):
            raise RuntimeError("Feature width changed.")

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

        self._validate_survivorship(df)

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

        if np.std(probs) < 5e-5:
            raise RuntimeError("Model collapsed.")

        if np.mean(probs) < 0.02 or np.mean(probs) > 0.98:
            raise RuntimeError("Probability collapse detected.")

    ############################################

    def _validate_signals(self, signals, expected_len):

        if len(signals) != expected_len:
            raise RuntimeError(
                "Signal length mismatch — generator dropped rows."
            )

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

            train_dates = unique_dates.iloc[
                start_idx - self.window_size : start_idx
            ]

            embargo_cutoff = train_dates.iloc[-1]

            test_dates = unique_dates.iloc[
                start_idx : start_idx + self.step_size
            ]

            if len(test_dates) < self.MIN_TEST_DAYS:
                break

            train_df = df.loc[
                (df["date"] >= train_dates.iloc[0]) &
                (df["date"] <= embargo_cutoff)
            ].copy()

            test_df = df.loc[
                (df["date"] >= test_dates.iloc[0]) &
                (df["date"] <= test_dates.iloc[-1])
            ].copy()

            self._assert_no_future_leak(train_df, test_df)

            train_df = self.regime_detector.detect(train_df.copy())
            test_df = self.regime_detector.detect(test_df.copy())

            self._validate_training_frame(train_df)
            self._distribution_guard(train_df, test_df)

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

                self._validate_signals(signals, len(signal_slice))

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

            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=initial_capital
            )

            if metrics["sharpe_ratio"] > self.MAX_SHARPE_REALITY:
                raise RuntimeError(
                    "Sharpe too high — simulation likely invalid."
                )

            curve = np.array(metrics["equity_curve"], dtype=float)

            if np.std(curve) < self.MIN_EQUITY_STD:
                raise RuntimeError(
                    "Equity curve unnaturally smooth — overfit suspected."
                )

            if curve[-1] > initial_capital * self.MAX_CAPITAL_MULTIPLE:
                raise RuntimeError(
                    "Equity explosion detected — simulation bug."
                )

            if equity_curve:
                scale = equity_curve[-1] / curve[0]
                equity_curve.extend((curve * scale)[1:].tolist())
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
