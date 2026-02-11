import numpy as np
import pandas as pd

from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class WalkForwardValidator:
    """
    Institutional Walk-Forward Validator.

    Guarantees:
    - strict chronological integrity
    - date-aligned windowing
    - no cross-asset leakage
    - capital death protection
    - deterministic simulation
    """

    def __init__(
        self,
        model_trainer,
        signal_generator,
        window_size=252 * 2,
        step_size=63
    ):
        self.model_trainer = model_trainer
        self.signal_generator = signal_generator
        self.window_size = window_size
        self.step_size = step_size

        self.engine = PortfolioBacktestEngine()
        self.regime_detector = MarketRegimeDetector()

    # ---------------------------------------------------

    def run(self, df: pd.DataFrame):

        if "date" not in df.columns:
            raise RuntimeError("WalkForward requires 'date' column.")

        if "ticker" not in df.columns:
            raise RuntimeError(
                "Portfolio walk-forward requires 'ticker' column."
            )

        # HARD SORT — deterministic ordering
        df = df.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        # Detect regimes AFTER sorting
        df = self.regime_detector.detect(df)

        unique_dates = df["date"].drop_duplicates().values

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

            train_dates = unique_dates[
                start_idx - self.window_size:start_idx
            ]

            test_dates = unique_dates[
                start_idx:start_idx + self.step_size
            ]

            if len(test_dates) < 2:
                break

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]

            # TRAIN
            model = self.model_trainer(train_df)

            grouped_prices = {}
            grouped_signals = {}

            for date, slice_df in test_df.groupby("date"):

                prices = dict(
                    zip(slice_df["ticker"], slice_df["close"])
                )

                signals_list = self.signal_generator(
                    model,
                    slice_df
                )

                signals = dict(
                    zip(slice_df["ticker"], signals_list)
                )

                grouped_prices[date] = prices
                grouped_signals[date] = signals

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

            capital = metrics["final_portfolio"]

            curve = np.array(metrics["equity_curve"])

            if not np.isfinite(curve).all():
                raise RuntimeError(
                    "Equity curve contains invalid values."
                )

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

    # ---------------------------------------------------

    def aggregate_results(
        self,
        results,
        equity_curve,
        regime_buckets
    ):

        df = pd.DataFrame(results)
        curve = np.array(equity_curve)

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak
        max_drawdown = float(drawdowns.min())

        return_std = float(df["strategy_return"].std())
        sharpe_std = float(df["sharpe_ratio"].std())

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
            "sharpe_std": sharpe_std,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "return_volatility": return_std,
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),
            "regime_performance": regime_summary,
            "num_windows": len(df)
        }
