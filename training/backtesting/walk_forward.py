import numpy as np
import pandas as pd

from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class WalkForwardValidator:
    """
    Institutional Walk-Forward Validator.
    """

    EMBARGO_DAYS = 5
    MIN_TRADES_PER_WINDOW = 5

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

        df = df.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

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

            embargo_start = start_idx - self.EMBARGO_DAYS

            train_dates = unique_dates[
                start_idx - self.window_size:embargo_start
            ]

            test_dates = unique_dates[
                start_idx:start_idx + self.step_size
            ]

            if len(test_dates) < 2:
                break

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]

            model = self.model_trainer(train_df)

            grouped_prices = {}
            grouped_signals = {}

            trade_counter = 0

            for date, slice_df in test_df.groupby("date"):

                prices = dict(
                    zip(slice_df["ticker"], slice_df["close"])
                )

                signals_list = self.signal_generator(
                    model,
                    slice_df
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

            capital = float(metrics["final_portfolio"])

            curve = np.array(metrics["equity_curve"], dtype=float)

            if not np.isfinite(curve).all():
                raise RuntimeError(
                    "Equity curve contains invalid values."
                )

            # FIX — prevent boundary duplication
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

    # ---------------------------------------------------

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
