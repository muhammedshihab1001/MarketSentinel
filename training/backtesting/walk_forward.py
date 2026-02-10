import numpy as np
import pandas as pd

from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class WalkForwardValidator:
    """
    Institutional Walk-Forward Validator.

    Guarantees:
    - strict time ordering
    - multi-asset portfolio simulation
    - capital compounding
    - regime-aware evaluation
    - promotion-grade statistics
    """

    def __init__(
        self,
        model_trainer,
        signal_generator,
        window_size=252*2,
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

        # HARD TIME SORT
        df = df.sort_values("date").reset_index(drop=True)

        # Detect regimes once
        df = self.regime_detector.detect(df)

        results = []
        equity_curve = []

        regime_buckets = {
            "BULL": [],
            "BEAR": [],
            "SIDEWAYS": []
        }

        start = self.window_size
        capital = 10_000

        while start < len(df):

            train_df = df.iloc[start-self.window_size:start]
            test_df = df.iloc[start:start+self.step_size]

            if len(test_df) < 2:
                break

            # -----------------------------------
            # TRAIN
            # -----------------------------------

            model = self.model_trainer(train_df)

            # -----------------------------------
            # BUILD PORTFOLIO STRUCTURE
            # -----------------------------------

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

            # -----------------------------------
            # RUN PORTFOLIO ENGINE
            # -----------------------------------

            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=capital
            )

            capital = metrics["final_portfolio"]

            equity_curve.extend(metrics["equity_curve"])
            results.append(metrics)

            # -----------------------------------
            # REGIME TAGGING
            # -----------------------------------

            dominant_regime = (
                test_df["regime"]
                .value_counts()
                .idxmax()
            )

            if dominant_regime in regime_buckets:
                regime_buckets[dominant_regime].append(metrics)

            start += self.step_size

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

        # -------------------------------
        # MAX DRAWDOWN
        # -------------------------------

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak
        max_drawdown = float(drawdowns.min())

        # -------------------------------
        # STABILITY
        # -------------------------------

        return_std = float(df["strategy_return"].std())
        sharpe_std = float(df["sharpe_ratio"].std())

        # -------------------------------
        # PROFIT FACTOR
        # -------------------------------

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(
            df[df["strategy_return"] < 0]["strategy_return"].sum()
        ) or 1e-6

        profit_factor = float(gains / losses)

        # -------------------------------
        # REGIME METRICS
        # -------------------------------

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

        # -------------------------------

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
