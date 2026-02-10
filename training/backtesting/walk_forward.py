import numpy as np
import pandas as pd

from training.backtesting.backtest_engine import BacktestEngine
from training.backtesting.regime import MarketRegimeDetector


class WalkForwardValidator:
    """
    Institutional Walk-Forward Validator.

    Upgrades:
    ✅ regime-aware evaluation
    ✅ equity curve stitching
    ✅ drawdown tracking
    ✅ stability metrics
    ✅ promotion-grade outputs
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

        self.engine = BacktestEngine()
        self.regime_detector = MarketRegimeDetector()

    # ---------------------------------------------------

    def run(self, df: pd.DataFrame):

        # 🔥 Detect regimes ONCE (important for speed)
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

            # -----------------------------
            # Train
            # -----------------------------

            model = self.model_trainer(train_df)

            # -----------------------------
            # Predict
            # -----------------------------

            signals = self.signal_generator(model, test_df)
            prices = test_df["close"].values

            metrics = self.engine.run(
                prices,
                signals,
                initial_cash=capital
            )

            capital = metrics["final_portfolio"]

            equity_curve.append(capital)
            results.append(metrics)

            # -----------------------------
            # REGIME TAGGING
            # -----------------------------

            dominant_regime = (
                test_df["regime"]
                .value_counts()
                .idxmax()
            )

            if dominant_regime in regime_buckets:
                regime_buckets[dominant_regime].append(metrics)

            start += self.step_size

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

        if not results:
            raise ValueError("Walk-forward produced no windows")

        df = pd.DataFrame(results)
        curve = np.array(equity_curve)

        # ------------------------------------------------
        # MAX DRAWDOWN
        # ------------------------------------------------

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak
        max_drawdown = float(drawdowns.min())

        # ------------------------------------------------
        # RETURN STABILITY
        # ------------------------------------------------

        return_std = float(df["strategy_return"].std())

        # ------------------------------------------------
        # REGIME METRICS
        # ------------------------------------------------

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

        # ------------------------------------------------

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_buy_hold_return": float(df["buy_hold_return"].mean()),
            "avg_alpha": float(df["alpha"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "avg_trades": float(df["trade_count"].mean()),

            # 🔥 Institutional Risk Metrics
            "max_drawdown": max_drawdown,
            "return_volatility": return_std,
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),

            # 🔥 NEW — REGIME INTELLIGENCE
            "regime_performance": regime_summary,

            "num_windows": len(df)
        }
