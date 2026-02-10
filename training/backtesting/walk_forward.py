import numpy as np
import pandas as pd

from training.backtesting.backtest_engine import BacktestEngine


class WalkForwardValidator:
    """
    Institutional Walk-Forward Validator.

    Upgrades:
    ✅ equity curve stitching
    ✅ drawdown tracking
    ✅ regime visibility
    ✅ stability metrics
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

    # ---------------------------------------------------

    def run(self, df: pd.DataFrame):

        results = []
        equity_curve = []

        start = self.window_size
        capital = 10_000  # rolling capital simulation

        while start < len(df):

            train_df = df.iloc[start-self.window_size:start]
            test_df = df.iloc[start:start+self.step_size]

            if len(test_df) < 2:
                break

            # Train
            model = self.model_trainer(train_df)

            # Predict
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

            start += self.step_size

        return self.aggregate_results(results, equity_curve)

    # ---------------------------------------------------

    def aggregate_results(self, results, equity_curve):

        if not results:
            raise ValueError("Walk-forward produced no windows")

        df = pd.DataFrame(results)

        curve = np.array(equity_curve)

        # ------------------------------------------------
        # MAX DRAWDOWN (CRITICAL RISK METRIC)
        # ------------------------------------------------

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak
        max_drawdown = float(drawdowns.min())

        # ------------------------------------------------
        # RETURN STABILITY
        # ------------------------------------------------

        return_std = float(df["strategy_return"].std())

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_buy_hold_return": float(df["buy_hold_return"].mean()),
            "avg_alpha": float(df["alpha"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "avg_trades": float(df["trade_count"].mean()),

            # 🔥 NEW — Institutional Metrics
            "max_drawdown": max_drawdown,
            "return_volatility": return_std,
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),

            "num_windows": len(df)
        }
