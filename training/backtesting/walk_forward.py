import numpy as np
import pandas as pd

from training.backtesting.backtest_engine import BacktestEngine


class WalkForwardValidator:
    """
    Performs walk-forward validation for trading strategies.
    """

    def __init__(
        self,
        model_trainer,        # function that trains model
        signal_generator,    # function that produces signals
        window_size=252*2,   # ~2 years
        step_size=63         # ~1 quarter
    ):
        self.model_trainer = model_trainer
        self.signal_generator = signal_generator
        self.window_size = window_size
        self.step_size = step_size
        self.engine = BacktestEngine()

    def run(self, df: pd.DataFrame):

        results = []

        start = self.window_size

        while start < len(df):

            train_df = df.iloc[start-self.window_size:start]
            test_df = df.iloc[start:start+self.step_size]

            if len(test_df) < 2:
                break

            # Train model on past data
            model = self.model_trainer(train_df)

            # Generate signals on unseen future
            signals = self.signal_generator(model, test_df)

            prices = test_df["close"].values

            metrics = self.engine.run(prices, signals)

            results.append(metrics)

            start += self.step_size

        return self.aggregate_results(results)

    def aggregate_results(self, results):

        df = pd.DataFrame(results)

        return {
            "avg_strategy_return": df["strategy_return"].mean(),
            "avg_buy_hold_return": df["buy_hold_return"].mean(),
            "avg_alpha": df["alpha"].mean(),
            "avg_sharpe": df["sharpe_ratio"].mean(),
            "avg_trades": df["trade_count"].mean(),
            "num_windows": len(df)
        }
