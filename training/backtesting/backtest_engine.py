import numpy as np


class BacktestEngine:

    def run(self, prices, signals, initial_cash=10000):

        cash = initial_cash
        position = 0
        portfolio_values = []

        for price, signal in zip(prices, signals):

            if signal == "BUY" and cash > 0:
                position = cash / price
                cash = 0

            elif signal == "SELL" and position > 0:
                cash = position * price
                position = 0

            portfolio_value = cash + position * price
            portfolio_values.append(portfolio_value)

        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        sharpe = (
            np.mean(returns) / np.std(returns)
            if np.std(returns) != 0 else 0
        )

        return {
            "final_portfolio": portfolio_values[-1],
            "total_return": portfolio_values[-1] / initial_cash - 1,
            "sharpe_ratio": sharpe
        }
