import numpy as np


class BacktestEngine:
    """
    Realistic trading simulator.

    Features:
    - Capital tracking
    - Position management
    - Transaction cost simulation
    - Buy & Hold benchmark
    - Alpha calculation
    - Trade counting
    - Sharpe ratio
    """

    def run(
        self,
        prices,
        signals,
        initial_cash=10000,
        transaction_cost=0.001  # 0.1%
    ):

        if len(prices) != len(signals):
            raise ValueError("Prices and signals must have same length")

        if len(prices) < 2:
            return {
                "final_portfolio": initial_cash,
                "strategy_return": 0,
                "buy_hold_return": 0,
                "alpha": 0,
                "sharpe_ratio": 0,
                "trade_count": 0
            }

        cash = initial_cash
        position = 0
        portfolio_values = []
        trade_count = 0

        for price, signal in zip(prices, signals):

            # BUY
            if signal == "BUY" and cash > 0:
                position = (cash * (1 - transaction_cost)) / price
                cash = 0
                trade_count += 1

            # SELL
            elif signal == "SELL" and position > 0:
                cash = position * price * (1 - transaction_cost)
                position = 0
                trade_count += 1

            portfolio_value = cash + position * price
            portfolio_values.append(portfolio_value)

        portfolio_values = np.array(portfolio_values)

        # Strategy Return
        strategy_return = portfolio_values[-1] / initial_cash - 1

        # Buy & Hold Benchmark
        buy_hold_return = prices[-1] / prices[0] - 1

        # Alpha
        alpha = strategy_return - buy_hold_return

        # Compute returns safely
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(returns) > 1 and np.std(returns) != 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            "final_portfolio": float(portfolio_values[-1]),
            "strategy_return": float(strategy_return),
            "buy_hold_return": float(buy_hold_return),
            "alpha": float(alpha),
            "sharpe_ratio": float(sharpe),
            "trade_count": trade_count
        }
