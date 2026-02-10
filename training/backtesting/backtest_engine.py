import numpy as np


class BacktestEngine:
    """
    Institutional-grade trading simulator.

    Guarantees:
    - no lookahead execution
    - slippage modeling
    - partial capital deployment
    - forced position flattening
    - exposure tracking
    - turnover visibility
    """

    def run(
        self,
        prices,
        signals,
        initial_cash=10_000,
        transaction_cost=0.001,   # 10 bps
        slippage=0.0005,         # 5 bps
        position_size=1.0        # fraction of capital
    ):

        if len(prices) != len(signals):
            raise ValueError("Prices and signals must have same length")

        if len(prices) < 2:
            return self._empty_result(initial_cash)

        cash = initial_cash
        position = 0.0

        portfolio_values = []

        trade_count = 0
        time_in_market = 0

        prev_signal = "HOLD"

        for i in range(len(prices)):

            price = prices[i]

            # ------------------------------------------------
            # EXECUTE PREVIOUS SIGNAL (NO LOOKAHEAD)
            # ------------------------------------------------

            if prev_signal == "BUY" and cash > 0:

                execution_price = price * (1 + slippage)

                deploy_cash = cash * position_size

                shares = (
                    deploy_cash * (1 - transaction_cost)
                ) / execution_price

                position += shares
                cash -= deploy_cash

                trade_count += 1

            elif prev_signal == "SELL" and position > 0:

                execution_price = price * (1 - slippage)

                proceeds = (
                    position
                    * execution_price
                    * (1 - transaction_cost)
                )

                cash += proceeds
                position = 0

                trade_count += 1

            if position > 0:
                time_in_market += 1

            portfolio_value = cash + position * price
            portfolio_values.append(portfolio_value)

            prev_signal = signals[i]

        # ------------------------------------------------
        # FORCE LIQUIDATION
        # ------------------------------------------------

        if position > 0:

            final_price = prices[-1] * (1 - slippage)

            cash += position * final_price * (1 - transaction_cost)
            position = 0

            portfolio_values[-1] = cash

        portfolio_values = np.array(portfolio_values)

        strategy_return = portfolio_values[-1] / initial_cash - 1
        buy_hold_return = prices[-1] / prices[0] - 1
        alpha = strategy_return - buy_hold_return

        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(returns) > 1 and np.std(returns) != 0:
            sharpe = (
                np.mean(returns)
                / np.std(returns)
                * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        exposure = time_in_market / len(prices)
        turnover = trade_count / len(prices)

        return {
            "final_portfolio": float(portfolio_values[-1]),
            "strategy_return": float(strategy_return),
            "buy_hold_return": float(buy_hold_return),
            "alpha": float(alpha),
            "sharpe_ratio": float(sharpe),
            "trade_count": trade_count,
            "exposure": float(exposure),
            "turnover": float(turnover)
        }

    # ------------------------------------------------

    def _empty_result(self, initial_cash):

        return {
            "final_portfolio": initial_cash,
            "strategy_return": 0.0,
            "buy_hold_return": 0.0,
            "alpha": 0.0,
            "sharpe_ratio": 0.0,
            "trade_count": 0,
            "exposure": 0.0,
            "turnover": 0.0
        }


# -------------------------------------------------
# Compatibility Wrapper Functions
# -------------------------------------------------

def backtest_strategy(df, signal_column="signal"):

    engine = BacktestEngine()

    prices = df["close"].values
    signals = df[signal_column].values

    results = engine.run(prices, signals)

    return {
        "total_return": results["strategy_return"],
        "max_drawdown": 0,
        "sharpe_ratio": results["sharpe_ratio"]
    }


def signal_hit_rate(df, signal_column="signal"):

    df = df.copy()

    df["future_return"] = df["close"].shift(-1) / df["close"] - 1

    hits = df[
        ((df[signal_column] == "BUY") & (df["future_return"] > 0)) |
        ((df[signal_column] == "SELL") & (df["future_return"] < 0))
    ]

    total_signals = df[signal_column].isin(["BUY", "SELL"]).sum()

    return {
        "hit_rate": len(hits) / total_signals if total_signals > 0 else 0
    }
