import numpy as np


class PortfolioBacktestEngine:
    """
    Institutional multi-asset portfolio simulator.

    Features:
    - capital allocation
    - simultaneous positions
    - exposure limits
    - realistic equity curve
    """

    def run(
        self,
        grouped_prices,
        grouped_signals,
        initial_cash=10000,
        max_position_pct=0.20,
        transaction_cost=0.001
    ):

        cash = initial_cash
        positions = {}
        equity_curve = []

        dates = sorted(grouped_prices.keys())

        for date in dates:

            prices = grouped_prices[date]
            signals = grouped_signals[date]

            portfolio_value = cash

            # mark existing positions
            for ticker, shares in positions.items():

                if ticker in prices:
                    portfolio_value += shares * prices[ticker]

            # rebalance
            for ticker, signal in signals.items():

                if ticker not in prices:
                    continue

                price = prices[ticker]

                position_value = portfolio_value * max_position_pct

                # BUY
                if signal == "BUY" and ticker not in positions:

                    if cash > position_value:

                        shares = (position_value * (1 - transaction_cost)) / price

                        positions[ticker] = shares
                        cash -= position_value

                # SELL
                elif signal == "SELL" and ticker in positions:

                    shares = positions.pop(ticker)

                    cash += shares * price * (1 - transaction_cost)

            total_equity = cash

            for ticker, shares in positions.items():

                if ticker in prices:
                    total_equity += shares * prices[ticker]

            equity_curve.append(total_equity)

        curve = np.array(equity_curve)

        returns = np.diff(curve) / curve[:-1] if len(curve) > 1 else [0]

        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0 else 0
        )

        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak

        return {
            "final_portfolio": float(curve[-1]),
            "strategy_return": float(curve[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": float(sharpe),
            "equity_curve": curve.tolist()
        }
