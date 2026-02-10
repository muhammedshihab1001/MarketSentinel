import numpy as np


class PortfolioBacktestEngine:
    """
    Institutional multi-asset portfolio simulator.

    Features:
    - risk-parity allocation
    - exposure normalization
    - portfolio rebalancing
    - transaction cost modeling
    - realistic equity curve
    """

    def __init__(
        self,
        transaction_cost=0.001,
        target_vol=0.02,
        max_gross_exposure=1.0
    ):
        self.transaction_cost = transaction_cost
        self.target_vol = target_vol
        self.max_gross_exposure = max_gross_exposure

    # -----------------------------------------------------

    def _estimate_vol(self, prev_prices, prices):

        vols = {}

        if prev_prices is None:
            return {
                t: self.target_vol
                for t in prices
            }

        for ticker in prices:

            if ticker in prev_prices:

                ret = abs(
                    prices[ticker] /
                    prev_prices[ticker] - 1
                )

                vols[ticker] = max(ret, 1e-4)

            else:
                vols[ticker] = self.target_vol

        return vols

    # -----------------------------------------------------

    def _compute_weights(self, signals, vols):

        raw = {}

        for ticker, signal in signals.items():

            if signal != "BUY":
                continue

            vol = vols.get(ticker, self.target_vol)

            raw[ticker] = 1 / vol

        if not raw:
            return {}

        total = sum(raw.values())

        weights = {
            t: w / total
            for t, w in raw.items()
        }

        # enforce gross exposure ceiling
        scale = min(1.0, self.max_gross_exposure)

        return {
            t: w * scale
            for t, w in weights.items()
        }

    # -----------------------------------------------------

    def run(
        self,
        grouped_prices,
        grouped_signals,
        initial_cash=10000
    ):

        cash = initial_cash
        positions = {}

        equity_curve = []

        dates = sorted(grouped_prices.keys())

        prev_prices = None

        for date in dates:

            prices = grouped_prices[date]
            signals = grouped_signals[date]

            vols = self._estimate_vol(
                prev_prices,
                prices
            )

            weights = self._compute_weights(
                signals,
                vols
            )

            portfolio_value = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            target_positions = {}

            # -----------------------------
            # REBALANCE
            # -----------------------------

            for ticker, weight in weights.items():

                allocation = portfolio_value * weight

                shares = (
                    allocation *
                    (1 - self.transaction_cost)
                ) / prices[ticker]

                target_positions[ticker] = shares

            # close removed positions
            for ticker in list(positions.keys()):

                if ticker not in target_positions:

                    cash += (
                        positions[ticker] *
                        prices[ticker] *
                        (1 - self.transaction_cost)
                    )

                    del positions[ticker]

            # open / adjust
            for ticker, shares in target_positions.items():

                cost = shares * prices[ticker]

                current = positions.get(ticker, 0) * prices[ticker]

                delta = cost - current

                cash -= delta
                positions[ticker] = shares

            equity = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            equity_curve.append(equity)

            prev_prices = prices

        curve = np.array(equity_curve)

        returns = (
            np.diff(curve) / curve[:-1]
            if len(curve) > 1 else [0]
        )

        sharpe = (
            np.mean(returns) /
            np.std(returns) *
            np.sqrt(252)
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
