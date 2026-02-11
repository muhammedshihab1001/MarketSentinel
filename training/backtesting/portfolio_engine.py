import numpy as np


class PortfolioBacktestEngine:
    """
    Institutional multi-asset portfolio simulator.

    Guarantees:
    - delta-based transaction costs
    - no leverage unless explicitly allowed
    - volatility-aware allocation
    - capital protection
    - numeric stability
    """

    EPSILON = 1e-12

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

                ret = prices[ticker] / prev_prices[ticker] - 1

                # EWMA-style stabilization
                vols[ticker] = max(
                    abs(ret) * 0.7 + self.target_vol * 0.3,
                    1e-4
                )

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

            raw[ticker] = 1 / max(vol, 1e-4)

        if not raw:
            return {}

        total = sum(raw.values())

        weights = {
            t: w / total
            for t, w in raw.items()
        }

        gross = sum(abs(w) for w in weights.values())

        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    # -----------------------------------------------------

    def run(
        self,
        grouped_prices,
        grouped_signals,
        initial_cash=10000
    ):

        cash = float(initial_cash)
        positions = {}

        equity_curve = []

        dates = sorted(grouped_prices.keys())

        prev_prices = None

        for date in dates:

            prices = grouped_prices[date]
            signals = grouped_signals[date]

            vols = self._estimate_vol(prev_prices, prices)

            weights = self._compute_weights(signals, vols)

            portfolio_value = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            target_positions = {}

            # --------------------------------
            # TARGET SHARES
            # --------------------------------

            for ticker, weight in weights.items():

                allocation = portfolio_value * weight
                target_positions[ticker] = allocation / prices[ticker]

            # --------------------------------
            # DELTA TRADING WITH COST
            # --------------------------------

            for ticker in set(positions) | set(target_positions):

                current_shares = positions.get(ticker, 0)
                target_shares = target_positions.get(ticker, 0)

                delta_shares = target_shares - current_shares

                if abs(delta_shares) < self.EPSILON:
                    continue

                trade_notional = abs(delta_shares) * prices[ticker]
                cost = trade_notional * self.transaction_cost

                cash -= delta_shares * prices[ticker]
                cash -= cost

                if cash < -self.EPSILON:
                    raise RuntimeError(
                        "Backtest attempted leverage beyond cash."
                    )

                if target_shares == 0:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = target_shares

            equity = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            if not np.isfinite(equity):
                raise RuntimeError("Invalid equity value produced.")

            if equity <= 0:
                raise RuntimeError("Portfolio capital depleted.")

            equity_curve.append(equity)

            prev_prices = prices

        curve = np.array(equity_curve)

        if len(curve) < 2:
            returns = np.array([0.0])
        else:
            returns = np.diff(curve) / np.maximum(
                curve[:-1],
                self.EPSILON
            )

        std = np.std(returns)

        sharpe = (
            np.mean(returns) /
            max(std, self.EPSILON) *
            np.sqrt(252)
        )

        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / np.maximum(
            peak,
            self.EPSILON
        )

        return {
            "final_portfolio": float(curve[-1]),
            "strategy_return": float(curve[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": float(sharpe),
            "equity_curve": curve.tolist()
        }
