import numpy as np
from collections import defaultdict


class PortfolioBacktestEngine:

    EPSILON = 1e-12
    VOL_WINDOW = 20
    REBALANCE_THRESHOLD = 0.02
    CASH_BUFFER = 0.02
    MAX_INV_VOL = 5.0

    def __init__(
        self,
        transaction_cost=0.001,
        slippage=0.0005,
        target_vol=0.02,
        max_gross_exposure=1.0
    ):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.target_vol = target_vol
        self.max_gross_exposure = max_gross_exposure

        self.return_buffers = defaultdict(list)

    def _reset_state(self):
        self.return_buffers = defaultdict(list)

    def _safe_price(self, price):
        if price is None or not np.isfinite(price) or price <= 0:
            raise RuntimeError("Invalid market price encountered.")
        return float(price)

    def _update_vol_buffers(self, prev_prices, prices):

        vols = {}

        for ticker in prices:

            price = self._safe_price(prices[ticker])

            if prev_prices and ticker in prev_prices:

                prev = self._safe_price(prev_prices[ticker])

                ret = price / prev - 1
                buf = self.return_buffers[ticker]

                buf.append(ret)

                if len(buf) > self.VOL_WINDOW:
                    buf.pop(0)

                vol = np.std(buf) if len(buf) > 2 else self.target_vol
                vols[ticker] = max(vol, 1e-4)

            else:
                vols[ticker] = self.target_vol

        return vols

    def _compute_weights(self, signals, vols, current_weights):

        raw = {}

        for ticker, signal in signals.items():

            if signal != "BUY":
                continue

            vol = vols.get(ticker, self.target_vol)

            inv_vol = min(1 / max(vol, 1e-4), self.MAX_INV_VOL)

            raw[ticker] = inv_vol

        if not raw:
            return {}

        total = sum(raw.values())

        new_weights = {
            t: w / total
            for t, w in raw.items()
        }

        final = {}

        for t, w in new_weights.items():

            if abs(w - current_weights.get(t, 0)) < self.REBALANCE_THRESHOLD:
                final[t] = current_weights.get(t, 0)
            else:
                final[t] = w

        gross = sum(abs(w) for w in final.values())

        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            final = {t: w * scale for t, w in final.items()}

        portfolio_vol = np.sqrt(
            sum((vols[t] ** 2) * (w ** 2) for t, w in final.items())
        )

        if portfolio_vol > self.EPSILON:
            vol_scale = min(
                self.target_vol / portfolio_vol,
                self.max_gross_exposure
            )
            final = {t: w * vol_scale for t, w in final.items()}

        return final

    def run(
        self,
        grouped_prices,
        grouped_signals,
        initial_cash=10000
    ):

        self._reset_state()

        cash = float(initial_cash)
        positions = {}

        equity_curve = []
        turnover = 0.0

        prev_prices = None
        prev_signals = None

        for date in sorted(grouped_prices.keys()):

            prices = grouped_prices[date]

            if prev_prices is None:
                prev_prices = prices
                prev_signals = grouped_signals[date]
                equity_curve.append(cash)
                continue

            signals = prev_signals

            vols = self._update_vol_buffers(prev_prices, prices)

            portfolio_value = cash + sum(
                positions.get(t, 0) * self._safe_price(prices.get(t))
                for t in positions
            )

            deployable_capital = portfolio_value * (1 - self.CASH_BUFFER)

            current_weights = {
                t: (positions[t] * self._safe_price(prices[t])) / portfolio_value
                for t in positions
                if portfolio_value > self.EPSILON
            }

            weights = self._compute_weights(
                signals,
                vols,
                current_weights
            )

            target_positions = {
                t: (deployable_capital * w) / self._safe_price(prices[t])
                for t, w in weights.items()
            }

            simulated_cash = cash

            for ticker in set(positions) | set(target_positions):

                trade_price = self._safe_price(prices.get(ticker))

                delta = target_positions.get(ticker, 0) - positions.get(ticker, 0)

                trade_notional = abs(delta) * trade_price

                cost = trade_notional * (
                    self.transaction_cost + self.slippage
                )

                simulated_cash -= delta * trade_price
                simulated_cash -= cost

            if simulated_cash < -self.EPSILON:
                raise RuntimeError(
                    "Backtest attempted leverage beyond cash."
                )

            for ticker in set(positions) | set(target_positions):

                trade_price = self._safe_price(prices.get(ticker))

                current = positions.get(ticker, 0)
                target = target_positions.get(ticker, 0)

                delta = target - current

                if abs(delta) < self.EPSILON:
                    continue

                trade_notional = abs(delta) * trade_price
                cost = trade_notional * (
                    self.transaction_cost + self.slippage
                )

                turnover += trade_notional

                cash -= delta * trade_price
                cash -= cost

                if target == 0:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = target

            equity = cash + sum(
                positions.get(t, 0) * self._safe_price(prices.get(t))
                for t in positions
            )

            if not np.isfinite(equity):
                raise RuntimeError("Invalid equity value produced.")

            if equity <= 0:
                raise RuntimeError("Portfolio capital depleted.")

            equity_curve.append(float(equity))

            prev_prices = prices
            prev_signals = grouped_signals[date]

        curve = np.array(equity_curve, dtype=float)

        returns = (
            np.diff(curve) /
            np.maximum(curve[:-1], self.EPSILON)
        ) if len(curve) > 1 else np.array([0.0])

        periods_per_year = min(252, len(curve))

        sharpe = (
            np.mean(returns) /
            max(np.std(returns), self.EPSILON) *
            np.sqrt(periods_per_year)
        )

        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / np.maximum(peak, self.EPSILON)

        avg_turnover = turnover / max(curve.mean(), self.EPSILON)

        return {
            "final_portfolio": float(curve[-1]),
            "strategy_return": float(curve[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": float(sharpe),
            "avg_turnover": float(avg_turnover),
            "equity_curve": curve.tolist()
        }
