import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger("marketsentinel.portfolio")


class PortfolioBacktestEngine:

    EPSILON = 1e-12

    VOL_WINDOW = 20
    CASH_BUFFER = 0.10

    MAX_INV_VOL = 5.0
    MAX_POSITION_WEIGHT = 0.15
    MAX_GROSS_EXPOSURE = 0.60

    MIN_EQUITY_FLOOR = 0.40

    MIN_CONFIDENCE = 0.60
    MAX_POSITIONS = 3
    LONG_ONLY = True

    # NEW CONTROLS
    MAX_DAILY_TURNOVER = 0.40
    MIN_POSITION_CHANGE = 0.10

    def __init__(
        self,
        transaction_cost=0.0005,
        base_slippage=0.0004,
    ):
        self.transaction_cost = transaction_cost
        self.base_slippage = base_slippage
        self.return_buffers = defaultdict(list)

    ###################################################

    def _reset_state(self):
        self.return_buffers.clear()

    ###################################################
    # VOL TRACKING
    ###################################################

    def _update_volatility(self, prev_prices, prices):

        vols = {}

        for ticker, price in prices.items():

            if prev_prices and ticker in prev_prices:

                prev = prev_prices[ticker]
                if prev <= 0:
                    vols[ticker] = 0.02
                    continue

                ret = price / prev - 1

                buf = self.return_buffers[ticker]
                buf.append(ret)

                if len(buf) > self.VOL_WINDOW:
                    buf.pop(0)

                vol = np.std(buf) if len(buf) > 5 else 0.02
                vols[ticker] = max(vol, 1e-4)

            else:
                vols[ticker] = 0.02

        return vols

    ###################################################
    # WEIGHT ENGINE
    ###################################################

    def _compute_weights(self, signals, vols):

        candidates = []

        for t, decision in signals.items():

            if not isinstance(decision, dict):
                continue

            confidence = float(decision.get("confidence", 0.0))

            if confidence < self.MIN_CONFIDENCE:
                continue

            if decision["signal"] != "BUY":
                continue

            candidates.append((t, confidence))

        if not candidates:
            return {}

        candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )[:self.MAX_POSITIONS]

        raw = {}

        for t, confidence in candidates:

            vol = vols.get(t, 0.02)
            inv_vol = min(1 / max(vol, 1e-4), self.MAX_INV_VOL)

            raw[t] = confidence * inv_vol

        total = sum(raw.values())

        if total < self.EPSILON:
            return {}

        weights = {
            t: min((v / total) * self.MAX_GROSS_EXPOSURE,
                   self.MAX_POSITION_WEIGHT)
            for t, v in raw.items()
        }

        return weights

    ###################################################
    # MAIN BACKTEST
    ###################################################

    def run(self, grouped_prices, grouped_signals, initial_cash=10000):

        logger.info("Portfolio simulation started.")

        self._reset_state()

        window_start_cash = float(initial_cash)

        cash = float(initial_cash)
        positions = {}
        equity_curve = []
        turnover = []

        prev_prices = {}

        for date in sorted(grouped_prices.keys()):

            prices = grouped_prices.get(date, {})
            signals = grouped_signals.get(date, {})

            if not prices:
                continue

            ###################################################
            # MARK TO MARKET
            ###################################################

            portfolio_value = cash

            for t, qty in positions.items():
                price = prices.get(t, prev_prices.get(t))
                if price is None:
                    continue
                portfolio_value += qty * price

            equity_curve.append(float(portfolio_value))

            ###################################################
            # EQUITY FLOOR
            ###################################################

            if portfolio_value < window_start_cash * self.MIN_EQUITY_FLOOR:
                logger.warning("Equity floor breached — stopping simulation.")
                break

            ###################################################
            # TARGET WEIGHTS
            ###################################################

            vols = self._update_volatility(prev_prices, prices)
            weights = self._compute_weights(signals, vols)

            deployable = portfolio_value * (1 - self.CASH_BUFFER)

            target_positions = {
                t: (deployable * w) / prices[t]
                for t, w in weights.items()
                if t in prices and prices[t] > 0
            }

            ###################################################
            # EXECUTE WITH TURNOVER CONTROL
            ###################################################

            day_turnover = 0.0

            for ticker in set(list(positions.keys()) + list(target_positions.keys())):

                current = positions.get(ticker, 0)
                target = target_positions.get(ticker, 0)

                delta = target - current

                if abs(delta) < self.EPSILON:
                    continue

                # Skip tiny rebalance
                if abs(delta) / (abs(current) + self.EPSILON) < self.MIN_POSITION_CHANGE:
                    continue

                price = prices.get(ticker)
                if price is None:
                    continue

                trade_notional = abs(delta) * price

                # Turnover cap
                if (day_turnover + trade_notional) / portfolio_value > self.MAX_DAILY_TURNOVER:
                    continue

                cost = trade_notional * (
                    self.transaction_cost + self.base_slippage
                )

                if delta > 0 and (delta * price + cost) > cash:
                    continue

                cash -= delta * price
                cash -= cost

                if abs(target) < self.EPSILON:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = target

                day_turnover += trade_notional

            turnover.append(
                day_turnover / portfolio_value
                if portfolio_value > self.EPSILON else 0
            )

            prev_prices = prices.copy()

        ###################################################

        curve = np.array(equity_curve, dtype=float)

        if len(curve) < 2:
            return {
                "final_portfolio": window_start_cash,
                "strategy_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "avg_turnover": 0.0,
                "equity_curve": curve.tolist()
            }

        returns = np.diff(curve) / np.maximum(curve[:-1], 1e-12)

        vol = max(np.std(returns), 1e-6)

        sharpe = np.clip(
            np.mean(returns) / vol * np.sqrt(252),
            -5,
            5
        )

        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak

        logger.info("Portfolio simulation finished.")
        logger.info("Max drawdown: %.2f%%", drawdown.min() * 100)

        return {
            "final_portfolio": float(curve[-1]),
            "strategy_return": float(curve[-1] / window_start_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": float(sharpe),
            "avg_turnover": float(np.mean(turnover)),
            "equity_curve": curve.tolist()
        }