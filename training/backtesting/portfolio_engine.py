import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger("marketsentinel.portfolio")


class PortfolioBacktestEngine:

    EPSILON = 1e-12

    VOL_WINDOW = 20
    CASH_BUFFER = 0.08

    MAX_INV_VOL = 3.0
    MAX_POSITION_WEIGHT = 0.05
    MAX_GROSS_EXPOSURE = 0.70

    MIN_EQUITY_FLOOR = 0.60

    VALID_SIGNALS = {"BUY", "SELL"}

    ###################################################

    def __init__(
        self,
        transaction_cost=0.001,
        base_slippage=0.0006,
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

        for ticker in prices:

            if prev_prices and ticker in prev_prices:

                ret = prices[ticker] / prev_prices[ticker] - 1

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

        raw = {}

        for t, s in signals.items():

            if s not in self.VALID_SIGNALS:
                continue

            direction = 1 if s == "BUY" else -1
            vol = vols.get(t, 0.02)

            inv_vol = min(1 / max(vol, 1e-4), self.MAX_INV_VOL)

            raw[t] = direction * inv_vol

        if not raw:
            return {}

        total = sum(abs(v) for v in raw.values())

        weights = {
            t: np.clip(
                v / total,
                -self.MAX_POSITION_WEIGHT,
                self.MAX_POSITION_WEIGHT
            )
            for t, v in raw.items()
        }

        gross = sum(abs(w) for w in weights.values())

        if gross > self.MAX_GROSS_EXPOSURE:
            scale = self.MAX_GROSS_EXPOSURE / gross
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    ###################################################
    # MAIN BACKTEST
    ###################################################

    def run(self, grouped_prices, grouped_signals, initial_cash=10000):

        logger.info("Portfolio simulation started.")

        self._reset_state()

        cash = float(initial_cash)
        positions = {}
        equity_curve = []
        turnover = []

        prev_prices = None

        for date in sorted(grouped_prices.keys()):

            prices = grouped_prices[date]
            signals = grouped_signals[date]

            if not prices:
                continue

            ###################################################
            # 1️⃣ MARK TO MARKET USING PREVIOUS POSITIONS
            ###################################################

            portfolio_value = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
                if t in prices
            )

            equity_curve.append(float(portfolio_value))

            if portfolio_value < initial_cash * self.MIN_EQUITY_FLOOR:
                logger.warning("Equity floor breached — stopping simulation.")
                break

            ###################################################
            # 2️⃣ COMPUTE TARGET WEIGHTS
            ###################################################

            vols = self._update_volatility(prev_prices, prices)

            weights = self._compute_weights(signals, vols)

            deployable = portfolio_value * (1 - self.CASH_BUFFER)

            target_positions = {
                t: (deployable * w) / prices[t]
                for t, w in weights.items()
                if t in prices
            }

            ###################################################
            # 3️⃣ FORCE LIQUIDATION OF REMOVED POSITIONS
            ###################################################

            for t in list(positions.keys()):
                if t not in target_positions:
                    delta = -positions[t]
                    trade_notional = abs(delta) * prices.get(t, 0)

                    cost = trade_notional * (
                        self.transaction_cost + self.base_slippage
                    )

                    cash -= delta * prices.get(t, 0)
                    cash -= cost

                    positions.pop(t)

            ###################################################
            # 4️⃣ EXECUTE NEW TARGETS
            ###################################################

            day_turnover = 0.0

            for ticker, target in target_positions.items():

                current = positions.get(ticker, 0)
                delta = target - current

                if abs(delta) < self.EPSILON:
                    continue

                trade_notional = abs(delta) * prices[ticker]
                cost = trade_notional * (
                    self.transaction_cost + self.base_slippage
                )

                if delta > 0 and (delta * prices[ticker] + cost) > cash:
                    continue

                cash -= delta * prices[ticker]
                cash -= cost

                positions[ticker] = current + delta

                day_turnover += trade_notional

            turnover.append(day_turnover / portfolio_value if portfolio_value > 0 else 0)

            prev_prices = prices

        ###################################################

        curve = np.array(equity_curve, dtype=float)

        if len(curve) < 2:
            return {
                "final_portfolio": initial_cash,
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
            -3,
            3
        )

        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak

        logger.info("Portfolio simulation finished.")
        logger.info("Max drawdown: %.2f%%", drawdown.min() * 100)

        return {
            "final_portfolio": float(curve[-1]),
            "strategy_return": float(curve[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": float(sharpe),
            "avg_turnover": float(np.mean(turnover)),
            "equity_curve": curve.tolist()
        }
