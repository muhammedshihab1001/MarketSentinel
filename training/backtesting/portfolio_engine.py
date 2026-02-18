import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger("marketsentinel.portfolio")


class PortfolioBacktestEngine:

    EPSILON = 1e-12

    VOL_WINDOW = 20
    CASH_BUFFER = 0.08

    MAX_INV_VOL = 3.0
    MAX_POSITION_WEIGHT = 0.06
    MAX_GROSS_EXPOSURE = 0.75

    MAX_SINGLE_STEP_RETURN = 0.20
    MAX_GAP = 0.20

    MAX_TURNOVER_RATIO = 1.0
    MIN_EQUITY_FLOOR = 0.60  # Hard kill switch

    VOL_KILL_SWITCH = 0.07
    DRAWDOWN_REDUCTION_START = -0.20
    DRAWDOWN_MAX_REDUCTION = -0.40

    SLIPPAGE_IMPACT = 0.10
    BORROW_COST_ANNUAL = 0.02

    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

    ###################################################

    def __init__(
        self,
        transaction_cost=0.001,
        base_slippage=0.0006,
        target_vol=0.015
    ):
        self.transaction_cost = transaction_cost
        self.base_slippage = base_slippage
        self.target_vol = target_vol

        self.return_buffers = defaultdict(list)
        self.price_history = defaultdict(list)

    ###################################################

    def _reset_state(self):
        self.return_buffers.clear()
        self.price_history.clear()

    ###################################################
    # VOL TRACKING
    ###################################################

    def _update_buffers(self, prev_prices, prices):

        vols = {}

        for ticker in prices:

            if prev_prices and ticker in prev_prices:

                ret = prices[ticker] / prev_prices[ticker] - 1

                buf = self.return_buffers[ticker]
                buf.append(ret)

                if len(buf) > self.VOL_WINDOW:
                    buf.pop(0)

                vol = np.std(buf) if len(buf) > 5 else self.target_vol
                vols[ticker] = max(vol, 1e-4)

            else:
                vols[ticker] = self.target_vol

        return vols

    ###################################################
    # VOL TARGET SCALER
    ###################################################

    def _volatility_scaler(self, portfolio_returns):

        if len(portfolio_returns) < 10:
            return 1.0

        realized = np.std(portfolio_returns)

        if realized < 1e-6:
            return 1.0

        scale = self.target_vol / realized

        return float(np.clip(scale, 0.5, 1.5))

    ###################################################
    # DRAWDOWN CONTROL
    ###################################################

    def _drawdown_scaler(self, curve):

        peak = np.maximum.accumulate(curve)
        dd = (curve[-1] - peak[-1]) / peak[-1]

        if dd > self.DRAWDOWN_REDUCTION_START:
            return 1.0

        if dd < self.DRAWDOWN_MAX_REDUCTION:
            return 0.4

        # linear scaling
        span = self.DRAWDOWN_MAX_REDUCTION - self.DRAWDOWN_REDUCTION_START
        pos = dd - self.DRAWDOWN_REDUCTION_START
        return 1.0 + (pos / span) * (0.4 - 1.0)

    ###################################################
    # WEIGHT ENGINE
    ###################################################

    def _compute_weights(self, signals, vols):

        raw = {}

        for t, s in signals.items():

            if s == "HOLD":
                continue

            direction = 1 if s == "BUY" else -1
            vol = vols.get(t, self.target_vol)

            inv_vol = min(1 / max(vol, 1e-4), self.MAX_INV_VOL)
            raw[t] = direction * inv_vol

        if not raw:
            return {}

        total = sum(abs(v) for v in raw.values())

        weights = {
            t: np.clip(v / total, -self.MAX_POSITION_WEIGHT, self.MAX_POSITION_WEIGHT)
            for t, v in raw.items()
        }

        gross = sum(abs(w) for w in weights.values())

        if gross > self.MAX_GROSS_EXPOSURE:
            scale = self.MAX_GROSS_EXPOSURE / gross
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    ###################################################
    # EXECUTION
    ###################################################

    def run(self, grouped_prices, grouped_signals, initial_cash=10000):

        logger.info("Portfolio simulation started.")

        self._reset_state()

        cash = float(initial_cash)
        positions = {}
        equity_curve = []

        prev_prices = None
        portfolio_returns = []

        for date in sorted(grouped_prices.keys()):

            prices = grouped_prices[date]
            signals = grouped_signals[date]

            if not prices:
                continue

            if prev_prices is None:
                prev_prices = prices
                equity_curve.append(cash)
                continue

            vols = self._update_buffers(prev_prices, prices)

            portfolio_value = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            # HARD FLOOR STOP
            if portfolio_value < initial_cash * self.MIN_EQUITY_FLOOR:
                logger.warning("Equity floor breached — stopping simulation.")
                break

            weights = self._compute_weights(signals, vols)

            deployable = portfolio_value * (1 - self.CASH_BUFFER)

            target_positions = {
                t: (deployable * w) / prices[t]
                for t, w in weights.items()
                if t in prices
            }

            ###################################################
            # TRADE EXECUTION
            ###################################################

            for ticker in set(positions) | set(target_positions):

                if ticker not in prices:
                    continue

                current = positions.get(ticker, 0)
                target = target_positions.get(ticker, 0)

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

                new_position = current + delta

                if abs(new_position) < self.EPSILON:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = new_position

            equity = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            equity_curve.append(float(equity))

            if len(equity_curve) > 1:
                r = (equity_curve[-1] / equity_curve[-2]) - 1
                portfolio_returns.append(r)

            # APPLY VOL SCALING
            vol_scale = self._volatility_scaler(portfolio_returns)
            dd_scale = self._drawdown_scaler(np.array(equity_curve))

            exposure_scale = vol_scale * dd_scale

            # scale positions
            for t in positions:
                positions[t] *= exposure_scale

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

        returns = np.diff(curve) / np.maximum(curve[:-1], self.EPSILON)

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
            "avg_turnover": 0.0,
            "equity_curve": curve.tolist()
        }
