import numpy as np
from collections import defaultdict


class PortfolioBacktestEngine:

    EPSILON = 1e-12
    VOL_WINDOW = 20
    CASH_BUFFER = 0.02
    MAX_INV_VOL = 5.0

    MAX_SINGLE_STEP_RETURN = 0.40
    MAX_GAP = 0.35

    MAX_POSITION_WEIGHT = 0.12
    MAX_ABSOLUTE_POSITION = 0.25
    MAX_TURNOVER_RATIO = 1.8
    MIN_EQUITY_FLOOR = 0.35
    SLIPPAGE_IMPACT = 0.08

    BORROW_COST_ANNUAL = 0.03

    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

    def __init__(
        self,
        transaction_cost=0.001,
        base_slippage=0.0005,
        target_vol=0.02,
        max_gross_exposure=1.0
    ):
        self.transaction_cost = transaction_cost
        self.base_slippage = base_slippage
        self.target_vol = target_vol
        self.max_gross_exposure = max_gross_exposure

        self.return_buffers = defaultdict(list)

    ###################################################

    def _reset_state(self):
        self.return_buffers = defaultdict(list)

    ###################################################

    def _safe_price(self, price):

        try:
            price = float(price)
        except Exception:
            return None

        if not np.isfinite(price) or price <= 0:
            return None

        return price

    ###################################################
    # GAP GUARD
    ###################################################

    def _check_gap(self, prev_prices, prices):

        if prev_prices is None:
            return

        for t in prices:

            if t not in prev_prices:
                continue

            gap = abs(prices[t] / prev_prices[t] - 1)

            if gap > self.MAX_GAP:
                raise RuntimeError(
                    f"Untradeable gap detected in {t}: {gap:.2%}"
                )

    ###################################################

    def _clean_prices_and_signals(self, prices, signals):

        clean_prices = {}
        clean_signals = {}

        for ticker, price in prices.items():

            safe = self._safe_price(price)

            if safe is None:
                continue

            if ticker not in signals:
                continue

            if signals[ticker] not in self.VALID_SIGNALS:
                continue

            clean_prices[ticker] = safe
            clean_signals[ticker] = signals[ticker]

        return clean_prices, clean_signals

    ###################################################

    def _update_vol_buffers(self, prev_prices, prices):

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
    # HARD EXPOSURE CHECK
    ###################################################

    def _gross_exposure(self, positions, prices, equity):

        exposure = sum(
            abs(shares * prices.get(t, 0))
            for t, shares in positions.items()
        )

        return exposure / max(equity, 1)

    ###################################################

    def _compute_weights(self, signals, vols):

        raw = {}

        for ticker, signal in signals.items():

            if signal == "HOLD":
                continue

            direction = 1 if signal == "BUY" else -1

            vol = vols.get(ticker, self.target_vol)

            inv_vol = min(
                1 / max(vol, 1e-4),
                self.MAX_INV_VOL
            )

            raw[ticker] = direction * inv_vol

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

        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    ###################################################

    def _trade_slippage(self, notional, portfolio_value):

        size_ratio = min(notional / max(portfolio_value, 1), 0.25)
        impact = size_ratio * self.SLIPPAGE_IMPACT

        return self.base_slippage + impact

    ###################################################

    def _apply_borrow_cost(self, positions, prices):

        daily_rate = self.BORROW_COST_ANNUAL / 252

        borrow_cost = 0.0

        for t, shares in positions.items():

            if shares >= 0:
                continue

            borrow_cost += abs(shares * prices[t]) * daily_rate

        return borrow_cost

    ###################################################
    # 🔥 NEW — FORCE LIQUIDATION
    ###################################################

    def _force_liquidations(self, positions, target_positions):
        """
        If a ticker disappears from targets,
        close the position.
        """

        for ticker in list(positions.keys()):
            if ticker not in target_positions:
                target_positions[ticker] = 0.0

    ###################################################

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

            prices_raw = grouped_prices[date]
            signals_raw = grouped_signals[date]

            prices, signals_today = self._clean_prices_and_signals(
                prices_raw,
                signals_raw
            )

            if not prices:
                continue

            self._check_gap(prev_prices, prices)

            if prev_prices is None:
                prev_prices = prices
                prev_signals = signals_today
                equity_curve.append(cash)
                continue

            signals = prev_signals

            vols = self._update_vol_buffers(prev_prices, prices)

            portfolio_value = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
                if t in prices
            )

            if portfolio_value < initial_cash * self.MIN_EQUITY_FLOOR:
                raise RuntimeError("Equity breached institutional floor.")

            weights = self._compute_weights(signals, vols)

            deployable_capital = portfolio_value * (1 - self.CASH_BUFFER)

            target_positions = {
                t: (deployable_capital * w) / prices[t]
                for t, w in weights.items()
            }

            # FORCE CLOSE OLD POSITIONS
            self._force_liquidations(positions, target_positions)

            ###################################################
            # EXECUTE TRADES
            ###################################################

            for ticker in set(positions) | set(target_positions):

                if ticker not in prices:
                    continue

                trade_price = prices[ticker]

                current = positions.get(ticker, 0)
                target = target_positions.get(ticker, 0)

                delta = target - current

                if abs(delta) < self.EPSILON:
                    continue

                trade_notional = abs(delta) * trade_price

                slippage = self._trade_slippage(
                    trade_notional,
                    portfolio_value
                )

                cost = trade_notional * (
                    self.transaction_cost + slippage
                )

                if delta > 0 and (delta * trade_price + cost) > cash:
                    continue

                turnover += trade_notional / max(portfolio_value, 1)

                cash -= delta * trade_price
                cash -= cost

                if abs(target) < self.EPSILON:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = target

            ###################################################
            # HARD GROSS CHECK
            ###################################################

            equity = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            gross = self._gross_exposure(
                positions,
                prices,
                equity
            )

            if gross > self.max_gross_exposure * 1.05:
                raise RuntimeError(
                    "Gross exposure breach detected."
                )

            ###################################################

            borrow_cost = self._apply_borrow_cost(
                positions,
                prices
            )

            cash -= borrow_cost

            equity = cash + sum(
                positions.get(t, 0) * prices.get(t, 0)
                for t in positions
            )

            if equity_curve:

                step_return = equity / equity_curve[-1] - 1

                if abs(step_return) > self.MAX_SINGLE_STEP_RETURN:
                    raise RuntimeError(
                        "Unrealistic portfolio jump detected."
                    )

            equity_curve.append(float(equity))

            prev_prices = prices
            prev_signals = signals_today

        curve = np.array(equity_curve, dtype=float)

        returns = (
            np.diff(curve) /
            np.maximum(curve[:-1], self.EPSILON)
        ) if len(curve) > 1 else np.array([0.0])

        vol = max(np.std(returns), 1e-6)

        sharpe = np.clip(
            np.mean(returns) / vol * np.sqrt(252),
            -10,
            10
        )

        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / np.maximum(peak, self.EPSILON)

        avg_turnover = turnover / max(len(curve), 1)

        if avg_turnover > self.MAX_TURNOVER_RATIO:
            raise RuntimeError(
                "Turnover exceeds institutional threshold."
            )

        return {
            "final_portfolio": float(curve[-1]),
            "strategy_return": float(curve[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": float(sharpe),
            "avg_turnover": float(avg_turnover),
            "equity_curve": curve.tolist()
        }
