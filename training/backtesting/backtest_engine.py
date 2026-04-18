import numpy as np


class BacktestEngine:

    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

    MIN_CAPITAL = 1e-6
    MAX_DRAWDOWN_KILL = -0.70
    MAX_SHARPE = 5.0

    MIN_HOLD_BARS = 1
    REENTRY_COOLDOWN = 0

    MAX_POSITION_SIZE = 0.30
    MIN_POSITION_SIZE = 0.05
    MAX_SINGLE_BAR_RETURN = 0.40
    MAX_GAP = 0.35
    MAX_TURNOVER = 12.0

    VOL_TARGET = 0.02
    EPSILON = 1e-12

    MAX_BACKTEST_LENGTH = 1_000_000

    ############################################################

    def _validate_inputs(self, prices, signals, position_size):

        prices = np.asarray(prices, dtype=float)
        signals = list(signals)

        if not (0 < position_size <= self.MAX_POSITION_SIZE):
            raise RuntimeError(
                f"position_size must be within (0, {self.MAX_POSITION_SIZE}]"
            )

        if len(prices) != len(signals):
            raise RuntimeError("Prices and signals length mismatch.")

        if len(prices) > self.MAX_BACKTEST_LENGTH:
            raise RuntimeError("Backtest length exceeds safety limit.")

        if len(prices) < 2:
            return prices, signals

        if not np.isfinite(prices).all():
            raise RuntimeError("Non-finite prices detected.")

        if (prices <= 0).any():
            raise RuntimeError("Invalid prices detected.")

        unknown = set(signals) - self.VALID_SIGNALS
        if unknown:
            raise RuntimeError(f"Unknown signals detected: {unknown}")

        return prices, signals

    ############################################################

    def _gap_ok(self, prev_price, price):
        gap = abs(price / prev_price - 1)
        return gap <= self.MAX_GAP

    ############################################################

    def _dynamic_position_size(self, base_size, equity, initial_cash):

        growth_factor = equity / initial_cash

        scaled = base_size * np.clip(growth_factor, 0.7, 2.0)

        scaled = np.clip(scaled, self.MIN_POSITION_SIZE, self.MAX_POSITION_SIZE)

        return float(scaled)

    ############################################################

    def _sanitize_prices(self, prices):

        prices = np.asarray(prices, dtype=float)

        returns = np.diff(prices) / prices[:-1]

        mask = np.abs(returns) > self.MAX_SINGLE_BAR_RETURN

        if mask.any():

            returns[mask] = np.sign(returns[mask]) * self.MAX_SINGLE_BAR_RETURN

            prices = prices[0] * np.cumprod(np.concatenate([[1], 1 + returns]))

        return prices

    ############################################################

    def run(
        self,
        prices,
        signals,
        initial_cash=10_000,
        transaction_cost=0.0005,
        slippage=0.0003,
        position_size=0.25,
    ):

        prices, signals = self._validate_inputs(prices, signals, position_size)

        prices = self._sanitize_prices(prices)

        prices = np.asarray(prices, dtype=np.float64)

        if len(prices) < 2:
            return self._empty_result(initial_cash)

        cash = float(initial_cash)
        position = 0.0

        portfolio_values = []

        trade_count = 0
        time_in_market = 0
        capital_rotated = 0.0

        hold_bars = 0
        cooldown = 0

        peak_equity = initial_cash

        prev_price = prices[0]

        ####################################################
        # MAIN LOOP
        ####################################################

        for i in range(1, len(prices)):

            price = float(prices[i])
            signal = signals[i - 1]

            if not self._gap_ok(prev_price, price):
                signal = "HOLD"

            current_equity = cash + position * price

            if position > 0:
                hold_bars += 1

            if cooldown > 0:
                cooldown -= 1

            ################################################
            # BUY
            ################################################

            if (
                signal == "BUY"
                and cash > self.MIN_CAPITAL
                and position == 0
                and cooldown == 0
            ):

                dynamic_size = self._dynamic_position_size(
                    position_size, equity=current_equity, initial_cash=initial_cash
                )

                execution_price = max(price * (1 + slippage), self.EPSILON)

                deploy_cash = cash * dynamic_size

                shares = (deploy_cash * (1 - transaction_cost)) / execution_price

                position = float(shares)
                cash -= deploy_cash
                cash = max(cash, 0.0)

                capital_rotated += deploy_cash
                trade_count += 1
                hold_bars = 0

            ################################################
            # SELL
            ################################################

            elif signal == "SELL" and position > 0 and hold_bars >= self.MIN_HOLD_BARS:

                execution_price = max(price * (1 - slippage), self.EPSILON)

                proceeds = position * execution_price * (1 - transaction_cost)

                capital_rotated += proceeds

                cash += proceeds
                position = 0.0

                trade_count += 1
                cooldown = self.REENTRY_COOLDOWN

            ################################################
            # PORTFOLIO VALUE
            ################################################

            portfolio_value = cash + position * price

            if portfolio_value <= self.MIN_CAPITAL:
                portfolio_values.append(self.MIN_CAPITAL)
                break

            peak_equity = max(peak_equity, portfolio_value)

            drawdown = (portfolio_value - peak_equity) / peak_equity

            if drawdown < self.MAX_DRAWDOWN_KILL:
                portfolio_values.append(portfolio_value)
                break

            if portfolio_values:

                step_return = (
                    portfolio_value / (portfolio_values[-1] + self.EPSILON) - 1
                )

                step_return = np.clip(
                    step_return, -self.MAX_SINGLE_BAR_RETURN, self.MAX_SINGLE_BAR_RETURN
                )

            if position > 0:
                time_in_market += 1

            portfolio_values.append(float(portfolio_value))

            prev_price = price

        ####################################################
        # FORCE LIQUIDATION
        ####################################################

        if position > 0 and portfolio_values:

            final_price = prices[-1]
            liquidation_price = max(final_price * (1 - slippage), self.EPSILON)

            proceeds = position * liquidation_price * (1 - transaction_cost)

            cash += proceeds
            position = 0.0

            portfolio_values[-1] = float(cash)

            trade_count += 1

        if not portfolio_values:
            return self._empty_result(initial_cash)

        portfolio_values = np.array(portfolio_values, dtype=np.float64)

        portfolio_values = np.maximum(portfolio_values, self.MIN_CAPITAL)

        ####################################################
        # PERFORMANCE METRICS
        ####################################################

        strategy_return = portfolio_values[-1] / initial_cash - 1

        buy_hold_return = prices[-1] / prices[0] - 1

        alpha = strategy_return - buy_hold_return

        returns = np.diff(portfolio_values) / (portfolio_values[:-1] + self.EPSILON)

        returns = returns[np.isfinite(returns)]

        if len(returns) > 1:

            std = np.std(returns)

            sharpe = np.mean(returns) / std * np.sqrt(252) if std > 0 else 0.0

            sharpe = float(np.clip(sharpe, -self.MAX_SHARPE, self.MAX_SHARPE))

        else:

            sharpe = 0.0

        exposure = time_in_market / max(len(portfolio_values) - 1, 1)

        turnover = capital_rotated / initial_cash if initial_cash > 0 else 0.0

        if turnover > self.MAX_TURNOVER:
            raise RuntimeError("Turnover exceeds institutional threshold.")

        return {
            "final_portfolio": float(portfolio_values[-1]),
            "strategy_return": float(strategy_return),
            "buy_hold_return": float(buy_hold_return),
            "alpha": float(alpha),
            "sharpe_ratio": float(sharpe),
            "trade_count": trade_count,
            "exposure": float(exposure),
            "turnover": float(turnover),
            "equity_curve": portfolio_values.tolist(),
        }

    ############################################################

    def _empty_result(self, initial_cash):

        return {
            "final_portfolio": initial_cash,
            "strategy_return": 0.0,
            "buy_hold_return": 0.0,
            "alpha": 0.0,
            "sharpe_ratio": 0.0,
            "trade_count": 0,
            "exposure": 0.0,
            "turnover": 0.0,
            "equity_curve": [initial_cash],
        }
