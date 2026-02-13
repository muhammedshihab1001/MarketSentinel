import numpy as np


class BacktestEngine:

    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

    MIN_CAPITAL = 1e-6
    MAX_DRAWDOWN_KILL = -0.70
    MAX_SHARPE = 5.0

    MIN_HOLD_BARS = 2
    REENTRY_COOLDOWN = 1

    ############################################################

    def _validate_inputs(self, prices, signals, position_size):

        if position_size <= 0 or position_size > 1.0:
            raise RuntimeError(
                "position_size must be within (0, 1]."
            )

        if len(prices) != len(signals):
            raise RuntimeError("Prices and signals length mismatch.")

        if len(prices) < 2:
            return

        prices = np.asarray(prices, dtype=float)

        if not np.isfinite(prices).all():
            raise RuntimeError("Non-finite prices detected.")

        if (prices <= 0).any():
            raise RuntimeError("Invalid prices detected.")

        unknown = set(signals) - self.VALID_SIGNALS
        if unknown:
            raise RuntimeError(f"Unknown signals detected: {unknown}")

    ############################################################

    def run(
        self,
        prices,
        signals,
        initial_cash=10_000,
        transaction_cost=0.001,
        slippage=0.0005,
        position_size=1.0
    ):

        self._validate_inputs(prices, signals, position_size)

        prices = np.asarray(prices, dtype=float)
        signals = list(signals)

        if len(prices) < 2:
            return self._empty_result(initial_cash)

        cash = float(initial_cash)
        position = 0.0

        portfolio_values = []

        trade_count = 0
        time_in_market = 0
        capital_rotated = 0.0

        prev_signal = "HOLD"

        hold_bars = 0
        cooldown = 0

        peak_equity = initial_cash

        ####################################################

        for i in range(len(prices)):

            price = prices[i]

            ####################################################
            # EXECUTE PREVIOUS SIGNAL
            ####################################################

            if position > 0:
                hold_bars += 1

            if cooldown > 0:
                cooldown -= 1

            if (
                prev_signal == "BUY"
                and cash > self.MIN_CAPITAL
                and position == 0
                and cooldown == 0
            ):

                execution_price = price * (1 + slippage)

                deploy_cash = cash * position_size

                if deploy_cash > self.MIN_CAPITAL:

                    shares = (
                        deploy_cash * (1 - transaction_cost)
                    ) / execution_price

                    position = shares
                    cash -= deploy_cash

                    capital_rotated += deploy_cash
                    trade_count += 1
                    hold_bars = 0

            elif (
                prev_signal == "SELL"
                and position > 0
                and hold_bars >= self.MIN_HOLD_BARS
            ):

                execution_price = price * (1 - slippage)

                proceeds = (
                    position
                    * execution_price
                    * (1 - transaction_cost)
                )

                capital_rotated += proceeds

                cash += proceeds
                position = 0

                trade_count += 1
                cooldown = self.REENTRY_COOLDOWN

            ####################################################
            # EQUITY
            ####################################################

            portfolio_value = cash + position * price

            if portfolio_value <= self.MIN_CAPITAL:
                portfolio_values.append(self.MIN_CAPITAL)
                break

            peak_equity = max(peak_equity, portfolio_value)

            drawdown = (
                portfolio_value - peak_equity
            ) / peak_equity

            if drawdown < self.MAX_DRAWDOWN_KILL:
                portfolio_values.append(portfolio_value)
                break

            if position > 0:
                time_in_market += 1

            portfolio_values.append(portfolio_value)

            prev_signal = signals[i]

        ####################################################
        # FORCE LIQUIDATION
        ####################################################

        if position > 0:

            final_price = prices[len(portfolio_values) - 1] * (1 - slippage)

            cash += position * final_price * (1 - transaction_cost)
            position = 0

            portfolio_values[-1] = cash
            trade_count += 1

        ####################################################

        portfolio_values = np.array(portfolio_values, dtype=float)
        portfolio_values = np.maximum(portfolio_values, self.MIN_CAPITAL)

        strategy_return = portfolio_values[-1] / initial_cash - 1
        buy_hold_return = prices[-1] / prices[0] - 1
        alpha = strategy_return - buy_hold_return

        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) > 1:

            std = np.std(returns)

            sharpe = (
                np.mean(returns) / std * np.sqrt(252)
                if std > 0 else 0.0
            )

            sharpe = float(np.clip(sharpe, -self.MAX_SHARPE, self.MAX_SHARPE))

        else:
            sharpe = 0.0

        exposure = time_in_market / len(portfolio_values)

        turnover = (
            capital_rotated / initial_cash
            if initial_cash > 0 else 0.0
        )

        return {
            "final_portfolio": float(portfolio_values[-1]),
            "strategy_return": float(strategy_return),
            "buy_hold_return": float(buy_hold_return),
            "alpha": float(alpha),
            "sharpe_ratio": float(sharpe),
            "trade_count": trade_count,
            "exposure": float(exposure),
            "turnover": float(turnover),
            "equity_curve": portfolio_values.tolist()
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
            "equity_curve": [initial_cash]
        }
