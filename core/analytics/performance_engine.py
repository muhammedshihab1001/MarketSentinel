import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    annual_volatility: float
    annual_return: float
    turnover: float
    daily_returns: pd.Series
    equity_curve: pd.Series


class PerformanceEngine:

    TRADING_DAYS = 252

    ########################################################
    # CORE ENTRY
    ########################################################

    def evaluate(
        self,
        portfolio_df: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> PerformanceReport:
        """
        portfolio_df:
            columns: date, ticker, weight

        forward_returns:
            columns: date, ticker, forward_return
            forward_return at date t = return from t -> t+1
        """

        if portfolio_df.empty:
            raise RuntimeError("Portfolio dataframe empty.")

        if forward_returns.empty:
            raise RuntimeError("Forward returns dataframe empty.")

        # Ensure correct dtypes
        portfolio_df = portfolio_df.copy()
        forward_returns = forward_returns.copy()

        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        forward_returns["date"] = pd.to_datetime(forward_returns["date"])

        ########################################################
        # MERGE
        ########################################################

        merged = portfolio_df.merge(
            forward_returns,
            on=["date", "ticker"],
            how="inner"
        )

        if merged.empty:
            raise RuntimeError(
                "No overlapping data for performance evaluation."
            )

        if merged["forward_return"].isnull().any():
            raise RuntimeError("Null forward returns detected after merge.")

        ########################################################
        # DAILY RETURNS
        ########################################################

        merged["weighted_return"] = (
            merged["weight"].astype(float)
            * merged["forward_return"].astype(float)
        )

        daily = (
            merged.groupby("date")["weighted_return"]
            .sum()
            .sort_index()
        )

        if len(daily) < 2:
            raise RuntimeError("Insufficient daily returns for evaluation.")

        ########################################################
        # EQUITY CURVE
        ########################################################

        equity = (1 + daily).cumprod()

        cumulative_return = equity.iloc[-1] - 1

        sharpe = self._sharpe_ratio(daily)
        max_dd = self._max_drawdown(equity)
        hit = self._hit_rate(daily)
        ann_vol = self._annual_volatility(daily)
        ann_ret = self._annual_return(daily)
        turnover = self._turnover(portfolio_df)

        return PerformanceReport(
            cumulative_return=float(cumulative_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            hit_rate=float(hit),
            annual_volatility=float(ann_vol),
            annual_return=float(ann_ret),
            turnover=float(turnover),
            daily_returns=daily,
            equity_curve=equity
        )

    ########################################################
    # METRICS
    ########################################################

    def _sharpe_ratio(self, daily_returns):

        mean = daily_returns.mean()
        std = daily_returns.std()

        if std == 0:
            return 0.0

        return (mean / std) * np.sqrt(self.TRADING_DAYS)

    def _annual_volatility(self, daily_returns):
        return daily_returns.std() * np.sqrt(self.TRADING_DAYS)

    def _annual_return(self, daily_returns):

        cumulative = (1 + daily_returns).prod()
        years = len(daily_returns) / self.TRADING_DAYS

        if years <= 0:
            return 0.0

        return cumulative ** (1 / years) - 1

    def _max_drawdown(self, equity_curve):

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max

        return drawdown.min()

    def _hit_rate(self, daily_returns):
        return float((daily_returns > 0).mean())

    def _turnover(self, portfolio_df):

        pivot = (
            portfolio_df
            .pivot(index="date", columns="ticker", values="weight")
            .fillna(0)
            .sort_index()
        )

        diff = pivot.diff().abs()

        # institutional turnover definition
        turnover = 0.5 * diff.sum(axis=1)

        turnover = turnover.iloc[1:]  # drop first NaN day

        return float(turnover.mean())