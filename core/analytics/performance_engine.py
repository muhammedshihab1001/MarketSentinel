import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PerformanceReport:
    cumulative_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    hit_rate: float
    annual_volatility: float
    annual_return: float
    turnover: float
    beta: Optional[float]
    information_ratio: Optional[float]
    daily_returns: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    rolling_sharpe: pd.Series
    rolling_volatility: pd.Series


class PerformanceEngine:

    TRADING_DAYS = 252
    ROLLING_WINDOW = 63  # 3-month rolling window

    ########################################################
    # CORE ENTRY
    ########################################################

    def evaluate(
        self,
        portfolio_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceReport:

        if portfolio_df.empty:
            raise RuntimeError("Portfolio dataframe empty.")

        if forward_returns.empty:
            raise RuntimeError("Forward returns dataframe empty.")

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

        ########################################################
        # DRAWDOWN SERIES
        ########################################################

        rolling_max = equity.cummax()
        drawdown_series = (equity - rolling_max) / rolling_max

        ########################################################
        # METRICS
        ########################################################

        sharpe = self._sharpe_ratio(daily)
        sortino = self._sortino_ratio(daily)
        max_dd = drawdown_series.min()
        max_dd_duration = self._max_drawdown_duration(equity)
        ann_vol = self._annual_volatility(daily)
        ann_ret = self._annual_return(daily)
        calmar = self._calmar_ratio(ann_ret, max_dd)
        hit = self._hit_rate(daily)
        turnover = self._turnover(portfolio_df)
        beta = self._beta(daily, benchmark_returns)
        info_ratio = self._information_ratio(daily, benchmark_returns)

        ########################################################
        # ROLLING METRICS
        ########################################################

        rolling_sharpe = self._rolling_sharpe(daily)
        rolling_volatility = self._rolling_volatility(daily)

        return PerformanceReport(
            cumulative_return=float(cumulative_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_drawdown=float(max_dd),
            max_drawdown_duration=int(max_dd_duration),
            hit_rate=float(hit),
            annual_volatility=float(ann_vol),
            annual_return=float(ann_ret),
            turnover=float(turnover),
            beta=beta,
            information_ratio=info_ratio,
            daily_returns=daily,
            equity_curve=equity,
            drawdown_series=drawdown_series,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_volatility
        )

    ########################################################
    # METRICS
    ########################################################

    def _sharpe_ratio(self, daily_returns):

        std = daily_returns.std()

        if std == 0 or np.isnan(std):
            return 0.0

        return (daily_returns.mean() / std) * np.sqrt(self.TRADING_DAYS)

    def _sortino_ratio(self, daily_returns):

        downside = daily_returns[daily_returns < 0]

        if len(downside) == 0:
            return 0.0

        downside_std = downside.std()

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        return (daily_returns.mean() / downside_std) * np.sqrt(self.TRADING_DAYS)

    def _annual_volatility(self, daily_returns):
        return daily_returns.std() * np.sqrt(self.TRADING_DAYS)

    def _annual_return(self, daily_returns):

        cumulative = (1 + daily_returns).prod()
        years = len(daily_returns) / self.TRADING_DAYS

        if years <= 0:
            return 0.0

        return cumulative ** (1 / years) - 1

    def _max_drawdown_duration(self, equity_curve):

        rolling_max = equity_curve.cummax()
        drawdown = equity_curve < rolling_max

        duration = 0
        max_duration = 0

        for val in drawdown:
            if val:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        return max_duration

    def _calmar_ratio(self, annual_return, max_drawdown):

        if max_drawdown == 0:
            return 0.0

        return annual_return / abs(max_drawdown)

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
        turnover = 0.5 * diff.sum(axis=1)
        turnover = turnover.iloc[1:]

        return float(turnover.mean())

    def _beta(self, strategy_returns, benchmark_returns):

        if benchmark_returns is None:
            return None

        aligned = strategy_returns.align(benchmark_returns, join="inner")[0]
        bench = benchmark_returns.align(strategy_returns, join="inner")[0]

        if len(aligned) < 2:
            return None

        cov = np.cov(aligned, bench)[0][1]
        var = np.var(bench)

        if var == 0:
            return None

        return float(cov / var)

    def _information_ratio(self, strategy_returns, benchmark_returns):

        if benchmark_returns is None:
            return None

        aligned_strategy, aligned_bench = strategy_returns.align(
            benchmark_returns,
            join="inner"
        )

        active = aligned_strategy - aligned_bench

        if active.std() == 0:
            return None

        return float(
            (active.mean() / active.std()) * np.sqrt(self.TRADING_DAYS)
        )

    ########################################################
    # ROLLING METRICS
    ########################################################

    def _rolling_sharpe(self, daily_returns):

        rolling = daily_returns.rolling(self.ROLLING_WINDOW)

        return (
            rolling.mean() /
            rolling.std().replace(0, np.nan)
        ) * np.sqrt(self.TRADING_DAYS)

    def _rolling_volatility(self, daily_returns):

        return daily_returns.rolling(
            self.ROLLING_WINDOW
        ).std() * np.sqrt(self.TRADING_DAYS)