import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any


EPSILON = 1e-12


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cumulative_return": float(self.cumulative_return),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "max_drawdown": float(self.max_drawdown),
            "max_drawdown_duration": int(self.max_drawdown_duration),
            "hit_rate": float(self.hit_rate),
            "annual_volatility": float(self.annual_volatility),
            "annual_return": float(self.annual_return),
            "turnover": float(self.turnover),
            "beta": None if self.beta is None else float(self.beta),
            "information_ratio": None if self.information_ratio is None else float(self.information_ratio),
        }


class PerformanceEngine:

    TRADING_DAYS = 252
    ROLLING_WINDOW = 63

    ########################################################
    # CORE ENTRY
    ########################################################

    def evaluate(
        self,
        portfolio_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceReport:

        if portfolio_df.empty or forward_returns.empty:
            raise RuntimeError("Insufficient data for evaluation.")

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
            raise RuntimeError("No overlapping data for evaluation.")

        merged = merged.dropna(subset=["forward_return", "weight"])

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

        daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if len(daily) < 2:
            raise RuntimeError("Not enough daily returns.")

        ########################################################
        # EQUITY CURVE
        ########################################################

        equity = (1.0 + daily).cumprod()
        cumulative_return = float(equity.iloc[-1] - 1.0)

        ########################################################
        # DRAWDOWN
        ########################################################

        rolling_max = equity.cummax()
        drawdown_series = (equity - rolling_max) / (rolling_max + EPSILON)
        drawdown_series = drawdown_series.fillna(0.0)

        ########################################################
        # METRICS
        ########################################################

        sharpe = self._sharpe_ratio(daily)
        sortino = self._sortino_ratio(daily)
        max_dd = float(drawdown_series.min())
        max_dd_duration = int(self._max_drawdown_duration(equity))
        ann_vol = self._annual_volatility(daily)
        ann_ret = self._annual_return(daily)
        calmar = self._calmar_ratio(ann_ret, max_dd)
        hit = self._hit_rate(daily)
        turnover = self._turnover(portfolio_df)
        beta = self._beta(daily, benchmark_returns)
        info_ratio = self._information_ratio(daily, benchmark_returns)

        rolling_sharpe = self._rolling_sharpe(daily)
        rolling_volatility = self._rolling_volatility(daily)

        return PerformanceReport(
            cumulative_return=cumulative_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            hit_rate=hit,
            annual_volatility=ann_vol,
            annual_return=ann_ret,
            turnover=turnover,
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

    def _sharpe_ratio(self, daily):
        std = daily.std()
        if std <= EPSILON:
            return 0.0
        return float((daily.mean() / std) * np.sqrt(self.TRADING_DAYS))

    def _sortino_ratio(self, daily):
        downside = daily[daily < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = downside.std()
        if downside_std <= EPSILON:
            return 0.0
        return float((daily.mean() / downside_std) * np.sqrt(self.TRADING_DAYS))

    def _annual_volatility(self, daily):
        std = daily.std()
        if std <= EPSILON:
            return 0.0
        return float(std * np.sqrt(self.TRADING_DAYS))

    def _annual_return(self, daily):
        cumulative = (1.0 + daily).prod()
        years = len(daily) / self.TRADING_DAYS
        if years <= 0:
            return 0.0
        try:
            return float(cumulative ** (1.0 / years) - 1.0)
        except Exception:
            return 0.0

    def _max_drawdown_duration(self, equity):
        rolling_max = equity.cummax()
        drawdown = equity < rolling_max
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
        if abs(max_drawdown) <= EPSILON:
            return 0.0
        return float(annual_return / abs(max_drawdown))

    def _hit_rate(self, daily):
        return float((daily > 0).mean())

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
        return float(turnover.mean()) if len(turnover) > 0 else 0.0

    def _beta(self, strategy, benchmark):
        if benchmark is None:
            return None

        aligned_s, aligned_b = strategy.align(benchmark, join="inner")
        if len(aligned_s) < 2:
            return None

        var = np.var(aligned_b)
        if var <= EPSILON:
            return None

        cov = np.cov(aligned_s, aligned_b)[0][1]
        return float(cov / var)

    def _information_ratio(self, strategy, benchmark):
        if benchmark is None:
            return None

        aligned_s, aligned_b = strategy.align(benchmark, join="inner")
        active = aligned_s - aligned_b
        std = active.std()

        if std <= EPSILON:
            return None

        return float((active.mean() / std) * np.sqrt(self.TRADING_DAYS))

    ########################################################
    # ROLLING
    ########################################################

    def _rolling_sharpe(self, daily):
        rolling = daily.rolling(self.ROLLING_WINDOW)
        sharpe = rolling.mean() / rolling.std().replace(0, np.nan)
        sharpe = sharpe * np.sqrt(self.TRADING_DAYS)
        return sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _rolling_volatility(self, daily):
        vol = daily.rolling(self.ROLLING_WINDOW).std()
        vol = vol * np.sqrt(self.TRADING_DAYS)
        return vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)