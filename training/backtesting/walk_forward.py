import numpy as np
import pandas as pd
import logging

from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
)
from training.backtesting.regime import MarketRegimeDetector

logger = logging.getLogger("marketsentinel.walkforward")

FORWARD_DAYS = 5


class WalkForwardValidator:

    MIN_WINDOWS = 6
    MIN_TEST_DAYS = 20
    MIN_TRAIN_ROWS = 1500
    MIN_CROSS_SECTION = 10
    MIN_SCORE_STD = 1e-6
    MIN_TARGET_STD = 1e-6

    TOP_K = 10
    BOTTOM_K = 10
    TARGET_GROSS_EXPOSURE = 1.0
    MAX_POSITION_WEIGHT = 0.15

    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.0005
    MAX_SHARPE = 5.0

    SCORE_WINSOR_Q = 0.02
    BASE_DISPERSION = 0.03
    MIN_ACTIVE_POSITIONS = 4

    EPSILON = 1e-9

    def __init__(
        self,
        model_trainer,
        window_size=252,
        step_size=63,
        embargo_days=FORWARD_DAYS,
        debug=False
    ):
        self.model_trainer = model_trainer
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.embargo_days = int(embargo_days)
        self.debug = debug
        self.regime_detector = MarketRegimeDetector()

    # ==========================================================
    # EMBARGO
    # ==========================================================

    def _apply_embargo(self, train_df, test_start):
        embargo_cut = pd.Timestamp(test_start) - pd.Timedelta(days=self.embargo_days)
        return train_df[train_df["date"] < embargo_cut]

    # ==========================================================
    # TARGET BUILD
    # ==========================================================

    def _build_fold_target(self, df):

        df = df.sort_values(["date", "ticker"]).copy()

        df["raw_forward"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
        )

        df = df.dropna(subset=["raw_forward"])

        cs_mean = df.groupby("date")["raw_forward"].transform("mean")
        cs_std = df.groupby("date")["raw_forward"].transform("std").replace(0, np.nan)

        df["target"] = (df["raw_forward"] - cs_mean) / cs_std
        df = df[np.isfinite(df["target"])]

        if df["target"].std() < self.MIN_TARGET_STD:
            return pd.DataFrame()

        df.drop(columns=["raw_forward"], inplace=True)

        return df

    # ==========================================================
    # SCORE STABILIZATION
    # ==========================================================

    def _winsorize(self, x):
        lower = np.quantile(x, self.SCORE_WINSOR_Q)
        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)
        return np.clip(x, lower, upper)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + self.EPSILON)

    # ==========================================================
    # TURNOVER
    # ==========================================================

    def _compute_turnover(self, old_positions, new_positions):
        all_keys = set(old_positions.keys()) | set(new_positions.keys())
        turnover = 0.0
        for k in all_keys:
            turnover += abs(new_positions.get(k, 0.0) - old_positions.get(k, 0.0))
        return turnover

    # ==========================================================
    # MAIN WALK-FORWARD
    # ==========================================================

    def run(self, df: pd.DataFrame):

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        unique_dates = df["date"].drop_duplicates().sort_values()

        results = []
        equity_curve = []
        capital = 10_000.0
        prev_positions = {}

        start_idx = self.window_size

        while start_idx < len(unique_dates) - FORWARD_DAYS:

            train_dates = unique_dates.iloc[start_idx - self.window_size:start_idx]
            test_dates = unique_dates.iloc[start_idx:start_idx + self.step_size]

            if len(test_dates) < self.MIN_TEST_DAYS:
                break

            train_df = df[
                (df["date"] >= train_dates.iloc[0]) &
                (df["date"] <= train_dates.iloc[-1])
            ].copy()

            train_df = self._apply_embargo(train_df, test_dates.iloc[0])

            test_df = df[
                (df["date"] >= test_dates.iloc[0]) &
                (df["date"] <= test_dates.iloc[-1])
            ].copy()

            if len(train_df) < self.MIN_TRAIN_ROWS:
                start_idx += self.step_size
                continue

            train_df = self.regime_detector.detect(train_df)
            test_df = self.regime_detector.detect(test_df)

            train_df = self._build_fold_target(train_df)

            if train_df.empty:
                start_idx += self.step_size
                continue

            model = self.model_trainer(train_df)

            window_returns = []
            trade_count = 0
            turnover_sum = 0.0

            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(0, len(test_dates_sorted) - FORWARD_DAYS, FORWARD_DAYS):

                signal_date = test_dates_sorted[i]
                exit_date = test_dates_sorted[i + FORWARD_DAYS]

                if exit_date <= signal_date:
                    continue

                signal_slice = test_df[test_df["date"] == signal_date]
                exit_slice = test_df[test_df["date"] == exit_date]

                common = set(signal_slice["ticker"]) & set(exit_slice["ticker"])

                if len(common) < self.MIN_CROSS_SECTION:
                    continue

                signal_slice = signal_slice[signal_slice["ticker"].isin(common)].copy()
                exit_slice = exit_slice[exit_slice["ticker"].isin(common)].copy()

                X = validate_feature_schema(
                    signal_slice.loc[:, MODEL_FEATURES],
                    mode="inference"
                )

                scores = model.predict(X)

                if not np.all(np.isfinite(scores)):
                    continue

                score_std = np.std(scores)

                if score_std < self.MIN_SCORE_STD:
                    continue

                if score_std < self.BASE_DISPERSION:
                    continue

                scores = self._winsorize(scores)
                scores = (scores - scores.mean()) / (scores.std() + self.EPSILON)

                signal_slice["score"] = scores
                ranked = signal_slice.sort_values("score")

                longs = ranked.tail(self.TOP_K)
                shorts = ranked.head(self.BOTTOM_K)

                if len(longs) < self.MIN_ACTIVE_POSITIONS:
                    continue

                long_alpha = self._softmax(longs["score"].values)
                short_alpha = self._softmax(np.abs(shorts["score"].values))

                long_w = long_alpha / longs["volatility"].clip(lower=0.01).values
                short_w = short_alpha / shorts["volatility"].clip(lower=0.01).values

                long_w /= long_w.sum()
                short_w /= short_w.sum()

                long_w *= self.TARGET_GROSS_EXPOSURE / 2
                short_w *= self.TARGET_GROSS_EXPOSURE / 2

                positions = {}

                for t, w in zip(longs["ticker"], long_w):
                    positions[t] = min(float(w), self.MAX_POSITION_WEIGHT)

                for t, w in zip(shorts["ticker"], short_w):
                    positions[t] = -min(float(w), self.MAX_POSITION_WEIGHT)

                gross = sum(abs(v) for v in positions.values())

                if abs(gross - self.TARGET_GROSS_EXPOSURE) > 0.05:
                    continue

                turnover = self._compute_turnover(prev_positions, positions)
                turnover_sum += turnover

                merged = pd.merge(
                    signal_slice[["ticker", "close"]],
                    exit_slice[["ticker", "close"]],
                    on="ticker",
                    suffixes=("_entry", "_exit")
                )

                merged["ret"] = (
                    np.log(merged["close_exit"] * (1 - self.SLIPPAGE)) -
                    np.log(merged["close_entry"] * (1 + self.SLIPPAGE))
                )

                merged["ret"] = merged["ret"].clip(-0.5, 0.5)

                period_ret = 0.0

                for row in merged.itertuples():
                    if row.ticker in positions:
                        weight = positions[row.ticker]
                        cost = abs(weight) * self.TRANSACTION_COST
                        period_ret += weight * row.ret - cost

                prev_positions = positions
                window_returns.append(period_ret)
                trade_count += 1

            if not window_returns:
                start_idx += self.step_size
                continue

            window_returns = np.array(window_returns)

            for r in window_returns:
                capital *= np.exp(r)
                capital = max(capital, 1.0)
                equity_curve.append(capital)

            vol = np.std(window_returns)
            sharpe = 0.0 if vol < self.EPSILON else (
                np.mean(window_returns) / vol
            ) * np.sqrt(252 / FORWARD_DAYS)

            sharpe = float(np.clip(sharpe, -self.MAX_SHARPE, self.MAX_SHARPE))

            results.append({
                "strategy_return": float(window_returns.sum()),
                "sharpe_ratio": sharpe,
                "turnover": turnover_sum / max(trade_count, 1),
                "trade_count": trade_count,
                "win_rate": float(np.mean(window_returns > 0))
            })

            start_idx += self.step_size

        return self.aggregate_results(results, equity_curve)

    # ==========================================================

    def aggregate_results(self, results, equity_curve):

        if not results or not equity_curve:
            raise RuntimeError("Walk-forward produced no valid windows.")

        df = pd.DataFrame(results)
        curve = np.array(equity_curve)

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(df[df["strategy_return"] < 0]["strategy_return"].sum()) or 1e-6

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "profit_factor": float(gains / losses),
            "max_drawdown": float(drawdowns.min()),
            "return_volatility": float(df["strategy_return"].std()),
            "final_equity": float(curve[-1]),
            "avg_turnover": float(df["turnover"].mean()),
            "avg_win_rate": float(df["win_rate"].mean()),
            "avg_trades_per_window": float(df["trade_count"].mean()),
            "num_windows": int(len(df))
        }