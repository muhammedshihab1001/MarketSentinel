import numpy as np
import pandas as pd
import logging

from core.schema.feature_schema import (
    MODEL_FEATURES,
    DTYPE,
    validate_feature_schema,
)
from training.backtesting.regime import MarketRegimeDetector

logger = logging.getLogger("marketsentinel.walkforward")

FORWARD_DAYS = 10


class WalkForwardValidator:

    MIN_WINDOWS = 6
    MIN_TEST_DAYS = 20
    MIN_TRAIN_ROWS = 1500
    MIN_CROSS_SECTION = 10
    MIN_SCORE_STD = 1e-6

    TOP_K = 5
    BOTTOM_K = 5
    TARGET_GROSS_EXPOSURE = 1.0

    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.0005
    MAX_SHARPE = 5.0

    SCORE_WINSOR_Q = 0.02

    # Relaxed dispersion logic (adaptive)
    MIN_DISPERSION = 0.05

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

        self._direction = 1
        self._direction_locked = False

    ########################################################

    def _apply_embargo(self, train_df, test_start):
        embargo_cut = pd.Timestamp(test_start) - pd.Timedelta(days=self.embargo_days)
        return train_df[train_df["date"] < embargo_cut]

    ########################################################
    # TARGET (CROSS-SECTIONAL NEUTRALIZED)
    ########################################################

    def _build_fold_target(self, df):
        df = df.sort_values(["date", "ticker"]).copy()

        df["raw_forward"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
        )

        df = df.dropna(subset=["raw_forward"])

        cs_mean = df.groupby("date")["raw_forward"].transform("mean")
        df["target"] = df["raw_forward"] - cs_mean

        df.drop(columns=["raw_forward"], inplace=True)
        return df

    ########################################################

    def _winsorize(self, x):
        lower = np.quantile(x, self.SCORE_WINSOR_Q)
        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)
        return np.clip(x, lower, upper)

    ########################################################

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + self.EPSILON)

    ########################################################

    def _auto_detect_direction(self, scores, forward_returns):
        if self._direction_locked:
            return

        corr = np.corrcoef(scores, forward_returns)[0, 1]
        if np.isnan(corr):
            return

        self._direction = 1 if corr >= 0 else -1
        self._direction_locked = True

    ########################################################

    def _predict_scores(self, model, X):
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
            except Exception:
                pass
        return model.predict(X)

    ########################################################

    def run(self, df: pd.DataFrame):

        if df.empty:
            raise RuntimeError("WalkForward received empty dataset.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        if df.duplicated(subset=["date", "ticker"]).any():
            raise RuntimeError("Duplicate date/ticker rows detected.")

        unique_dates = df["date"].drop_duplicates().sort_values()

        if len(unique_dates) <= self.window_size:
            raise RuntimeError("Insufficient history for walk-forward.")

        results = []
        equity_curve = []
        capital = 10_000.0

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

            if train_df["target"].std() < self.EPSILON:
                start_idx += self.step_size
                continue

            model = self.model_trainer(train_df)

            window_returns = []
            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(0, len(test_dates_sorted) - FORWARD_DAYS, FORWARD_DAYS):

                signal_date = test_dates_sorted[i]
                exit_date = test_dates_sorted[i + FORWARD_DAYS]

                signal_slice = test_df[test_df["date"] == signal_date]
                exit_slice = test_df[test_df["date"] == exit_date]

                common_tickers = set(signal_slice["ticker"]) & set(exit_slice["ticker"])

                if len(common_tickers) < self.MIN_CROSS_SECTION:
                    continue

                signal_slice = signal_slice[
                    signal_slice["ticker"].isin(common_tickers)
                ].copy()

                exit_slice = exit_slice[
                    exit_slice["ticker"].isin(common_tickers)
                ].copy()

                X = signal_slice.loc[:, MODEL_FEATURES].astype(DTYPE)
                X = validate_feature_schema(X, mode="inference")

                scores = self._predict_scores(model, X)

                if np.std(scores) < self.MIN_SCORE_STD:
                    continue

                scores = self._winsorize(scores)
                scores = (scores - scores.mean()) / (scores.std() + self.EPSILON)

                # Adaptive dispersion gate
                if np.std(scores) < self.MIN_DISPERSION:
                    continue

                signal_slice["score"] = scores

                forward_ret = (
                    np.log(exit_slice["close"].values) -
                    np.log(signal_slice["close"].values)
                )

                self._auto_detect_direction(scores, forward_ret)

                ranked = signal_slice.sort_values("score")

                if self._direction == 1:
                    longs = ranked.tail(self.TOP_K)
                    shorts = ranked.head(self.BOTTOM_K)
                else:
                    longs = ranked.head(self.TOP_K)
                    shorts = ranked.tail(self.BOTTOM_K)

                long_alpha = self._softmax(longs["score"].values)
                short_alpha = self._softmax(np.abs(shorts["score"].values))

                long_vol = longs["volatility"].clip(lower=0.01).values
                short_vol = shorts["volatility"].clip(lower=0.01).values

                long_w = long_alpha / long_vol
                short_w = short_alpha / short_vol

                long_w /= long_w.sum()
                short_w /= short_w.sum()

                long_w *= self.TARGET_GROSS_EXPOSURE / 2
                short_w *= self.TARGET_GROSS_EXPOSURE / 2

                positions = {}

                for t, w in zip(longs["ticker"], long_w):
                    positions[t] = float(w)

                for t, w in zip(shorts["ticker"], short_w):
                    positions[t] = -float(w)

                merged = pd.merge(
                    signal_slice[["ticker", "close"]],
                    exit_slice[["ticker", "close"]],
                    on="ticker",
                    suffixes=("_entry", "_exit")
                )

                if merged.empty:
                    continue

                merged["ret"] = (
                    np.log(merged["close_exit"] * (1 - self.SLIPPAGE)) -
                    np.log(merged["close_entry"] * (1 + self.SLIPPAGE))
                )

                period_ret = 0.0

                for row in merged.itertuples():
                    if row.ticker in positions:
                        weight = positions[row.ticker]
                        gross_cost = abs(weight) * self.TRANSACTION_COST
                        period_ret += weight * row.ret - gross_cost

                window_returns.append(period_ret)

            if not window_returns:
                start_idx += self.step_size
                continue

            window_returns = np.array(window_returns, dtype=np.float64)

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
                "sharpe_ratio": sharpe
            })

            start_idx += self.step_size

        if len(results) < self.MIN_WINDOWS:
            raise RuntimeError("Insufficient WF windows.")

        return self.aggregate_results(results, equity_curve)

    ########################################################

    def aggregate_results(self, results, equity_curve):

        df = pd.DataFrame(results)
        curve = np.array(equity_curve, dtype=np.float64)

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(
            df[df["strategy_return"] < 0]["strategy_return"].sum()
        ) or 1e-6

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "profit_factor": float(gains / losses),
            "max_drawdown": float(drawdowns.min()),
            "return_volatility": float(df["strategy_return"].std()),
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),
            "num_windows": int(len(df))
        }