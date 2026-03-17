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

    MIN_WINDOWS = 4
    MIN_TEST_DAYS = 20
    MIN_TRAIN_ROWS = 1200
    MIN_CROSS_SECTION = 8
    MIN_SCORE_STD = 1e-6
    MIN_TARGET_STD = 1e-6
    MIN_TRADES_PER_WINDOW = 2

    TOP_K = 8
    BOTTOM_K = 8
    TARGET_GROSS_EXPOSURE = 1.0
    MAX_POSITION_WEIGHT = 0.20

    TRANSACTION_COST = 0.0008
    SLIPPAGE = 0.0005
    MAX_SHARPE = 5.0
    MAX_ABS_PERIOD_RETURN = 0.25

    SCORE_WINSOR_Q = 0.01
    EPSILON = 1e-9
    TARGET_CLIP = 5.0

    MAX_CAPITAL = 10_000_000

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

    # ----------------------------------------------------------

    def _validate_dataset(self, df):

        required = {"date", "ticker", "close"}

        missing = required - set(df.columns)

        if missing:

            raise RuntimeError(
                f"Dataset missing columns: {missing}. "
                f"Walk-forward validation requires real market data."
            )

        missing_features = set(MODEL_FEATURES) - set(df.columns)

        if missing_features:

            raise RuntimeError(
                f"Missing model features: {missing_features}"
            )

    # ----------------------------------------------------------

    def _apply_embargo(self, train_df, test_start):

        embargo_cut = (
            pd.Timestamp(test_start)
            - pd.Timedelta(days=self.embargo_days)
        )

        return train_df[train_df["date"] < embargo_cut]

    # ----------------------------------------------------------

    def _build_fold_target(self, df):

        df = df.sort_values(["date", "ticker"]).copy()

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        forward_prices = df.groupby("ticker")["close"].shift(-FORWARD_DAYS)

        df["raw_forward"] = np.log(forward_prices) - np.log(df["close"])

        df = df.dropna(subset=["raw_forward"])

        cs_mean = df.groupby("date")["raw_forward"].transform("mean")
        cs_std = df.groupby("date")["raw_forward"].transform("std")

        df["target"] = (
            (df["raw_forward"] - cs_mean) /
            (cs_std + self.EPSILON)
        )

        df["target"] = np.clip(
            df["target"],
            -self.TARGET_CLIP,
            self.TARGET_CLIP
        )

        df = df[np.isfinite(df["target"])]

        if df["target"].std() < self.MIN_TARGET_STD:
            return pd.DataFrame()

        df.drop(columns=["raw_forward"], inplace=True)

        return df

    # ----------------------------------------------------------

    def _winsorize(self, x):

        if len(x) < 5:
            return x

        try:

            lower = np.quantile(x, self.SCORE_WINSOR_Q)
            upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)

            return np.clip(x, lower, upper)

        except Exception:
            return x

    # ----------------------------------------------------------

    def _softmax(self, x):

        x = x - np.max(x)

        e = np.exp(x)

        return e / (np.sum(e) + self.EPSILON)

    # ----------------------------------------------------------

    def _compute_turnover(self, old_positions, new_positions):

        all_keys = set(old_positions) | set(new_positions)

        turnover = 0.0

        for k in all_keys:

            turnover += abs(
                new_positions.get(k, 0.0)
                - old_positions.get(k, 0.0)
            )

        return turnover

    # ----------------------------------------------------------

    def run(self, df: pd.DataFrame):

        self._validate_dataset(df)

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        df["date"] = pd.to_datetime(df["date"], utc=True)

        unique_dates = df["date"].drop_duplicates().sort_values()

        results = []
        equity_curve = []

        capital = 10_000.0

        start_idx = self.window_size

        while start_idx < len(unique_dates) - FORWARD_DAYS:

            prev_positions = {}

            train_dates = unique_dates.iloc[start_idx - self.window_size:start_idx]
            test_dates = unique_dates.iloc[start_idx:start_idx + self.step_size]

            if len(test_dates) < self.MIN_TEST_DAYS:
                break

            train_df = df[
                (df["date"] >= train_dates.iloc[0])
                & (df["date"] <= train_dates.iloc[-1])
            ].copy()

            train_df = self._apply_embargo(train_df, test_dates.iloc[0])

            test_df = df[
                (df["date"] >= test_dates.iloc[0])
                & (df["date"] <= test_dates.iloc[-1])
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

            # SAFE FEATURE VALIDATION
            X_train = validate_feature_schema(
                train_df.loc[:, MODEL_FEATURES],
                mode="training"
            )

            # FIX: Drop original MODEL_FEATURES before concat to prevent duplicate columns
            non_feature_cols = [c for c in train_df.columns if c not in set(MODEL_FEATURES)]
            train_df = pd.concat(
                [train_df[non_feature_cols].reset_index(drop=True), X_train],
                axis=1
            )

            model = self.model_trainer(train_df)

            window_returns = []
            trade_count = 0
            turnover_sum = 0.0

            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(0, len(test_dates_sorted) - FORWARD_DAYS, FORWARD_DAYS):

                signal_date = test_dates_sorted[i]
                exit_date = test_dates_sorted[i + FORWARD_DAYS]

                signal_slice = test_df[test_df["date"] == signal_date].copy()
                exit_slice = test_df[test_df["date"] == exit_date].copy()

                signal_slice = signal_slice.replace([np.inf, -np.inf], np.nan)
                signal_slice = signal_slice.dropna(subset=MODEL_FEATURES)

                common = set(signal_slice["ticker"]) & set(exit_slice["ticker"])

                if len(common) < self.MIN_CROSS_SECTION:
                    continue

                signal_slice = signal_slice[signal_slice["ticker"].isin(common)]
                exit_slice = exit_slice[exit_slice["ticker"].isin(common)]

                X = validate_feature_schema(
                    signal_slice.loc[:, MODEL_FEATURES],
                    mode="inference"
                )

                scores = model.predict(X)

                if not np.all(np.isfinite(scores)):
                    continue

                if np.std(scores) < self.MIN_SCORE_STD:
                    continue

                scores = self._winsorize(scores)

                std = scores.std()

                if std < self.EPSILON:
                    continue

                scores = (scores - scores.mean()) / (std + self.EPSILON)

                signal_slice["score"] = scores

                ranked = signal_slice.sort_values("score")

                longs = ranked.tail(self.TOP_K)
                shorts = ranked.head(self.BOTTOM_K)

                if len(longs) < self.TOP_K or len(shorts) < self.BOTTOM_K:
                    continue

                long_w = self._softmax(longs["score"].values)
                short_w = self._softmax(np.abs(shorts["score"].values))

                positions = {}

                for t, w in zip(longs["ticker"], long_w):
                    positions[t] = min(float(w), self.MAX_POSITION_WEIGHT)

                for t, w in zip(shorts["ticker"], short_w):
                    positions[t] = -min(float(w), self.MAX_POSITION_WEIGHT)

                turnover = self._compute_turnover(prev_positions, positions)

                turnover_sum += turnover

                merged = pd.merge(
                    signal_slice[["ticker", "close"]],
                    exit_slice[["ticker", "close"]],
                    on="ticker",
                    suffixes=("_entry", "_exit")
                )

                merged["ret"] = (
                    np.log(merged["close_exit"] * (1 - self.SLIPPAGE))
                    - np.log(merged["close_entry"] * (1 + self.SLIPPAGE))
                )

                period_ret = 0.0

                for row in merged.itertuples():

                    if row.ticker in positions:

                        weight = positions[row.ticker]

                        cost = abs(weight) * self.TRANSACTION_COST

                        period_ret += weight * row.ret - cost

                period_ret = float(
                    np.clip(
                        period_ret,
                        -self.MAX_ABS_PERIOD_RETURN,
                        self.MAX_ABS_PERIOD_RETURN
                    )
                )

                if not np.isfinite(capital):
                    capital = 1.0

                capital *= np.exp(period_ret)

                capital = max(capital, 1.0)

                if capital > self.MAX_CAPITAL:
                    capital = self.MAX_CAPITAL

                prev_positions = positions

                window_returns.append(period_ret)

                equity_curve.append(capital)

                trade_count += 1

            if len(window_returns) < self.MIN_TRADES_PER_WINDOW:

                start_idx += self.step_size
                continue

            window_returns = np.array(window_returns)

            vol = np.std(window_returns)

            sharpe = 0.0 if vol < self.EPSILON else (
                np.mean(window_returns) / vol
            ) * np.sqrt(252 / FORWARD_DAYS)

            sharpe = float(
                np.clip(
                    sharpe,
                    -self.MAX_SHARPE,
                    self.MAX_SHARPE
                )
            )

            results.append({
                "strategy_return": float(window_returns.sum()),
                "sharpe_ratio": sharpe,
                "turnover": turnover_sum / max(trade_count, 1),
                "trade_count": trade_count,
                "win_rate": float(np.mean(window_returns > 0))
            })

            start_idx += self.step_size

        return self.aggregate_results(results, equity_curve)

    # ----------------------------------------------------------

    def aggregate_results(self, results, equity_curve):

        if not results or not equity_curve:

            raise RuntimeError(
                "Walk-forward produced no valid windows."
            )

        df = pd.DataFrame(results)

        curve = np.array(equity_curve)

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
            "avg_turnover": float(df["turnover"].mean()),
            "avg_win_rate": float(df["win_rate"].mean()),
            "avg_trades_per_window": float(df["trade_count"].mean()),
            "num_windows": int(len(df))
        }