import numpy as np
import pandas as pd
import logging

from core.schema.feature_schema import MODEL_FEATURES, DTYPE
from training.backtesting.regime import MarketRegimeDetector

logger = logging.getLogger("marketsentinel.walkforward")

FORWARD_DAYS = 5


class WalkForwardValidator:

    MIN_WINDOWS = 6
    MIN_TEST_DAYS = 20

    TOP_K = 3
    BOTTOM_K = 3
    TARGET_GROSS_EXPOSURE = 1.0

    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.0005
    MAX_SHARPE = 5.0

    def __init__(
        self,
        model_trainer,
        window_size=252,
        step_size=63,
        embargo_days=FORWARD_DAYS
    ):
        self.model_trainer = model_trainer
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.embargo_days = int(embargo_days)
        self.regime_detector = MarketRegimeDetector()

    ########################################################

    def _apply_embargo(self, train_df, test_start):
        embargo_cut = pd.Timestamp(test_start) - pd.Timedelta(days=self.embargo_days)
        return train_df[train_df["date"] < embargo_cut]

    ########################################################
    # Fold-Local Target
    ########################################################

    def _build_fold_target(self, df):

        if "close" not in df.columns:
            if "target" not in df.columns:
                raise RuntimeError("Dataset missing both close and target.")
            return df

        df = df.sort_values(["ticker", "date"]).copy()

        df["forward_log_return"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
        )

        df = df.dropna(subset=["forward_log_return"])

        df["alpha_rank_pct"] = (
            df.groupby("date")["forward_log_return"]
            .rank(pct=True)
        )

        df["target"] = np.nan
        df.loc[df["alpha_rank_pct"] >= 0.7, "target"] = 1
        df.loc[df["alpha_rank_pct"] <= 0.3, "target"] = 0

        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)

        return df

    ########################################################

    def run(self, df: pd.DataFrame):

        if df.empty:
            raise RuntimeError("WalkForward received empty dataset.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        unique_dates = df["date"].drop_duplicates().sort_values()

        if len(unique_dates) <= self.window_size:
            raise RuntimeError("Insufficient history for walk-forward.")

        results = []
        equity_curve = []
        capital = 10_000.0

        start_idx = self.window_size
        window_id = 1

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

            if train_df.empty or test_df.empty:
                start_idx += self.step_size
                window_id += 1
                continue

            train_df = self.regime_detector.detect(train_df)
            test_df = self.regime_detector.detect(test_df)

            train_df = self._build_fold_target(train_df)

            if train_df["target"].nunique() < 2:
                start_idx += self.step_size
                window_id += 1
                continue

            model = self.model_trainer(train_df)

            window_returns = []
            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(len(test_dates_sorted) - FORWARD_DAYS):

                signal_date = test_dates_sorted[i]
                exit_date = test_dates_sorted[i + FORWARD_DAYS]

                signal_slice = test_df[test_df["date"] == signal_date]
                exit_slice = test_df[test_df["date"] == exit_date]

                if signal_slice.empty or exit_slice.empty:
                    continue

                X = signal_slice.loc[:, MODEL_FEATURES].astype(DTYPE)

                if X.isnull().any().any():
                    continue

                probs = model.predict_proba(X)[:, 1]

                signal_slice = signal_slice.copy()
                signal_slice["score"] = probs

                # 🔒 deterministic ranking
                ranked = signal_slice.sort_values(["score", "ticker"])

                longs = ranked.tail(self.TOP_K)
                shorts = ranked.head(self.BOTTOM_K)

                positions = {}

                long_vol = longs["volatility"].replace(0, 1e-6)
                short_vol = shorts["volatility"].replace(0, 1e-6)

                long_weights = 1.0 / long_vol
                short_weights = 1.0 / short_vol

                long_weights /= long_weights.sum()
                short_weights /= short_weights.sum()

                long_weights *= self.TARGET_GROSS_EXPOSURE / 2
                short_weights *= self.TARGET_GROSS_EXPOSURE / 2

                for t, w in zip(longs["ticker"], long_weights):
                    positions[t] = float(w)

                for t, w in zip(shorts["ticker"], short_weights):
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
                window_id += 1
                continue

            window_returns = np.array(window_returns, dtype=float)

            for r in window_returns:
                capital *= np.exp(r)
                equity_curve.append(capital)

            vol = np.std(window_returns)
            sharpe = (
                (np.mean(window_returns) / (vol + 1e-9))
                * np.sqrt(252 / FORWARD_DAYS)
            )

            sharpe = float(np.clip(sharpe, -self.MAX_SHARPE, self.MAX_SHARPE))

            results.append({
                "strategy_return": float(window_returns.sum()),
                "sharpe_ratio": sharpe
            })

            start_idx += self.step_size
            window_id += 1

        if len(results) < self.MIN_WINDOWS:
            raise RuntimeError("Insufficient WF windows.")

        return self.aggregate_results(results, equity_curve)

    ########################################################

    def aggregate_results(self, results, equity_curve):

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
            "equity_curve": curve.tolist(),
            "num_windows": int(len(df))
        }