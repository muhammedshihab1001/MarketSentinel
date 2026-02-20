import numpy as np
import pandas as pd
import logging

from core.schema.feature_schema import MODEL_FEATURES, CORE_FEATURES, DTYPE
from training.backtesting.regime import MarketRegimeDetector

logger = logging.getLogger("marketsentinel.walkforward")


class WalkForwardValidator:

    MIN_WINDOWS = 6
    MIN_TEST_DAYS = 20

    TOP_K = 2
    BOTTOM_K = 2

    ########################################################

    def __init__(
        self,
        model_trainer,
        window_size=252,
        step_size=63,
        embargo_days=14
    ):
        self.model_trainer = model_trainer
        self.window_size = window_size
        self.step_size = step_size
        self.embargo_days = embargo_days
        self.regime_detector = MarketRegimeDetector()

    ########################################################

    def _apply_embargo(self, train_df, test_start):
        embargo_cut = pd.Timestamp(test_start) - pd.Timedelta(days=self.embargo_days)
        return train_df[train_df["date"] < embargo_cut]

    ########################################################

    def run(self, df: pd.DataFrame):

        logger.info("Walk-forward validation started.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        df = self.regime_detector.detect(df)

        unique_dates = pd.to_datetime(
            df["date"].drop_duplicates()
        ).sort_values()

        results = []
        equity_curve = []

        capital = 10_000.0
        start_idx = self.window_size
        window_id = 1

        while start_idx < len(unique_dates) - 1:

            logger.info("Running WF window #%s", window_id)

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

            model = self.model_trainer(train_df)

            window_returns = []

            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(len(test_dates_sorted) - 1):

                signal_date = test_dates_sorted[i]
                next_date = test_dates_sorted[i + 1]

                signal_slice = test_df[test_df["date"] == signal_date]
                next_slice = test_df[test_df["date"] == next_date]

                if signal_slice.empty or next_slice.empty:
                    continue

                X = signal_slice.loc[:, MODEL_FEATURES].astype(DTYPE)
                probs = model.predict_proba(X)[:, 1]

                signal_slice = signal_slice.copy()
                signal_slice["score"] = probs

                # Rank cross-section
                ranked = signal_slice.sort_values("score")

                longs = ranked.tail(self.TOP_K)
                shorts = ranked.head(self.BOTTOM_K)

                positions = {}

                weight_long = 1.0 / self.TOP_K
                weight_short = -1.0 / self.BOTTOM_K

                for row in longs.itertuples():
                    positions[row.ticker] = weight_long

                for row in shorts.itertuples():
                    positions[row.ticker] = weight_short

                # Compute next-day returns
                merged = pd.merge(
                    signal_slice[["ticker", "close"]],
                    next_slice[["ticker", "close"]],
                    on="ticker",
                    suffixes=("_today", "_next")
                )

                merged["ret"] = (
                    np.log(merged["close_next"]) -
                    np.log(merged["close_today"])
                )

                daily_ret = 0.0

                for row in merged.itertuples():
                    if row.ticker in positions:
                        daily_ret += positions[row.ticker] * row.ret

                window_returns.append(daily_ret)

            if not window_returns:
                start_idx += self.step_size
                window_id += 1
                continue

            window_returns = np.array(window_returns)

            # Compound capital
            for r in window_returns:
                capital *= (1 + r)
                equity_curve.append(capital)

            results.append({
                "strategy_return": float(window_returns.sum()),
                "sharpe_ratio": float(
                    np.mean(window_returns) /
                    (np.std(window_returns) + 1e-9) *
                    np.sqrt(252)
                )
            })

            start_idx += self.step_size
            window_id += 1

        if len(results) < self.MIN_WINDOWS:
            raise RuntimeError("Insufficient WF windows.")

        logger.info("Walk-forward completed successfully.")

        return self.aggregate_results(results, equity_curve)

    ########################################################

    def aggregate_results(self, results, equity_curve):

        df = pd.DataFrame(results)
        curve = np.array(equity_curve, dtype=float)

        peak = np.maximum.accumulate(curve)
        drawdowns = (curve - peak) / peak

        gains = df[df["strategy_return"] > 0]["strategy_return"].sum()
        losses = abs(df[df["strategy_return"] < 0]["strategy_return"].sum()) or 1e-6

        profit_factor = float(gains / losses)

        return {
            "avg_strategy_return": float(df["strategy_return"].mean()),
            "avg_sharpe": float(df["sharpe_ratio"].mean()),
            "profit_factor": profit_factor,
            "max_drawdown": float(drawdowns.min()),
            "return_volatility": float(df["strategy_return"].std()),
            "final_equity": float(curve[-1]) if len(curve) else 0.0,
            "equity_curve": curve.tolist(),
            "num_windows": int(len(df))
        }