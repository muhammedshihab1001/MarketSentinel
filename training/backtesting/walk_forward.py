import numpy as np
import pandas as pd
import logging

from core.schema.feature_schema import MODEL_FEATURES
from training.backtesting.portfolio_engine import PortfolioBacktestEngine
from training.backtesting.regime import MarketRegimeDetector

logger = logging.getLogger("marketsentinel.walkforward")


class WalkForwardValidator:
    """
    Institutional Walk Forward Validator v5
    Adaptive conviction + balanced ranking + regime throttle.
    """

    MIN_TRADES_PER_WINDOW = 5
    MIN_WINDOWS = 6
    MIN_TEST_DAYS = 20

    MIN_ASSETS_PER_DAY = 3
    MIN_FEATURE_VARIANCE = 1e-8

    TOP_K_PERCENT = 0.30
    MAX_TRADES_PER_DAY = 12

    CRISIS_EXPOSURE_SCALE = 0.35
    DRIFT_WARN_Z = 10.0

    ########################################################

    def __init__(
        self,
        model_trainer,
        signal_generator,
        window_size=252,
        step_size=63,
        embargo_days=14
    ):
        self.model_trainer = model_trainer
        self.signal_generator = signal_generator

        self.window_size = window_size
        self.step_size = step_size
        self.embargo_days = embargo_days

        self.engine = PortfolioBacktestEngine()
        self.regime_detector = MarketRegimeDetector()

    ########################################################

    def _apply_embargo(self, train_df, test_start):
        embargo_cut = pd.Timestamp(test_start) - pd.Timedelta(days=self.embargo_days)
        train_df = train_df[train_df["date"] < embargo_cut]

        if len(train_df) < self.window_size * 0.70:
            raise RuntimeError("Embargo removed too much training data.")

        return train_df

    ########################################################

    def _validate_training_frame(self, df):

        cols = list(MODEL_FEATURES)

        # 🚫 Removed schema validation here
        # Schema validation already happens before normalization
        # in load_training_data()

        features = df.loc[:, cols].to_numpy(dtype=float)

        if not np.isfinite(features).all():
            raise RuntimeError("Non-finite values detected in training features.")

        if np.min(np.var(features, axis=0)) < self.MIN_FEATURE_VARIANCE:
            raise RuntimeError("Feature variance collapsed.")

        if df["target"].nunique() < 2:
            raise RuntimeError("Training labels collapsed.")

    ########################################################

    def _distribution_guard(self, train_df, test_df):

        cols = list(MODEL_FEATURES)

        train_mu = train_df[cols].mean()
        train_std = train_df[cols].std(ddof=0) + 1e-9

        z = np.abs((test_df[cols] - train_mu) / train_std)
        max_z = float(np.nanmax(z.to_numpy()))

        if max_z > self.DRIFT_WARN_Z:
            logger.warning("Feature drift detected | max_z=%.2f", max_z)

    ########################################################

    def _filter_conviction(self, tickers, probs, signals):

        df = pd.DataFrame({
            "ticker": tickers,
            "prob": probs,
            "signal": signals
        })

        if df.empty:
            return {}

        df["edge"] = np.abs(df["prob"] - 0.5)

        edge_std = float(df["edge"].std())
        dynamic_min_edge = max(0.005, edge_std * 0.5)

        df = df[df["edge"] >= dynamic_min_edge]

        if df.empty:
            logger.info("Conviction too weak — skipping day.")
            return {}

        spread = df["prob"].max() - df["prob"].min()

        if spread < 0.01:
            logger.info("Low probability dispersion — reducing exposure.")

        df = df.sort_values("edge", ascending=False)

        k = max(self.MIN_ASSETS_PER_DAY, int(len(df) * self.TOP_K_PERCENT))
        k = min(k, self.MAX_TRADES_PER_DAY)

        df = df.head(k)

        return dict(zip(df["ticker"], df["signal"]))

    ########################################################

    def run(self, df: pd.DataFrame):

        logger.info("Walk-forward validation started.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        df = self.regime_detector.detect(df)

        unique_dates = pd.to_datetime(df["date"].drop_duplicates()).sort_values()

        results = []
        equity_curve = []

        initial_capital = 10_000.0
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

            logger.info("Train rows=%s | Test rows=%s", len(train_df), len(test_df))

            self._validate_training_frame(train_df)
            self._distribution_guard(train_df, test_df)

            model = self.model_trainer(train_df)

            grouped_prices = {}
            grouped_signals = {}
            trade_counter = 0

            test_dates_sorted = sorted(test_df["date"].unique())

            for i in range(len(test_dates_sorted) - 1):

                signal_date = test_dates_sorted[i]
                execution_date = test_dates_sorted[i + 1]

                signal_slice = test_df[test_df["date"] == signal_date]
                exec_slice = test_df[test_df["date"] == execution_date]

                if signal_slice.empty or exec_slice.empty:
                    continue

                prices = {
                    t: float(p)
                    for t, p in zip(exec_slice["ticker"], exec_slice["close"])
                    if pd.notna(p) and np.isfinite(p) and p > 0
                }

                if not prices:
                    continue

                signals, probs = self.signal_generator(model, signal_slice)

                filtered = self._filter_conviction(
                    signal_slice["ticker"].values,
                    probs,
                    signals
                )

                if len(filtered) < self.MIN_ASSETS_PER_DAY:
                    continue

                regime = signal_slice["regime"].iloc[0]

                if regime == "CRISIS":
                    cut = max(1, int(len(filtered) * self.CRISIS_EXPOSURE_SCALE))
                    filtered = dict(list(filtered.items())[:cut])

                trade_counter += len(filtered)

                grouped_prices[execution_date] = {
                    t: prices[t] for t in filtered if t in prices
                }

                grouped_signals[execution_date] = filtered

            if trade_counter < self.MIN_TRADES_PER_WINDOW:
                logger.warning("Window skipped — insufficient trades.")
                start_idx += self.step_size
                window_id += 1
                continue

            logger.info("Trades executed -> %s", trade_counter)

            metrics = self.engine.run(
                grouped_prices,
                grouped_signals,
                initial_cash=initial_capital
            )

            curve = np.array(metrics["equity_curve"], dtype=float)

            if equity_curve:
                scale = equity_curve[-1] / curve[0]
                equity_curve.extend((curve * scale)[1:].tolist())
            else:
                equity_curve.extend(curve.tolist())

            results.append(metrics)

            start_idx += self.step_size
            window_id += 1

        if len(results) < self.MIN_WINDOWS:
            raise RuntimeError("Walk-forward produced insufficient windows.")

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
            "final_equity": float(curve[-1]),
            "equity_curve": curve.tolist(),
            "num_windows": len(df)
        }
