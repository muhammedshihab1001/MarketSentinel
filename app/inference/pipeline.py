import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from typing import List

from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.features.feature_engineering import FeatureEngineer
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
    DTYPE,
)
from core.market.universe import MarketUniverse

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache
from core.agent.signal_agent import SignalAgent

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    PIPELINE_FAILURES,
    INFERENCE_IN_PROGRESS
)

logger = logging.getLogger("marketsentinel.pipeline")


# =========================================================
# SHARED MODEL LOADER
# =========================================================

_SHARED_MODEL_LOADER = None
_MODEL_LOCK = threading.Lock()


def get_shared_model_loader():
    global _SHARED_MODEL_LOADER
    if _SHARED_MODEL_LOADER is None:
        with _MODEL_LOCK:
            if _SHARED_MODEL_LOADER is None:
                logger.info("Initializing shared ModelLoader")
                _SHARED_MODEL_LOADER = ModelLoader()
                _ = _SHARED_MODEL_LOADER.xgb
    return _SHARED_MODEL_LOADER


# =========================================================
# INFERENCE PIPELINE
# =========================================================

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    MAX_UNIVERSE_WIDTH = 500
    MIN_UNIVERSE_WIDTH = 15

    TOP_K = 10
    BOTTOM_K = 10
    TOP_SELECTION = 5

    MIN_SCORE_STD = 1e-6
    EPSILON = 1e-9
    WEIGHT_TOLERANCE = 1e-6

    SCORE_WINSOR_Q = 0.02
    MIN_LIQUIDITY = 1e6  # liquidity filter

    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))
    CROSS_SECTIONAL_WINDOW_DAYS = int(os.getenv("CS_WINDOW_DAYS", "30"))

    def __init__(self):
        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.models = get_shared_model_loader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.agent = SignalAgent()

        self._validate_models_loaded()

    # =========================================================

    def _validate_models_loaded(self):
        if self.models.schema_signature != get_schema_signature():
            raise RuntimeError("Schema signature mismatch.")
        logger.info("Model verified | version=%s", self.models.xgb_version)

    # =========================================================

    def _winsorize(self, x):
        lower = np.quantile(x, self.SCORE_WINSOR_Q)
        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)
        return np.clip(x, lower, upper)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + self.EPSILON)

    # =========================================================

    def run_snapshot(self, tickers: List[str]):

        production_universe = sorted(set(MarketUniverse.get_universe()))

        if not production_universe:
            raise RuntimeError("Universe empty.")

        INFERENCE_IN_PROGRESS.inc()
        start_time = time.time()

        try:

            df = self._build_cross_sectional_frame(production_universe)

            if len(df["ticker"].unique()) < self.MIN_UNIVERSE_WIDTH:
                raise RuntimeError("Universe too small for institutional scoring.")

            latest_df = self._select_latest_snapshot(df)

            # Liquidity filter
            latest_df = latest_df[
                latest_df["dollar_volume"] > self.MIN_LIQUIDITY
            ].copy()

            if len(latest_df) < self.MIN_UNIVERSE_WIDTH:
                raise RuntimeError("Insufficient liquid instruments.")

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference"
            ).astype(DTYPE)

            drift_result = self._safe_drift(feature_df)

            raw_scores = self.models.xgb.predict(feature_df)

            if not np.all(np.isfinite(raw_scores)):
                raise RuntimeError("Non-finite raw scores.")

            if np.std(raw_scores) < self.MIN_SCORE_STD:
                logger.warning("Score dispersion too low — neutralizing.")
                raw_scores = np.zeros(len(raw_scores))

            scores = self._winsorize(raw_scores)
            scores = (scores - scores.mean()) / (scores.std() + self.EPSILON)

            latest_df = latest_df.copy()
            latest_df["score"] = scores

            ranked = latest_df.sort_values(
                ["score", "ticker"],
                ascending=[True, True]
            )

            top_k = min(self.TOP_K, len(ranked) // 2)
            bottom_k = min(self.BOTTOM_K, len(ranked) // 2)

            longs = ranked.tail(top_k)
            shorts = ranked.head(bottom_k)

            weights = self._construct_portfolio(longs, shorts)

            if drift_result.get("drift_detected"):
                logger.warning("Drift detected — scaling exposure.")
                for k in weights:
                    weights[k] *= 0.5

            snapshot_rows = self._build_snapshot(latest_df, weights)

            top_5 = sorted(
                snapshot_rows,
                key=lambda x: x["score"],
                reverse=True
            )[:self.TOP_SELECTION]

            result = {
                "snapshot_date": str(latest_df["date"].iloc[0]),
                "universe_size": int(len(latest_df)),
                "drift": drift_result,
                "top_5": top_5,
                "signals": snapshot_rows
            }

            MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
            MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(
                time.time() - start_time
            )

            return result

        except Exception:
            PIPELINE_FAILURES.labels(stage="snapshot").inc()
            logger.exception("Snapshot failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()

    # =========================================================

    def _safe_drift(self, feature_df):
        try:
            return self.drift_detector.detect(feature_df)
        except Exception:
            return {
                "drift_detected": False,
                "drift_state": "bypass",
                "severity_score": 0
            }

    # =========================================================

    def _build_snapshot(self, df, weights):

        prob_stats = {
            "mean": float(df["score"].mean()),
            "std": float(df["score"].std()),
            "min": float(df["score"].min()),
            "max": float(df["score"].max())
        }

        snapshot_rows = []

        for _, row in df.iterrows():

            direction = "LONG" if row["score"] > 0 else "SHORT"

            agent_output = self.agent.analyze(
                row={**row.to_dict(), "signal": direction},
                probability_stats=prob_stats
            )

            snapshot_rows.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "score": float(row["score"]),
                "weight": float(weights.get(row["ticker"], 0.0)),
                "agent": agent_output
            })

        return snapshot_rows

    # =========================================================

    def _select_latest_snapshot(self, df: pd.DataFrame):
        latest_date = df["date"].max()
        latest_df = df[df["date"] == latest_date]
        if latest_df.empty:
            raise RuntimeError("No rows for latest snapshot.")
        return latest_df.reset_index(drop=True)

    # =========================================================

    def _build_cross_sectional_frame(self, tickers: List[str]):

        end_date = pd.Timestamp.utcnow()
        start_date = end_date - pd.Timedelta(days=self.INFERENCE_LOOKBACK_DAYS)

        price_map = self.market_data.get_price_data_batch(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            min_history=60
        )

        datasets = []

        for ticker, price_df in price_map.items():

            if price_df is None or price_df.empty:
                continue

            dataset = self.feature_store.get_features(
                price_df=price_df,
                sentiment_df=None,
                ticker=ticker,
                training=False
            )

            if dataset is None or dataset.empty:
                continue

            datasets.append(dataset)

        if not datasets:
            raise RuntimeError("All tickers failed feature build.")

        df = pd.concat(datasets, ignore_index=True)
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        latest_date = df["date"].max()
        cutoff = latest_date - pd.Timedelta(days=self.CROSS_SECTIONAL_WINDOW_DAYS)

        df = df[df["date"] >= cutoff].copy()

        df = FeatureEngineer.add_cross_sectional_features(df)
        df = FeatureEngineer.finalize(df)

        return df

    # =========================================================

    def _construct_portfolio(self, longs, shorts):

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

        weights = {}

        for t, w in zip(longs["ticker"], long_w):
            weights[t] = float(w)

        for t, w in zip(shorts["ticker"], short_w):
            weights[t] = -float(w)

        gross = sum(abs(v) for v in weights.values())

        if abs(gross - self.TARGET_GROSS_EXPOSURE) > self.WEIGHT_TOLERANCE:
            raise RuntimeError(f"Gross exposure mismatch: {gross}")

        return weights