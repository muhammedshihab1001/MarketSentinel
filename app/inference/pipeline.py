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
# INFERENCE PIPELINE (INSTITUTIONAL HYBRID)
# =========================================================

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    MIN_UNIVERSE_WIDTH = 15

    TOP_K = 10
    BOTTOM_K = 10
    TOP_SELECTION = 5

    MIN_SCORE_STD = 1e-6
    EPSILON = 1e-9
    WEIGHT_TOLERANCE = 1e-6

    SCORE_WINSOR_Q = 0.02
    MIN_LIQUIDITY = 1e6

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
    # UTILS
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
    # AGENTS
    # =========================================================

    def _risk_agent(self, drift_result):
        exposure_scale = drift_result.get("exposure_scale", 1.0)
        return float(np.clip(exposure_scale, 0.0, 1.0))

    def _macro_agent(self, df):
        breadth = df["breadth"].iloc[0]
        if breadth > 0.65:
            return 1.1
        if breadth < 0.35:
            return 0.75
        return 1.0

    def _technical_agent(self, row):
        rsi = row.get("rsi", 50)
        momentum = row.get("momentum_composite", 0)

        if rsi > 70:
            return 0.85
        if rsi < 30:
            return 1.1
        if momentum > 0:
            return 1.05
        return 1.0

    # =========================================================

    def run_snapshot(self, tickers: List[str]):

        universe = sorted(set(tickers or MarketUniverse.get_universe()))
        if not universe:
            raise RuntimeError("Universe empty.")

        INFERENCE_IN_PROGRESS.inc()
        start_time = time.time()

        try:

            df = self._build_cross_sectional_frame(universe)

            latest_df = self._select_latest_snapshot(df)
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

            if np.std(raw_scores) < self.MIN_SCORE_STD:
                raw_scores = np.zeros(len(raw_scores))

            alpha_scores = self._winsorize(raw_scores)
            alpha_scores = (
                alpha_scores - alpha_scores.mean()
            ) / (alpha_scores.std() + self.EPSILON)

            latest_df["alpha_score"] = alpha_scores

            # -----------------------------
            # Hybrid Overlay
            # -----------------------------

            risk_mult = self._risk_agent(drift_result)
            macro_mult = self._macro_agent(latest_df)

            final_scores = []

            for _, row in latest_df.iterrows():
                tech_mult = self._technical_agent(row)

                final_score = (
                    row["alpha_score"]
                    * macro_mult
                    * tech_mult
                )

                final_scores.append(final_score)

            latest_df["score"] = np.array(final_scores)

            # Score statistics for API
            score_mean = float(latest_df["score"].mean())
            score_std = float(latest_df["score"].std())

            ranked = latest_df.sort_values("score")

            longs = ranked.tail(self.TOP_K)
            shorts = ranked.head(self.BOTTOM_K)

            weights = self._construct_portfolio(longs, shorts)

            # Apply risk scaling to final weights
            for k in weights:
                weights[k] *= risk_mult

            snapshot_rows = self._build_snapshot(
                latest_df,
                weights,
                macro_mult,
                risk_mult,
                score_mean,
                score_std
            )

            top_5 = sorted(
                snapshot_rows,
                key=lambda x: x["score"],
                reverse=True
            )[:self.TOP_SELECTION]

            result = {
                "snapshot_date": str(latest_df["date"].iloc[0]),
                "universe_size": int(len(latest_df)),
                "drift": drift_result,
                "macro_multiplier": macro_mult,
                "risk_multiplier": risk_mult,
                "score_mean": score_mean,
                "score_std": score_std,
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
                "severity_score": 0,
                "exposure_scale": 1.0
            }

    # =========================================================

    def _build_snapshot(
        self,
        df,
        weights,
        macro_mult,
        risk_mult,
        score_mean,
        score_std
    ):

        snapshot_rows = []

        for _, row in df.iterrows():

            direction = "LONG" if row["score"] > 0 else "SHORT"

            agent_output = self.agent.analyze(
                row={**row.to_dict(), "signal": direction},
                probability_stats={
                    "mean": score_mean,
                    "std": score_std
                }
            )

            snapshot_rows.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "alpha_score": float(row["alpha_score"]),
                "score": float(row["score"]),
                "weight": float(weights.get(row["ticker"], 0.0)),
                "macro_multiplier": float(macro_mult),
                "risk_multiplier": float(risk_mult),
                "agent": agent_output
            })

        return snapshot_rows

    # =========================================================

    def _select_latest_snapshot(self, df):
        latest_date = df["date"].max()
        return df[df["date"] == latest_date].reset_index(drop=True)

    # =========================================================

    def _build_cross_sectional_frame(self, tickers):

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

            df = FeatureEngineer.build_feature_pipeline(
                price_df=price_df,
                sentiment_df=None,
                training=False
            )

            if df is not None and not df.empty:
                datasets.append(df)

        if not datasets:
            raise RuntimeError("All tickers failed feature build.")

        df = pd.concat(datasets, ignore_index=True)
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        latest_date = df["date"].max()
        cutoff = latest_date - pd.Timedelta(days=self.CROSS_SECTIONAL_WINDOW_DAYS)

        return df[df["date"] >= cutoff].copy()

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
            raise RuntimeError("Gross exposure mismatch.")

        return weights