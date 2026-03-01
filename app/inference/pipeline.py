# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v2.3
# Multi-Agent Driven Selection (CV-Ready)
# =========================================================

import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from typing import List

from core.data.market_data_service import MarketDataService
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

EPSILON = 1e-12

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


class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    TARGET_VOL = 0.12
    MIN_UNIVERSE_WIDTH = 12
    MIN_SCORE_STD = 1e-6
    MAX_POSITION_WEIGHT = 0.20

    TOP_K = 10
    BOTTOM_K = 10
    TOP_SELECTION = 5

    SCORE_WINSOR_Q = 0.02
    BASE_LIQUIDITY = 1e6

    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))
    CROSS_SECTIONAL_WINDOW_DAYS = int(os.getenv("CS_WINDOW_DAYS", "30"))

    def __init__(self):
        self.market_data = MarketDataService()
        self.models = get_shared_model_loader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.signal_agent = SignalAgent()

        self._validate_models_loaded()

    # ---------------------------------------------------------

    def _validate_models_loaded(self):
        if self.models.schema_signature != get_schema_signature():
            raise RuntimeError("Schema signature mismatch.")
        logger.info("Model verified | version=%s", self.models.xgb_version)

    # ---------------------------------------------------------

    def _winsorize(self, x):
        lower = np.quantile(x, self.SCORE_WINSOR_Q)
        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)
        return np.clip(x, lower, upper)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + EPSILON)

    # ---------------------------------------------------------
    # MAIN SNAPSHOT
    # ---------------------------------------------------------

    def run_snapshot(self, tickers: List[str]):

        universe = sorted(set(tickers or MarketUniverse.get_universe()))
        if not universe:
            raise RuntimeError("Universe empty.")

        INFERENCE_IN_PROGRESS.inc()
        start_time = time.time()

        try:

            df = self._build_cross_sectional_frame(universe)
            latest_df = df[df["date"] == df["date"].max()].copy()

            latest_df = latest_df.replace([np.inf, -np.inf], np.nan)
            latest_df = latest_df.dropna(subset=MODEL_FEATURES)

            if latest_df.empty:
                raise RuntimeError("Latest snapshot invalid.")

            liquidity_threshold = max(
                self.BASE_LIQUIDITY * 0.5,
                latest_df["dollar_volume"].median()
            )

            latest_df = latest_df[
                latest_df["dollar_volume"] > liquidity_threshold
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
                raise RuntimeError("Score dispersion too low.")

            raw_scores = self._winsorize(raw_scores)
            raw_scores = (
                raw_scores - raw_scores.mean()
            ) / (raw_scores.std() + EPSILON)

            latest_df["raw_model_score"] = raw_scores

            # -------------------------------------------------
            # RUN SIGNAL AGENT FOR ALL INSTRUMENTS
            # -------------------------------------------------

            snapshot_rows = []
            score_mean = float(raw_scores.mean())
            score_std = float(raw_scores.std())

            for _, row in latest_df.iterrows():

                direction = "LONG" if row["raw_model_score"] > 0 else "SHORT"

                agent_output = self.signal_agent.analyze(
                    row={**row.to_dict(), "signal": direction},
                    probability_stats={"mean": score_mean, "std": score_std},
                    drift_score=drift_result.get("severity_score", 0)
                )

                snapshot_rows.append({
                    "date": str(row["date"]),
                    "ticker": row["ticker"],
                    "raw_model_score": float(row["raw_model_score"]),
                    "agent_score": float(agent_output["agent_score"]),
                    "weight": 0.0,
                    "agent": agent_output
                })

            # -------------------------------------------------
            # SELECT TOP 5 BY AGENT SCORE (MULTI-AGENT DRIVEN)
            # -------------------------------------------------

            top_5 = sorted(
                snapshot_rows,
                key=lambda x: x["agent_score"],
                reverse=True
            )[:self.TOP_SELECTION]

            # -------------------------------------------------
            # PORTFOLIO STILL BASED ON MODEL SCORE (STABLE)
            # -------------------------------------------------

            ranked = latest_df.sort_values("raw_model_score")
            longs = ranked.tail(self.TOP_K)
            shorts = ranked.head(self.BOTTOM_K)

            weights = self._construct_portfolio(longs, shorts)

            for row in snapshot_rows:
                row["weight"] = float(weights.get(row["ticker"], 0.0))

            result = {
                "snapshot_date": str(latest_df["date"].max()),
                "universe_size": int(len(latest_df)),
                "gross_exposure": float(sum(abs(x["weight"]) for x in snapshot_rows)),
                "net_exposure": float(sum(x["weight"] for x in snapshot_rows)),
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

    # ---------------------------------------------------------

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

    # ---------------------------------------------------------

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

            try:
                df = FeatureEngineer.build_feature_pipeline(
                    price_df=price_df,
                    sentiment_df=None,
                    training=False
                )
                datasets.append(df)
            except Exception:
                logger.warning("Feature build failed for %s", ticker)

        if not datasets:
            raise RuntimeError("All tickers failed feature build.")

        df = pd.concat(datasets, ignore_index=True)
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        df = FeatureEngineer.add_cross_sectional_features(df)
        df = FeatureEngineer.finalize(df)

        return df

    # ---------------------------------------------------------

    def _construct_portfolio(self, longs, shorts):

        long_alpha = self._softmax(longs["raw_model_score"].values)
        short_alpha = self._softmax(np.abs(shorts["raw_model_score"].values))

        long_w = long_alpha / (long_alpha.sum() + EPSILON)
        short_w = short_alpha / (short_alpha.sum() + EPSILON)

        long_w *= self.TARGET_GROSS_EXPOSURE / 2
        short_w *= self.TARGET_GROSS_EXPOSURE / 2

        weights = {}

        for t, w in zip(longs["ticker"], long_w):
            weights[t] = min(float(w), self.MAX_POSITION_WEIGHT)

        for t, w in zip(shorts["ticker"], short_w):
            weights[t] = -min(float(w), self.MAX_POSITION_WEIGHT)

        return weights