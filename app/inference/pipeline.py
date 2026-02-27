import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Dict, Tuple

from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.features.feature_engineering import FeatureEngineer
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
    DTYPE,
    LONG_PERCENTILE,
    SHORT_PERCENTILE,
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
                logger.info("Initializing shared ModelLoader (pipeline)")
                _SHARED_MODEL_LOADER = ModelLoader()
                _ = _SHARED_MODEL_LOADER.xgb
    return _SHARED_MODEL_LOADER


# =========================================================
# CIRCUIT BREAKER
# =========================================================

class CircuitBreaker:
    def __init__(self, threshold=3, cooldown=120):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure = None
        self._lock = threading.Lock()

    def allow(self):
        with self._lock:
            if self.failures < self.threshold:
                return True
            if self.last_failure and (time.time() - self.last_failure) > self.cooldown:
                logger.warning("Circuit breaker reset.")
                self.failures = 0
                return True
            logger.critical("Inference blocked by circuit breaker.")
            return False

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure = time.time()

    def record_success(self):
        with self._lock:
            self.failures = 0


# =========================================================
# INFERENCE PIPELINE
# =========================================================

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    TOP_K = 3
    BOTTOM_K = 3
    TOP_SELECTION = 5   # <-- NEW: Final 5 best picks

    MIN_SCORE_STD = 1e-6
    WEIGHT_TOLERANCE = 1e-6

    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))
    CROSS_SECTIONAL_WINDOW_DAYS = int(os.getenv("CS_WINDOW_DAYS", "30"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "200"))
    SNAPSHOT_CACHE_TTL = int(os.getenv("SNAPSHOT_CACHE_TTL", "15"))

    def __init__(self):
        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.models = get_shared_model_loader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.breaker = CircuitBreaker()
        self.agent = SignalAgent()

        self._snapshot_cache: Dict[Tuple[str, ...], Tuple[Dict, float]] = {}
        self._snapshot_lock = threading.Lock()

        self._validate_models_loaded()

    # =========================================================

    def _validate_models_loaded(self):
        if self.models.schema_signature != get_schema_signature():
            raise RuntimeError(
                "Schema signature mismatch between training and inference."
            )
        logger.info(
            "Model + schema verified | version=%s",
            self.models.xgb_version
        )

    # =========================================================

    def run_snapshot(self, tickers: List[str]):

        expanded_tickers = sorted(set(MarketUniverse.get_universe()))

        if len(expanded_tickers) > self.MAX_BATCH_SIZE:
            raise RuntimeError("Batch size exceeds MAX_BATCH_SIZE.")

        if not self.breaker.allow():
            raise RuntimeError("Inference blocked by circuit breaker.")

        INFERENCE_IN_PROGRESS.inc()
        start_time = time.time()

        try:

            df = self._build_cross_sectional_frame(expanded_tickers)
            latest_df = self._select_latest_snapshot(df)

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference"
            ).astype(DTYPE)

            # ---------------- Model ----------------
            scores = self.models.xgb.predict_proba(feature_df)[:, 1]

            if np.std(scores) < self.MIN_SCORE_STD:
                logger.warning("Score collapse — flattening.")
                scores = np.full(len(scores), 0.5)

            latest_df = latest_df.copy()
            latest_df["score"] = scores

            # Cross-sectional ranking
            latest_df["rank_pct"] = latest_df["score"].rank(
                method="first",
                pct=True
            )

            latest_df["signal"] = latest_df["rank_pct"].apply(
                lambda x:
                "LONG" if x >= LONG_PERCENTILE
                else "SHORT" if x <= SHORT_PERCENTILE
                else "NEUTRAL"
            )

            weights = self._construct_portfolio(latest_df)

            prob_stats = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }

            snapshot_rows = []

            for _, row in latest_df.iterrows():
                agent_output = self.agent.analyze(
                    row=row.to_dict(),
                    probability_stats=prob_stats
                )

                snapshot_rows.append({
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "score": float(row["score"]),
                    "rank_pct": float(row["rank_pct"]),
                    "signal": row["signal"],
                    "weight": float(weights.get(row["ticker"], 0.0)),
                    "agent": agent_output
                })

            # 🚀 SELECT TOP 5 ALPHA PICKS
            alpha_sorted = sorted(
                snapshot_rows,
                key=lambda x: x["agent"]["strength_score"],
                reverse=True
            )

            top_5 = alpha_sorted[:self.TOP_SELECTION]

            result = {
                "snapshot_date": str(latest_df["date"].iloc[0]),
                "universe_size": int(len(latest_df)),
                "probability_stats": prob_stats,
                "top_5": top_5,
                "signals": snapshot_rows
            }

            MODEL_INFERENCE_COUNT.labels(
                model="xgboost"
            ).inc()

            MODEL_INFERENCE_LATENCY.labels(
                model="xgboost"
            ).observe(time.time() - start_time)

            self.breaker.record_success()
            return result

        except Exception:
            PIPELINE_FAILURES.labels(stage="snapshot").inc()
            self.breaker.record_failure()
            logger.exception("Snapshot inference failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()

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
        start_date = end_date - pd.Timedelta(
            days=self.INFERENCE_LOOKBACK_DAYS
        )

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
        cutoff = latest_date - pd.Timedelta(
            days=self.CROSS_SECTIONAL_WINDOW_DAYS
        )

        df = df[df["date"] >= cutoff].copy()

        df = FeatureEngineer.add_cross_sectional_features(df)
        df = FeatureEngineer.finalize(df)

        return df

    # =========================================================

    def _construct_portfolio(self, latest_df):

        latest_df = latest_df.sort_values(
            ["score", "ticker"]
        )

        n = len(latest_df)
        k_long = min(self.TOP_K, n // 2)
        k_short = min(self.BOTTOM_K, n // 2)

        longs = latest_df.tail(k_long)
        shorts = latest_df.head(k_short)

        long_vol = longs["volatility"].replace(0, 1e-6)
        short_vol = shorts["volatility"].replace(0, 1e-6)

        long_weights = (1.0 / long_vol)
        short_weights = (1.0 / short_vol)

        long_weights /= long_weights.sum()
        short_weights /= short_weights.sum()

        long_weights *= self.TARGET_GROSS_EXPOSURE / 2
        short_weights *= self.TARGET_GROSS_EXPOSURE / 2

        weights = {}

        for t, w in zip(longs["ticker"], long_weights):
            weights[t] = float(w)

        for t, w in zip(shorts["ticker"], short_weights):
            weights[t] = -float(w)

        gross = sum(abs(v) for v in weights.values())

        if abs(gross - self.TARGET_GROSS_EXPOSURE) > self.WEIGHT_TOLERANCE:
            raise RuntimeError(f"Gross exposure mismatch: {gross}")

        return weights