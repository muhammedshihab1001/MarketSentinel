import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Dict

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


# ============================================================
# CIRCUIT BREAKER
# ============================================================

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


# ============================================================
# INFERENCE PIPELINE
# ============================================================

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    TOP_K = 3
    BOTTOM_K = 3

    MIN_PROB_STD = 1e-6
    WEIGHT_TOLERANCE = 1e-6

    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))
    CROSS_SECTIONAL_WINDOW_DAYS = int(os.getenv("CS_WINDOW_DAYS", "30"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "200"))
    SNAPSHOT_CACHE_TTL = int(os.getenv("SNAPSHOT_CACHE_TTL", "15"))

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.models = ModelLoader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.breaker = CircuitBreaker()
        self.agent = SignalAgent()

        self._snapshot_cache = {}
        self._snapshot_lock = threading.Lock()

        _ = self.models.xgb
        self._validate_models_loaded()

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_models_loaded(self):

        container = self.models._xgb_container

        if container is None:
            raise RuntimeError("Model container missing.")

        if container.schema_signature != get_schema_signature():
            raise RuntimeError(
                "Schema signature mismatch between training and inference."
            )

        logger.info("Model + schema signature verified.")

    # ============================================================
    # MISSING METHOD #1 (FIXED)
    # ============================================================

    def _select_latest_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select latest available date per ticker.
        Ensures one row per ticker for inference snapshot.
        """

        if df is None or df.empty:
            raise RuntimeError("Feature dataframe empty before snapshot selection.")

        if "date" not in df.columns:
            raise RuntimeError("Missing 'date' column in feature dataframe.")

        # Get most recent date in dataset
        latest_date = df["date"].max()

        latest_df = df[df["date"] == latest_date].copy()

        if latest_df.empty:
            raise RuntimeError("No rows found for latest snapshot date.")

        return latest_df.reset_index(drop=True)

    # ============================================================
    # PUBLIC SNAPSHOT ENTRYPOINT
    # ============================================================

    def run_snapshot(self, tickers: List[str]):

        cache_key = tuple(sorted(tickers))
        cached = self._snapshot_cache.get(cache_key)
        if cached:
            result, ts = cached
            if time.time() - ts <= self.SNAPSHOT_CACHE_TTL:
                return result

        if len(tickers) > self.MAX_BATCH_SIZE:
            raise RuntimeError("Batch size exceeds MAX_BATCH_SIZE.")

        if not self.breaker.allow():
            raise RuntimeError("Inference blocked by circuit breaker.")

        INFERENCE_IN_PROGRESS.inc()
        start_time = time.time()

        try:

            df = self._build_cross_sectional_frame(tickers)
            latest_df = self._select_latest_snapshot(df)

            universe = set(MarketUniverse.get_universe())
            unknown = set(latest_df["ticker"]) - universe
            if unknown:
                raise RuntimeError(f"Unknown tickers detected: {unknown}")

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference"
            ).astype(DTYPE)

            drift_result = self.drift_detector.detect(feature_df)

            if drift_result.get("drift_state") == "hard":
                raise RuntimeError("Hard drift detected.")

            probs = self.models.xgb.predict_proba(feature_df)[:, 1]
            probs = np.clip(probs, 1e-6, 1 - 1e-6)

            if np.std(probs) < self.MIN_PROB_STD:
                raise RuntimeError("Probability collapse detected.")

            latest_df["score"] = probs
            latest_df["rank_pct"] = latest_df["score"].rank(method="first", pct=True)

            latest_df["signal"] = latest_df["rank_pct"].apply(
                lambda x: "LONG" if x >= LONG_PERCENTILE
                else ("SHORT" if x <= SHORT_PERCENTILE else "NEUTRAL")
            )

            weights = self._construct_portfolio(latest_df)

            snapshot_rows = []

            for _, row in latest_df.iterrows():

                agent_output = self.agent.analyze(
                    row=row.to_dict(),
                    probability_stats={
                        "mean": float(np.mean(probs)),
                        "std": float(np.std(probs)),
                        "min": float(np.min(probs)),
                        "max": float(np.max(probs))
                    }
                )

                snapshot_rows.append({
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "score": float(row["score"]),
                    "rank_pct": float(row["rank_pct"]),
                    "signal": row["signal"],
                    "weight": float(weights.get(row["ticker"], 0.0)),
                    "volatility": float(row.get("volatility", 0.0)),
                    "agent": agent_output
                })

            result = {
                "snapshot_date": str(latest_df["date"].iloc[0]),
                "universe_size": len(latest_df),
                "signals": snapshot_rows,
                "drift": drift_result
            }

            MODEL_INFERENCE_COUNT.inc()
            MODEL_INFERENCE_LATENCY.observe(time.time() - start_time)

            self._snapshot_cache[cache_key] = (result, time.time())
            self.breaker.record_success()

            return result

        except Exception:
            PIPELINE_FAILURES.labels(stage="snapshot").inc()
            self.breaker.record_failure()
            logger.exception("Snapshot inference failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()

    # ============================================================
    # MISSING METHOD #2 (FIXED)
    # ============================================================

    def run_historical_with_features(self, tickers: List[str], date: pd.Timestamp):

        df = self._build_cross_sectional_frame(tickers)

        if df.empty:
            raise RuntimeError("Historical dataframe empty.")

        df = df[df["date"] == date]

        if df.empty:
            raise RuntimeError("No historical rows for date.")

        latest_df = df.copy()

        feature_df = validate_feature_schema(
            latest_df.loc[:, MODEL_FEATURES],
            mode="inference"
        ).astype(DTYPE)

        probs = self.models.xgb.predict_proba(feature_df)[:, 1]
        latest_df["score"] = probs

        latest_df["rank_pct"] = latest_df["score"].rank(method="first", pct=True)
        latest_df["signal"] = latest_df["rank_pct"].apply(
            lambda x: "LONG" if x >= LONG_PERCENTILE
            else ("SHORT" if x <= SHORT_PERCENTILE else "NEUTRAL")
        )

        weights = self._construct_portfolio(latest_df)

        return weights

    # ============================================================
    # FEATURE BUILD
    # ============================================================

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

    # ============================================================
    # PORTFOLIO
    # ============================================================

    def _construct_portfolio(self, latest_df):

        latest_df = latest_df.sort_values(["score", "ticker"])

        k_long = min(self.TOP_K, len(latest_df) // 2)
        k_short = min(self.BOTTOM_K, len(latest_df) // 2)

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