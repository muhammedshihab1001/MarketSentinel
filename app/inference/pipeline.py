import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from datetime import timedelta
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

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    PIPELINE_FAILURES,
    INFERENCE_IN_PROGRESS
)

logger = logging.getLogger("marketsentinel.pipeline")


############################################################
# CIRCUIT BREAKER (UNCHANGED)
############################################################

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


############################################################
# INFERENCE PIPELINE
############################################################

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    TOP_K = 3
    BOTTOM_K = 3

    MIN_PROB_STD = 1e-6
    WEIGHT_TOLERANCE = 1e-6

    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))
    MAX_DATA_STALENESS_DAYS = 5
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "200"))

    def __init__(self):

        self.market_data = MarketDataService()
        self.feature_store = FeatureStore()
        self.models = ModelLoader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()
        self.breaker = CircuitBreaker()

        _ = self.models.xgb
        self._validate_models_loaded()

    ############################################################
    # VALIDATION
    ############################################################

    def _validate_models_loaded(self):

        container = self.models._xgb_container

        if container is None:
            raise RuntimeError("Model container missing.")

        if container.schema_signature != get_schema_signature():
            raise RuntimeError(
                "Schema signature mismatch between training and inference."
            )

        logger.info("Model + schema signature verified.")

    ############################################################
    # PUBLIC ENTRYPOINTS
    ############################################################

    def run_single(self, ticker: str):
        return self.run_batch([ticker])

    def run_batch(self, tickers: List[str]):

        if len(tickers) > self.MAX_BATCH_SIZE:
            raise RuntimeError("Batch size exceeds MAX_BATCH_SIZE.")

        if not self.breaker.allow():
            raise RuntimeError("Inference blocked by circuit breaker.")

        INFERENCE_IN_PROGRESS.inc()

        try:
            df = self._build_cross_sectional_frame(tickers)
            latest_df = self._select_latest_snapshot(df)
            result = self._score_and_construct(latest_df)

            self.breaker.record_success()
            return result

        except Exception:
            PIPELINE_FAILURES.labels(stage="run_batch").inc()
            self.breaker.record_failure()
            logger.exception("Inference pipeline failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()

    ############################################################
    # 🔥 FULLY COMPATIBLE HISTORICAL BACKTEST
    ############################################################

    def run_historical_with_features(self, *args, **kwargs):

        """
        Compatible with:
        - positional df
        - df= keyword
        - evaluation_date
        - future kwargs
        """

        df = None
        evaluation_date = None

        # positional
        if len(args) >= 1:
            df = args[0]

        # keyword override
        if "df" in kwargs:
            df = kwargs["df"]

        if "evaluation_date" in kwargs:
            evaluation_date = kwargs["evaluation_date"]

        if df is None:
            raise RuntimeError("Historical inference requires dataframe.")

        if df.empty:
            raise RuntimeError("Empty dataframe for historical inference.")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        results = []

        if evaluation_date is not None:
            dates = [pd.Timestamp(evaluation_date)]
        else:
            dates = sorted(df["date"].unique())

        for date in dates:

            snapshot = df[df["date"] == date].copy()

            if snapshot.empty:
                continue

            try:
                feature_df = validate_feature_schema(
                    snapshot.loc[:, MODEL_FEATURES],
                    mode="inference"
                ).astype(DTYPE)

                probs = self.models.xgb.predict_proba(feature_df)[:, 1]
                probs = np.clip(probs, 1e-6, 1 - 1e-6)

                snapshot["score"] = probs

                weights = self._construct_portfolio(snapshot)

                for _, row in snapshot.iterrows():
                    results.append({
                        "date": row["date"],
                        "ticker": row["ticker"],
                        "score": float(row["score"]),
                        "weight": float(weights.get(row["ticker"], 0.0))
                    })

            except Exception as e:
                logger.warning(f"Historical skip {date} — {e}")
                continue

        return results

    ############################################################
    # PARALLELIZED FEATURE ORCHESTRATION
    ############################################################

    def _build_cross_sectional_frame(self, tickers: List[str]):

        end_date = pd.Timestamp.utcnow()
        start_date = end_date - pd.Timedelta(days=self.INFERENCE_LOOKBACK_DAYS)

        # 🔥 PARALLEL PRICE FETCH
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
                logger.warning(f"No price data for {ticker}")
                continue

            dataset = self.feature_store.get_features(
                price_df=price_df,
                sentiment_df=None,
                ticker=ticker,
                training=False
            )

            if dataset is None or dataset.empty:
                logger.warning(f"No features generated for {ticker}")
                continue

            datasets.append(dataset)

        if not datasets:
            raise RuntimeError("All tickers failed feature build.")

        df = pd.concat(datasets, ignore_index=True)
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        df = FeatureEngineer.add_cross_sectional_features(df)
        df = FeatureEngineer.finalize(df)

        return df

    ############################################################
    # SNAPSHOT (UNCHANGED)
    ############################################################

    def _select_latest_snapshot(self, df: pd.DataFrame):

        latest_date = df["date"].max()

        if pd.isna(latest_date):
            raise RuntimeError("No valid dates found in inference dataset.")

        now_utc = pd.Timestamp.utcnow()

        if latest_date.tzinfo is not None:
            latest_date = latest_date.tz_convert("UTC").tz_localize(None)

        if now_utc.tzinfo is not None:
            now_utc = now_utc.tz_convert("UTC").tz_localize(None)

        if (now_utc.normalize() - latest_date.normalize()) > timedelta(days=self.MAX_DATA_STALENESS_DAYS):
            raise RuntimeError("Inference data appears stale.")

        return df[df["date"] == df["date"].max()].copy()

    ############################################################
    # SCORING (UNCHANGED)
    ############################################################

    def _score_and_construct(self, latest_df):

        universe = set(MarketUniverse.get_universe())
        unknown = set(latest_df["ticker"]) - universe

        if unknown:
            raise RuntimeError(f"Unknown tickers detected: {unknown}")

        feature_df = validate_feature_schema(
            latest_df.loc[:, MODEL_FEATURES],
            mode="inference"
        ).astype(DTYPE)

        t0 = time.time()
        probs = self.models.xgb.predict_proba(feature_df)[:, 1]
        latency = time.time() - t0

        MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
        MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(latency)

        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        if np.std(probs) < self.MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected.")

        latest_df = latest_df.copy()
        latest_df["score"] = probs
        latest_df["rank_pct"] = latest_df["score"].rank(method="first", pct=True)

        latest_df["signal"] = latest_df["rank_pct"].apply(
            lambda x: "LONG" if x >= LONG_PERCENTILE
            else ("SHORT" if x <= SHORT_PERCENTILE else "NEUTRAL")
        )

        drift_result = self.drift_detector.detect(feature_df)

        if drift_result.get("drift_detected", False) and self.drift_detector.hard_fail:
            raise RuntimeError("Inference blocked due to feature drift.")

        weights = self._construct_portfolio(latest_df)

        return [
            {
                "date": row["date"],
                "ticker": row["ticker"],
                "score": float(row["score"]),
                "signal": row["signal"],
                "weight": float(weights.get(row["ticker"], 0.0))
            }
            for _, row in latest_df.iterrows()
        ]
    ############################################################
    # 🔥 INDUSTRIAL SNAPSHOT (NON-BREAKING ADDITION)
    ############################################################

    def run_snapshot(self, tickers: List[str]):
        """
        Extended inference snapshot for ML introspection.
        Does NOT modify run_batch behavior.
        """

        if len(tickers) > self.MAX_BATCH_SIZE:
            raise RuntimeError("Batch size exceeds MAX_BATCH_SIZE.")

        if not self.breaker.allow():
            raise RuntimeError("Inference blocked by circuit breaker.")

        INFERENCE_IN_PROGRESS.inc()

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

            probs = self.models.xgb.predict_proba(feature_df)[:, 1]
            probs = np.clip(probs, 1e-6, 1 - 1e-6)

            if np.std(probs) < self.MIN_PROB_STD:
                raise RuntimeError("Probability collapse detected.")

            latest_df = latest_df.copy()
            latest_df["score"] = probs
            latest_df["rank_pct"] = latest_df["score"].rank(
                method="first",
                pct=True
            )

            latest_df["signal"] = latest_df["rank_pct"].apply(
                lambda x: "LONG" if x >= LONG_PERCENTILE
                else ("SHORT" if x <= SHORT_PERCENTILE else "NEUTRAL")
            )

            weights = self._construct_portfolio(latest_df)

            # 🔥 Probability diagnostics
            prob_stats = {
                "mean": float(np.mean(probs)),
                "std": float(np.std(probs)),
                "min": float(np.min(probs)),
                "max": float(np.max(probs))
            }

            # 🔥 Long-short spread
            long_scores = latest_df[latest_df["signal"] == "LONG"]["score"]
            short_scores = latest_df[latest_df["signal"] == "SHORT"]["score"]

            spread = None
            if not long_scores.empty and not short_scores.empty:
                spread = float(long_scores.mean() - short_scores.mean())

            snapshot_rows = []

            for _, row in latest_df.iterrows():
                snapshot_rows.append({
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "score": float(row["score"]),
                    "rank_pct": float(row["rank_pct"]),
                    "signal": row["signal"],
                    "weight": float(weights.get(row["ticker"], 0.0)),
                    "volatility": float(row.get("volatility", 0.0)),
                    "momentum_20_z": float(row.get("momentum_20_z", 0.0)),
                })

            self.breaker.record_success()

            return {
                "snapshot_date": str(latest_df["date"].iloc[0]),
                "universe_size": int(len(latest_df)),
                "probability_stats": prob_stats,
                "long_short_spread": spread,
                "signals": snapshot_rows
            }

        except Exception:
            self.breaker.record_failure()
            logger.exception("Snapshot inference failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()
    ############################################################
    # PORTFOLIO (UNCHANGED)
    ############################################################

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