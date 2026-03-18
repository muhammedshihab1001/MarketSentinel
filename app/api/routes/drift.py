# =========================================================
# DRIFT STATUS ROUTE v2.3
# FIX: reads drift from cached snapshot (Redis) — was 18s, now <100ms
# Falls back to full computation only if cache is empty
# =========================================================

import asyncio
import time
import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.cache import RedisCache
from app.inference.pipeline import get_shared_model_loader
from core.market.universe import MarketUniverse
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    DTYPE,
)
from core.monitoring.drift_detector import DriftDetector
from core.monitoring.retrain_trigger import RetrainTrigger
from core.logging.logger import get_logger

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

router = APIRouter()
logger = get_logger("marketsentinel.drift")

REQUEST_TIMEOUT = 180
MAX_CONCURRENT = 2
MIN_UNIVERSE_WIDTH = 10

# Cache drift result for 5 minutes — avoids re-running full inference on every call
DRIFT_CACHE_TTL = int(os.getenv("DRIFT_CACHE_TTL", "300"))

drift_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# BASELINE META HELPER
# =========================================================

def _get_baseline_meta():

    detector = DriftDetector()
    path = detector.BASELINE_PATH

    if not os.path.exists(path):
        return {
            "baseline_exists": False,
            "baseline_version": None,
            "baseline_model_version": None,
            "baseline_dataset_hash": None,
        }

    try:
        with open(path, encoding="utf-8") as f:
            baseline = json.load(f)

        meta = baseline.get("meta", {})

        return {
            "baseline_exists": True,
            "baseline_version": meta.get("baseline_version"),
            "baseline_model_version": meta.get("model_version"),
            "baseline_dataset_hash": meta.get("dataset_hash"),
        }

    except Exception:
        return {
            "baseline_exists": False,
            "baseline_version": None,
            "baseline_model_version": None,
            "baseline_dataset_hash": None,
        }


# =========================================================
# DRIFT FROM CACHED SNAPSHOT (fast path)
# =========================================================

def _drift_from_cache():
    """
    Reads drift state from the background snapshot cached in Redis.
    The background_snapshot_loop in main.py runs every 120s and
    stores the full snapshot including drift. This avoids re-running
    full inference (18s) on every drift endpoint call.
    """

    cache = RedisCache()

    if not cache.enabled:
        return None

    # Try the background snapshot cache key first
    bg_key = cache.build_key({"type": "background_snapshot"})
    snapshot = cache.get(bg_key)

    if snapshot and isinstance(snapshot, dict):
        drift = snapshot.get("drift")
        if drift and isinstance(drift, dict) and "drift_state" in drift:
            logger.info("Drift served from background snapshot cache")
            return drift

    return None


# =========================================================
# DRIFT FULL COMPUTATION (slow path — fallback only)
# =========================================================

def _drift_full_compute():
    """
    Full drift computation — only runs when Redis cache is empty.
    This is the original 18s path. Should rarely be needed.
    """

    from app.api.routes.predict import get_pipeline

    start_time = time.time()

    pipeline = get_pipeline()
    tickers = MarketUniverse.get_universe()

    if not tickers or len(tickers) < MIN_UNIVERSE_WIDTH:
        raise RuntimeError("Universe too small for drift detection.")

    import pandas as pd
    df = pipeline._build_cross_sectional_frame(tickers)

    if df.empty:
        raise RuntimeError("Feature frame empty.")

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    if latest_df.empty:
        raise RuntimeError("No latest snapshot available for drift.")

    feature_df = validate_feature_schema(
        latest_df.loc[:, MODEL_FEATURES],
        mode="inference",
    ).astype(DTYPE)

    try:
        drift_result = pipeline.drift_detector.detect(feature_df)
    except Exception as e:
        logger.warning("Drift detector failure: %s", str(e))
        drift_result = {
            "drift_detected": False,
            "severity_score": 0.0,
            "drift_state": "baseline_missing",
            "exposure_scale": 1.0,
        }

    logger.info(
        "Drift full computation | latency=%.2fs",
        time.time() - start_time,
    )

    return drift_result, len(latest_df)


# =========================================================
# SHARED SYNC LOGIC
# =========================================================

def _drift_status_sync():

    start_time = time.time()

    loader = get_shared_model_loader()

    # Fast path — try Redis cache first
    cached_drift = _drift_from_cache()

    if cached_drift:
        drift_result = cached_drift
        universe_size = len(MarketUniverse.get_universe())
        snapshot_date = drift_result.get("snapshot_date", "unknown")
    else:
        # Slow path — full computation
        logger.info("Drift cache miss — running full computation")
        drift_result, universe_size = _drift_full_compute()
        snapshot_date = "computed_live"

    retrain_trigger = RetrainTrigger()
    retrain_info = retrain_trigger.evaluate(drift_result)

    baseline_meta = _get_baseline_meta()

    return {
        "drift_detected": drift_result.get("drift_detected", False),
        "severity_score": drift_result.get("severity_score", 0),
        "drift_confidence": drift_result.get("drift_confidence", 0),
        "drift_state": drift_result.get("drift_state", "unknown"),
        "exposure_scale": drift_result.get("exposure_scale", 1.0),
        "coverage": drift_result.get("coverage", 0),
        "retrain_required": retrain_info["retrain_required"],
        "retrain_events": retrain_info["events"],
        "cooldown_active": retrain_info.get("cooldown_active", False),
        "cooldown_remaining_seconds": retrain_info.get("cooldown_remaining_seconds", 0),
        **baseline_meta,
        "model_version": loader.xgb_version,
        "schema_signature": loader.schema_signature,
        "artifact_hash": loader.artifact_hash,
        "dataset_hash": loader.dataset_hash,
        "universe_size": universe_size,
        "snapshot_date": snapshot_date,
        "served_from_cache": cached_drift is not None,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# GET /drift  (frontend calls this)
# =========================================================

@router.get("/drift")
async def drift():

    endpoint = "/drift"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with drift_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_drift_status_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Drift check timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Drift status check failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /drift-status  (backward compatibility)
# =========================================================

@router.get("/drift-status")
async def drift_status():

    endpoint = "/drift-status"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with drift_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_drift_status_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Drift check timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Drift status check failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)