# =========================================================
# DRIFT STATUS ROUTE v2.7
#
# FIX v2.7: Route changed from @router.get("") back to
#   @router.get("/drift") because main.py includes this
#   router WITHOUT a prefix:
#     app.include_router(drift.router)   ← no prefix=
#   FastAPI raises FastAPIError when BOTH prefix and path
#   are empty. Since main.py has no prefix, the route
#   path itself must carry "/drift".
#
# All other fixes from v2.6 retained:
#   - cache.ping() not cache.enabled
#   - get_model_loader() not get_shared_model_loader()
#   - loader.version / loader.metadata not loader.xgb_version
#   - trigger.evaluate(int) not trigger.evaluate(dict)
#   - snapshot["snapshot"]["drift"] not snapshot["drift"]
# =========================================================

import asyncio
import time
import os
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from core.monitoring.drift_detector import DriftDetector
from core.monitoring.retrain_trigger import RetrainTrigger
from core.market.universe import MarketUniverse
from core.logging.logger import get_logger

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

router = APIRouter()
logger = get_logger("marketsentinel.drift")

REQUEST_TIMEOUT = 60
MAX_CONCURRENT = 2

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"

drift_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# BASELINE META
# =========================================================

def _get_baseline_meta() -> dict:
    try:
        detector = DriftDetector()
        path = detector.BASELINE_PATH

        if not os.path.exists(path):
            return {
                "baseline_exists": False,
                "baseline_version": None,
                "baseline_model_version": None,
                "baseline_dataset_hash": None,
            }

        with open(path, encoding="utf-8") as f:
            baseline = json.load(f)
        meta = baseline.get("meta", {})
        return {
            "baseline_exists": True,
            "baseline_version": meta.get("baseline_version"),
            "baseline_model_version": meta.get("model_version"),
            "baseline_dataset_hash": meta.get("dataset_hash"),
        }
    except Exception as e:
        logger.debug("Baseline meta unavailable: %s", e)
        return {
            "baseline_exists": False,
            "baseline_version": None,
            "baseline_model_version": None,
            "baseline_dataset_hash": None,
        }


# =========================================================
# CACHE READ
# =========================================================

def _drift_from_cache(cache):
    """
    Read drift block from background snapshot.
    Returns (drift_dict, full_snapshot) or None.
    """
    try:
        if not cache.ping():
            return None

        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)

        if not snapshot_result or not isinstance(snapshot_result, dict):
            return None

        drift = (
            snapshot_result
            .get("snapshot", {})
            .get("drift", {})
        )

        if drift and isinstance(drift, dict) and "drift_state" in drift:
            logger.info("Drift served from background snapshot cache")
            return drift, snapshot_result

        return None

    except Exception as e:
        logger.debug("Cache drift read failed: %s", e)
        return None


# =========================================================
# SYNC HANDLER
# =========================================================

def _drift_status_sync(cache) -> dict:

    start_time = time.time()

    # Model loader attributes
    model_version = ""
    schema_signature = ""
    artifact_hash = ""
    dataset_hash = ""

    try:
        from app.inference.model_loader import get_model_loader
        loader = get_model_loader()
        model_version = getattr(loader, "version", "") or ""
        meta = getattr(loader, "metadata", {}) or {}
        schema_signature = meta.get("schema_signature", "")
        artifact_hash = getattr(loader, "artifact_hash", "") or ""
        dataset_hash = meta.get("dataset_hash", "")
    except Exception as e:
        logger.debug("Model loader unavailable for drift: %s", e)

    # Try cache fast path
    cache_result = _drift_from_cache(cache)
    served_from_cache = False
    snapshot_date = "unknown"
    universe_size = len(MarketUniverse.get_universe())

    if cache_result is not None:
        drift_result, snapshot_result = cache_result
        served_from_cache = True
        snapshot_date = (
            snapshot_result
            .get("snapshot", {})
            .get("snapshot_date", "unknown")
        )
    else:
        logger.info("Drift cache miss — reading from DriftDetector.health()")
        try:
            detector = DriftDetector()
            health = detector.health()
            drift_result = {
                "drift_detected": health.get("drift_detected", False),
                "severity_score": health.get("severity_score", 0),
                "drift_state": health.get("drift_state", "none"),
                "drift_confidence": health.get("drift_confidence", 0.0),
                "exposure_scale": health.get("exposure_scale", 1.0),
                "coverage": health.get("coverage", 0),
            }
        except Exception as e:
            logger.warning("DriftDetector.health() failed: %s", e)
            drift_result = {
                "drift_detected": False,
                "severity_score": 0,
                "drift_state": "unavailable",
                "drift_confidence": 0.0,
                "exposure_scale": 1.0,
                "coverage": 0,
            }

    severity_score = int(drift_result.get("severity_score", 0))

    # RetrainTrigger
    retrain_required = False
    cooldown_remaining_seconds = 0
    cooldown_active = False

    try:
        trigger = RetrainTrigger()
        retrain_required = bool(trigger.evaluate(severity_score))
        cooldown_remaining_seconds = max(0, int(trigger.cooldown_remaining()))
        cooldown_active = cooldown_remaining_seconds > 0
    except Exception as e:
        logger.debug("RetrainTrigger unavailable: %s", e)

    baseline_meta = _get_baseline_meta()

    return {
        "drift_detected": bool(drift_result.get("drift_detected", False)),
        "severity_score": severity_score,
        "drift_confidence": round(float(drift_result.get("drift_confidence", 0.0)), 4),
        "drift_state": drift_result.get("drift_state", "none"),
        "exposure_scale": round(float(drift_result.get("exposure_scale", 1.0)), 4),
        "coverage": drift_result.get("coverage", 0),
        "retrain_required": retrain_required,
        "cooldown_active": cooldown_active,
        "cooldown_remaining_seconds": cooldown_remaining_seconds,
        **baseline_meta,
        "model_version": model_version,
        "schema_signature": schema_signature,
        "artifact_hash": artifact_hash,
        "dataset_hash": dataset_hash,
        "universe_size": universe_size,
        "snapshot_date": snapshot_date,
        "served_from_cache": served_from_cache,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# GET /drift
# FIX v2.7: Path is "/drift" not "" because main.py does
#   app.include_router(drift.router) with NO prefix.
#   Both prefix="" and path="" → FastAPIError on startup.
# =========================================================

@router.get("/drift")
async def get_drift(request: Request):
    """
    Returns current model drift state and retrain trigger status.

    severity_score: integer 0-15 — NOT a percentage.
    drift_state: none | low | moderate | high | critical | unavailable
    """
    endpoint = "/drift"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        cache = getattr(request.app.state, "cache", None)
        if cache is None:
            from app.inference.cache import RedisCache
            cache = RedisCache()

        async with drift_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_drift_status_sync, cache),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Drift check timed out")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Drift route failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


@router.get("/drift-status")
async def drift_status(request: Request):
    """Backward compat alias for GET /drift."""
    return await get_drift(request)