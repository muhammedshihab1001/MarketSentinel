# =========================================================
# DRIFT STATUS ROUTE v2.6
#
# Updated from git history v2.5 — fixes all broken refs:
#
# FIX 1: cache.enabled doesn't exist → use cache.ping()
# FIX 2: get_shared_model_loader() → get_model_loader()
# FIX 3: loader.xgb_version → loader.version
#         loader.schema_signature → loader.metadata.get(...)
#         loader.artifact_hash stays as-is
#         loader.dataset_hash → loader.metadata.get(...)
# FIX 4: retrain_trigger.evaluate(drift_result) old dict API
#         → trigger.evaluate(severity_score: int) + separate
#           trigger.cooldown_remaining()
# FIX 5: Route was @router.get("/drift") but router is
#         mounted at prefix="/drift" in main.py — caused
#         404 because it registered as /drift/drift.
#         Changed to @router.get("") to match prefix.
# FIX 6: _drift_from_cache used wrong snapshot structure
#         — drift block lives at snapshot["snapshot"]["drift"]
#         not snapshot["drift"].
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
# FAST PATH — READ FROM BACKGROUND SNAPSHOT CACHE
# =========================================================

def _drift_from_cache(cache):
    """
    Read drift block from background snapshot.
    FIX 1: cache.enabled doesn't exist — use cache.ping()
    FIX 6: drift lives at snapshot["snapshot"]["drift"]
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

    # ── FIX 2+3: Correct model loader import and attributes ─
    model_version = ""
    schema_signature = ""
    artifact_hash = ""
    dataset_hash = ""

    try:
        from app.inference.model_loader import get_model_loader
        loader = get_model_loader()
        # FIX 3: correct attribute names on ModelLoader
        model_version = getattr(loader, "version", "") or ""
        meta = getattr(loader, "metadata", {}) or {}
        schema_signature = meta.get("schema_signature", "")
        artifact_hash = getattr(loader, "artifact_hash", "") or ""
        dataset_hash = meta.get("dataset_hash", "")
    except Exception as e:
        logger.debug("Model loader unavailable for drift: %s", e)

    # ── Try cache fast path ───────────────────────────────
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
        # No snapshot cached yet — read directly from detector
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

    # ── FIX 4: Correct RetrainTrigger API ─────────────────
    retrain_required = False
    cooldown_remaining_seconds = 0
    cooldown_active = False

    try:
        trigger = RetrainTrigger()
        # FIX 4: evaluate() takes severity_score int, not full dict
        retrain_required = bool(trigger.evaluate(severity_score))
        cooldown_remaining_seconds = max(0, int(trigger.cooldown_remaining()))
        cooldown_active = cooldown_remaining_seconds > 0
    except Exception as e:
        logger.debug("RetrainTrigger unavailable: %s", e)

    baseline_meta = _get_baseline_meta()

    return {
        # Core drift
        "drift_detected": bool(drift_result.get("drift_detected", False)),
        "severity_score": severity_score,
        "drift_confidence": round(float(drift_result.get("drift_confidence", 0.0)), 4),
        "drift_state": drift_result.get("drift_state", "none"),
        "exposure_scale": round(float(drift_result.get("exposure_scale", 1.0)), 4),
        "coverage": drift_result.get("coverage", 0),

        # Retrain trigger
        "retrain_required": retrain_required,
        "cooldown_active": cooldown_active,
        "cooldown_remaining_seconds": cooldown_remaining_seconds,

        # Baseline
        **baseline_meta,

        # Model meta
        "model_version": model_version,
        "schema_signature": schema_signature,
        "artifact_hash": artifact_hash,
        "dataset_hash": dataset_hash,

        # Context
        "universe_size": universe_size,
        "snapshot_date": snapshot_date,
        "served_from_cache": served_from_cache,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# GET /drift
# FIX 5: "" not "/drift" — router is mounted at
#         prefix="/drift" in main.py already.
# =========================================================

@router.get("")
@router.get("/")
async def get_drift(request: Request):
    """
    Returns current model drift state and retrain trigger status.

    severity_score: integer 0-15. NOT a percentage.
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


@router.get("/status")
async def drift_status(request: Request):
    """Backward compat alias for GET /drift."""
    return await get_drift(request)