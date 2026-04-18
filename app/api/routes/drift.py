# =========================================================
# DRIFT STATUS ROUTE v2.8
# SWAGGER FIX: Added tags, summary, description
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

router = APIRouter(tags=["drift"])
logger = get_logger("marketsentinel.drift")

REQUEST_TIMEOUT = 60
MAX_CONCURRENT = 2

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"

drift_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


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


def _drift_from_cache(cache):
    try:
        if not cache.ping():
            return None

        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)

        if not snapshot_result or not isinstance(snapshot_result, dict):
            return None

        drift = snapshot_result.get("snapshot", {}).get("drift", {})

        if drift and isinstance(drift, dict) and "drift_state" in drift:
            return drift, snapshot_result

        return None

    except Exception as e:
        logger.debug("Cache drift read failed: %s", e)
        return None


def _drift_status_sync(cache) -> dict:
    start_time = time.time()

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

    cache_result = _drift_from_cache(cache)
    served_from_cache = False
    snapshot_date = "unknown"
    universe_size = len(MarketUniverse.get_universe())

    if cache_result is not None:
        drift_result, snapshot_result = cache_result
        served_from_cache = True
        snapshot_date = snapshot_result.get("snapshot", {}).get(
            "snapshot_date", "unknown"
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

    retrain_required = False
    cooldown_remaining_seconds = 0
    cooldown_active = False

    try:
        trigger = RetrainTrigger()
        _retrain_result = trigger.evaluate(severity_score)
        retrain_required = bool(_retrain_result.get("retrain_required", False))
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


@router.get(
    "/drift",
    summary="Model Drift State",
    description="""
Returns the current model drift state and retrain trigger status.

**severity_score:** Integer 0–15. NOT a percentage. Display as `"7 / 15"`.
| Score | State | Meaning |
|---|---|---|
| 0–4 | none/low | Model stable |
| 5–8 | moderate | Monitor closely |
| 9–12 | high | Consider retraining |
| 13–15 | critical | Retrain immediately |

**drift_state:** `none` | `low` | `moderate` | `high` | `critical` | `unavailable`

**retrain_required:** True when severity_score >= DRIFT_RETRAIN_THRESHOLD (default: 8)

**baseline_exists:** False = no baseline.json on disk. Run training to create one.

**Requires:** Owner or Demo authentication (counts against `drift` quota).
""",
    response_description="Drift state, severity score, retrain trigger, and baseline info.",
)
async def get_drift(request: Request):
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


@router.get(
    "/drift-status",
    summary="Drift Status (alias)",
    description="Backward compatibility alias for GET /drift.",
    include_in_schema=False,
)
async def drift_status(request: Request):
    return await get_drift(request)
