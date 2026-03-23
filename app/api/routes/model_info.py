# =========================================================
# MODEL INFO ROUTE v2.4
# FIX: get_shared_model_loader → get_model_loader
# FIX: loader.xgb_version → loader.version
# FIX: loader.xgb → loader.model
# FIX: metadata fields read from loader.metadata dict
# =========================================================

import asyncio
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any

from app.inference.model_loader import get_model_loader
from core.schema.feature_schema import MODEL_FEATURES
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.model_info")

REQUEST_TIMEOUT = 20
MAX_CONCURRENT = 4

model_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# MODEL INFO  →  GET /model/info
# =========================================================

@router.get("/info")
async def model_info():

    endpoint = "/model/info"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_model_info_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Model info timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Model info retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _model_info_sync():

    start_time = time.time()
    loader = get_model_loader()
    meta = loader.metadata or {}

    return {
        "model_version": loader.version or "unknown",
        "schema_signature": loader.schema_signature or "unknown",
        "dataset_hash": meta.get("dataset_hash", "unknown"),
        "training_code_hash": meta.get("training_code_hash", "unknown"),
        "artifact_hash": loader.artifact_hash or "unknown",
        "feature_checksum": meta.get("feature_checksum", "unknown"),
        "feature_count": len(MODEL_FEATURES),
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# FEATURE IMPORTANCE  →  GET /model/feature-importance
# =========================================================

@router.get("/feature-importance")
async def feature_importance():

    endpoint = "/model/feature-importance"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_feature_importance_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Feature importance timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Feature importance retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _feature_importance_sync():

    start_time = time.time()
    loader = get_model_loader()
    model = loader.model
    meta = loader.metadata or {}

    importance = []
    try:
        if model is not None and hasattr(model, "export_feature_importance"):
            raw = model.export_feature_importance()
            importance = [
                {"feature": item["feature"], "importance": float(item["importance"])}
                for item in raw.get("feature_importance", [])
            ]
        elif model is not None and hasattr(model, "model") and model.model is not None:
            scores = model.model.get_score(importance_type="gain")
            total = sum(scores.values()) or 1.0
            importance = sorted(
                [{"feature": f, "importance": round(v / total, 6)} for f, v in scores.items()],
                key=lambda x: x["importance"],
                reverse=True,
            )
    except Exception as e:
        logger.warning("Feature importance extraction failed: %s", e)

    return {
        "model_version": loader.version or "unknown",
        "feature_checksum": meta.get("feature_checksum", "unknown"),
        "best_iteration": getattr(model, "best_iteration", None) if model else None,
        "training_fingerprint": getattr(model, "training_fingerprint", None) if model else None,
        "importance": importance,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# MODEL DIAGNOSTICS  →  GET /model/diagnostics
# =========================================================

@router.get("/diagnostics")
async def model_diagnostics() -> Dict[str, Any]:

    endpoint = "/model/diagnostics"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_model_diagnostics_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Model diagnostics timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Model diagnostics failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _model_diagnostics_sync():

    start_time = time.time()
    loader = get_model_loader()
    model = loader.model
    meta = loader.metadata or {}

    return {
        "model_version": loader.version or "unknown",
        "artifact_hash": loader.artifact_hash or "unknown",
        "schema_signature": loader.schema_signature or "unknown",
        "dataset_hash": meta.get("dataset_hash", "unknown"),
        "training_code_hash": meta.get("training_code_hash", "unknown"),
        "feature_checksum": meta.get("feature_checksum", "unknown"),
        "feature_count": len(MODEL_FEATURES),
        "training_fingerprint": getattr(model, "training_fingerprint", None) if model else None,
        "training_cols": getattr(model, "training_cols", None) if model else None,
        "param_checksum": getattr(model, "param_checksum", None) if model else None,
        "booster_checksum": getattr(model, "booster_checksum", None) if model else None,
        "best_iteration": getattr(model, "best_iteration", None) if model else None,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }