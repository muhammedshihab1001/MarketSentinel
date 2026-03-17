import logging
import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any

from app.inference.pipeline import get_shared_model_loader
from core.schema.feature_schema import MODEL_FEATURES
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.model_info")

REQUEST_TIMEOUT = 20
MAX_CONCURRENT = 4

model_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# MODEL INFO
# =========================================================

@router.get("/model-info")
async def model_info():

    endpoint = "/model-info"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_model_info_sync),
                timeout=REQUEST_TIMEOUT
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
    loader = get_shared_model_loader()

    return {
        "model_version": loader.xgb_version,
        "schema_signature": loader.schema_signature,
        "dataset_hash": loader.dataset_hash,
        "training_code_hash": loader.training_code_hash,
        "artifact_hash": loader.artifact_hash,
        "feature_checksum": loader.feature_checksum,
        "feature_count": len(MODEL_FEATURES),
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time())
    }


# =========================================================
# FEATURE IMPORTANCE (FIXED CONTRACT)
# =========================================================

@router.get("/feature-importance")
async def feature_importance():

    endpoint = "/feature-importance"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_feature_importance_sync),
                timeout=REQUEST_TIMEOUT
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
    loader = get_shared_model_loader()
    model = loader.xgb

    # FIX: get_feature_importance returns a list of dicts, not a rich dict
    importance_list = loader.get_feature_importance()

    importance = [
        {"feature": item["feature"], "importance": float(item["importance"])}
        for item in importance_list
    ]

    return {
        "model_version": loader.xgb_version,
        "feature_checksum": loader.feature_checksum,
        "best_iteration": getattr(model, "best_iteration", None),
        "training_fingerprint": getattr(model, "training_fingerprint", None),
        "importance": importance,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time())
    }


# =========================================================
# MODEL DIAGNOSTICS
# =========================================================

@router.get("/model-diagnostics")
async def model_diagnostics() -> Dict[str, Any]:

    endpoint = "/model-diagnostics"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_model_diagnostics_sync),
                timeout=REQUEST_TIMEOUT
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
    loader = get_shared_model_loader()
    model = loader.xgb

    return {
        "model_version": loader.xgb_version,
        "artifact_hash": loader.artifact_hash,
        "schema_signature": loader.schema_signature,
        "dataset_hash": loader.dataset_hash,
        "training_code_hash": loader.training_code_hash,
        "feature_checksum": loader.feature_checksum,
        "feature_count": len(MODEL_FEATURES),

        # Model internals
        "training_fingerprint": getattr(model, "training_fingerprint", None),
        "training_cols": getattr(model, "training_cols", None),
        "param_checksum": getattr(model, "param_checksum", None),
        "booster_checksum": getattr(model, "booster_checksum", None),
        "best_iteration": getattr(model, "best_iteration", None),

        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time())
    }