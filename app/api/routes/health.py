import time
import logging
from fastapi import APIRouter

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

from app.inference.pipeline import get_shared_model_loader
from core.schema.feature_schema import get_schema_signature

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger("marketsentinel.health")


# =========================================================
# LIVENESS
# =========================================================

@router.get("/live")
def liveness():
    """
    Basic container liveness probe.
    Used by orchestrators to verify service is alive.
    """

    endpoint = "/health/live"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        return {
            "status": "alive",
            "service": "MarketSentinel",
            "timestamp": int(time.time())
        }

    except Exception:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Health check failure")
        return {
            "status": "error"
        }

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# READINESS
# =========================================================

@router.get("/ready")
def readiness():
    """
    Readiness probe.
    Ensures model and critical dependencies are loaded.
    """

    endpoint = "/health/ready"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        loader = get_shared_model_loader()

        # Basic loader validation
        if loader.xgb is None:
            raise RuntimeError("Model not loaded")

        if loader.schema_signature != get_schema_signature():
            raise RuntimeError("Schema signature mismatch")

        return {
            "status": "ready",
            "model_version": loader.xgb_version,
            "schema_signature": loader.schema_signature,
            "artifact_hash": loader.artifact_hash,
            "timestamp": int(time.time())
        }

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Readiness check failed")
        return {
            "status": "not_ready",
            "error": str(e)
        }

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# MODEL HEALTH
# =========================================================

@router.get("/model")
def model_health():
    """
    Deep model health inspection.
    Used for governance verification.
    """

    endpoint = "/health/model"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        loader = get_shared_model_loader()
        model = loader.xgb

        return {
            "status": "ok",
            "model_version": loader.xgb_version,
            "artifact_hash": loader.artifact_hash,
            "schema_signature": loader.schema_signature,
            "dataset_hash": loader.dataset_hash,
            "training_code_hash": loader.training_code_hash,
            "booster_checksum": getattr(model, "booster_checksum", None),
            "best_iteration": getattr(model, "best_iteration", None),
            "feature_count": getattr(model, "training_cols", None),
            "timestamp": int(time.time())
        }

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Model health check failed")
        return {
            "status": "error",
            "error": str(e)
        }

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )