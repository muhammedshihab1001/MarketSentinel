# =========================================================
# HEALTH ROUTES v2.0 — with DB health check
# =========================================================

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

from app.inference.pipeline import get_shared_model_loader
from core.schema.feature_schema import get_schema_signature
from core.db.engine import check_db_health
from core.logging.logger import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger("marketsentinel.health")


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
            "timestamp": int(time.time()),
        }

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Health liveness check failure")

        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
            },
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# READINESS (now includes DB health)
# =========================================================

@router.get("/ready")
def readiness():
    """
    Readiness probe.
    Ensures model, database, and critical dependencies are loaded.
    """

    endpoint = "/health/ready"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        loader = get_shared_model_loader()

        if loader.xgb is None:
            raise RuntimeError("Model not loaded")

        if loader.schema_signature != get_schema_signature():
            raise RuntimeError("Schema signature mismatch")

        # DB health check
        db_status = check_db_health()

        return {
            "status": "ready",
            "model_version": loader.xgb_version,
            "schema_signature": loader.schema_signature,
            "artifact_hash": loader.artifact_hash,
            "database": db_status,
            "timestamp": int(time.time()),
        }

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Readiness check failed")

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
            },
        )

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

        if model is None:
            raise RuntimeError("Model not loaded")

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
            "timestamp": int(time.time()),
        }

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Model health check failed")

        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
            },
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# DATABASE HEALTH (new endpoint)
# =========================================================

@router.get("/db")
def database_health():
    """
    Database health check endpoint.
    Returns PostgreSQL connection status and latency.
    """

    endpoint = "/health/db"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        db_status = check_db_health()

        return {
            "status": db_status["status"],
            "latency_ms": db_status.get("latency_ms"),
            "error": db_status.get("error"),
            "timestamp": int(time.time()),
        }

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Database health check failed")

        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
            },
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )