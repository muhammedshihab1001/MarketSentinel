# =========================================================
# HEALTH ROUTES v2.1
#
# Changes from v2.0:
# FIX 1: Response now includes data_synced field that
#         frontend Health.tsx reads to show sync status.
# FIX 2: uptime_seconds now computed from app startup_time
#         stored in app.state — was always 0 before.
# FIX 3: artifact_hash included in /health/ready response
#         so frontend can display it without a /model/info call.
# FIX 4: drift_baseline_loaded field added — checks if
#         baseline.json exists on disk.
# =========================================================

import time
import os
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.health")

router = APIRouter()

DRIFT_BASELINE_PATH = os.getenv(
    "DRIFT_BASELINE_PATH",
    "artifacts/drift/baseline.json",
)


def _get_uptime(request: Request) -> float:
    """Return seconds since app startup."""
    try:
        startup_time = request.app.state.startup_time
        return round(time.time() - startup_time, 1)
    except AttributeError:
        return 0.0


def _get_model_version(request: Request) -> str:
    try:
        loader = request.app.state.model_loader
        return loader.version or "unknown"
    except AttributeError:
        return "unknown"


def _get_artifact_hash(request: Request) -> str:
    try:
        loader = request.app.state.model_loader
        return loader.artifact_hash or "unknown"
    except AttributeError:
        return "unknown"


def _get_model_loaded(request: Request) -> bool:
    try:
        loader = request.app.state.model_loader
        return loader.is_loaded()
    except AttributeError:
        return False


def _get_redis_connected(request: Request) -> bool:
    try:
        cache = request.app.state.cache
        return cache.ping()
    except AttributeError:
        return False


def _get_db_connected(request: Request) -> bool:
    try:
        from core.db.engine import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception:
        return False


def _get_data_synced(request: Request) -> bool:
    """
    Returns True if at least one ticker has been synced to the DB.
    Quick check — counts rows in ohlcv_daily.
    """
    try:
        from core.db.engine import get_engine
        import sqlalchemy as sa
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                sa.text("SELECT COUNT(*) FROM ohlcv_daily LIMIT 1")
            )
            count = result.scalar()
            return (count or 0) > 0
    except Exception:
        return False


def _get_drift_baseline_loaded() -> bool:
    return os.path.exists(DRIFT_BASELINE_PATH)


# =========================================================
# GET /health/ready
# Primary health check — used by Docker healthcheck
# and frontend Health page.
# =========================================================

@router.get("/health/ready")
async def health_ready(request: Request):
    endpoint = "/health/ready"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        models_loaded = _get_model_loaded(request)
        redis_connected = _get_redis_connected(request)
        db_connected = _get_db_connected(request)
        # FIX: data_synced now included
        data_synced = _get_data_synced(request)
        drift_baseline_loaded = _get_drift_baseline_loaded()

        ready = models_loaded and db_connected

        response_data = {
            "ready": ready,
            "models_loaded": models_loaded,
            "redis_connected": redis_connected,
            "db_connected": db_connected,
            # FIX: New field — frontend Health.tsx reads this
            "data_synced": data_synced,
            "drift_baseline_loaded": drift_baseline_loaded,
            # FIX: Model version and artifact hash now in response
            "model_version": _get_model_version(request),
            "artifact_hash": _get_artifact_hash(request),
            # FIX: Real uptime from app startup time
            "uptime_seconds": _get_uptime(request),
        }

        status_code = 200 if ready else 503
        return JSONResponse(content=response_data, status_code=status_code)

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Health check failed")
        return JSONResponse(
            content={"ready": False, "error": str(e)},
            status_code=503,
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /health/live
# Liveness probe — just confirms the process is running.
# Does NOT check DB or model.
# =========================================================

@router.get("/health/live")
async def health_live():
    return JSONResponse(content={"alive": True})


# =========================================================
# GET /health/db
# Database-specific health check.
# =========================================================

@router.get("/health/db")
async def health_db(request: Request):
    endpoint = "/health/db"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        from core.db.engine import get_engine
        import sqlalchemy as sa

        engine = get_engine()
        t0 = time.time()
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        latency_ms = round((time.time() - t0) * 1000, 2)

        return JSONResponse(content={
            "db_connected": True,
            "latency_ms": latency_ms,
        })

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        return JSONResponse(
            content={"db_connected": False, "error": str(e)},
            status_code=503,
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /health/model
# Model-specific health check.
# =========================================================

@router.get("/health/model")
async def health_model(request: Request):
    endpoint = "/health/model"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    try:
        loader = request.app.state.model_loader
        return JSONResponse(content={
            "model_loaded": loader.is_loaded(),
            "model_version": loader.version or "unknown",
            "artifact_hash": loader.artifact_hash or "unknown",
        })
    except AttributeError:
        return JSONResponse(
            content={"model_loaded": False, "model_version": "unknown"},
            status_code=503,
        )