# =========================================================
# HEALTH ROUTES v2.2
# SWAGGER FIX: Added tags, summary, description
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

router = APIRouter(tags=["health"])

DRIFT_BASELINE_PATH = os.getenv(
    "DRIFT_BASELINE_PATH",
    "artifacts/drift/baseline.json",
)


def _get_uptime(request: Request) -> float:
    try:
        return round(time.time() - request.app.state.startup_time, 1)
    except AttributeError:
        return 0.0


def _get_model_version(request: Request) -> str:
    try:
        return request.app.state.model_loader.version or "unknown"
    except AttributeError:
        return "unknown"


def _get_artifact_hash(request: Request) -> str:
    try:
        return request.app.state.model_loader.artifact_hash or "unknown"
    except AttributeError:
        return "unknown"


def _get_model_loaded(request: Request) -> bool:
    try:
        return request.app.state.model_loader.is_loaded()
    except AttributeError:
        return False


def _get_redis_connected(request: Request) -> bool:
    try:
        return request.app.state.cache.ping()
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
    try:
        from core.db.engine import get_engine
        import sqlalchemy as sa
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT COUNT(*) FROM ohlcv_daily LIMIT 1"))
            count = result.scalar()
            return (count or 0) > 0
    except Exception:
        return False


def _get_drift_baseline_loaded() -> bool:
    return os.path.exists(DRIFT_BASELINE_PATH)


@router.get(
    "/health/ready",
    summary="Readiness Check",
    description="""
Primary health check used by Docker healthcheck and the frontend Health page.

Returns **200** when ready (model loaded + DB connected).
Returns **503** when not ready.

**Fields:**
- `ready`: overall readiness (model + DB must both be true)
- `models_loaded`: XGBoost model loaded from artifacts
- `redis_connected`: Redis cache available (degraded mode if false)
- `db_connected`: PostgreSQL connection healthy
- `data_synced`: at least 1 ticker has OHLCV data in DB
- `drift_baseline_loaded`: baseline.json exists on disk
- `uptime_seconds`: seconds since boot

**No authentication required.**
""",
    response_description="200 = ready, 503 = not ready.",
)
async def health_ready(request: Request):
    endpoint = "/health/ready"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        models_loaded = _get_model_loaded(request)
        redis_connected = _get_redis_connected(request)
        db_connected = _get_db_connected(request)
        data_synced = _get_data_synced(request)
        drift_baseline_loaded = _get_drift_baseline_loaded()

        ready = models_loaded and db_connected

        response_data = {
            "ready": ready,
            "models_loaded": models_loaded,
            "redis_connected": redis_connected,
            "db_connected": db_connected,
            "data_synced": data_synced,
            "drift_baseline_loaded": drift_baseline_loaded,
            "model_version": _get_model_version(request),
            "artifact_hash": _get_artifact_hash(request),
            "uptime_seconds": _get_uptime(request),
        }

        return JSONResponse(content=response_data, status_code=200 if ready else 503)

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Health check failed")
        return JSONResponse(
            content={"ready": False, "error": str(e)},
            status_code=503,
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


@router.get(
    "/health/live",
    summary="Liveness Probe",
    description="""
Kubernetes/Docker liveness probe. Returns 200 if the process is running.
Does NOT check DB or model — use /health/ready for full status.

**No authentication required.**
""",
)
async def health_live():
    return JSONResponse(content={"alive": True})


@router.get(
    "/health/db",
    summary="Database Health",
    description="""
Checks PostgreSQL connectivity and returns query latency.
Returns 200 if connected, 503 if not.

**No authentication required.**
""",
)
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


@router.get(
    "/health/model",
    summary="Model Health",
    description="""
Returns loaded model version and artifact hash.
Returns 503 if no model is loaded.

**No authentication required.**
""",
)
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
