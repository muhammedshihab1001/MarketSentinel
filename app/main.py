# =====================================================
# MARKET SENTINEL APPLICATION ENTRYPOINT v2.1
# Hybrid Multi-Agent | CV-Optimized Architecture
# =====================================================

import logging
import time
import gc
import hashlib
import os
import psutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest

from core.config.env_loader import init_env, get_int, get_env, get_bool
from core.schema.feature_schema import get_schema_signature

from app.api.routes import (
    drift,
    model_info,
    portfolio,
    universe,
    health,
    predict,
    performance,
    equity,
    agent
)

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache
from core.monitoring.drift_detector import DriftDetector


# =====================================================
# ENV INITIALIZATION
# =====================================================

init_env()

logger = logging.getLogger("marketsentinel")


# =====================================================
# SYSTEM METADATA
# =====================================================

BOOT_ID = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
STARTUP_TIMEOUT_SEC = get_int("STARTUP_TIMEOUT_SEC", 120)
APP_VERSION = get_env("APP_VERSION", "4.1.0")

CORS_ORIGINS = get_env("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",")]


# =====================================================
# READINESS STATE
# =====================================================

class ReadinessState:

    def __init__(self):
        self.models_loaded = False
        self.redis_connected = False
        self.drift_baseline_loaded = False

        self.schema_signature = None
        self.model_version = None
        self.artifact_hash = None
        self.dataset_hash = None
        self.training_code_hash = None

        self.llm_enabled = False
        self.llm_model = None
        self.llm_rate_limit = None
        self.llm_cache_enabled = None

        self.boot_id = BOOT_ID
        self.start_time = int(time.time())
        self.boot_memory_mb = None
        self.config_fingerprint = None

    @property
    def ready(self):
        return self.models_loaded


readiness = ReadinessState()


# =====================================================
# RESPONSE HELPERS
# =====================================================

def api_success(data):
    return {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def api_error(message):
    return {
        "success": False,
        "data": None,
        "error": message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =====================================================
# GLOBAL EXCEPTION HANDLER
# =====================================================

async def global_exception_handler(request: Request, exc: Exception):

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=api_error(exc.detail)
        )

    logger.exception("Unhandled API error")

    return JSONResponse(
        status_code=500,
        content=api_error("internal_server_error")
    )


# =====================================================
# LIFESPAN MANAGEMENT
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    boot_start = time.time()

    try:
        logger.info("===================================")
        logger.info(" MarketSentinel Boot Sequence ")
        logger.info(" Boot ID: %s", BOOT_ID)
        logger.info("===================================")

        # -------------------------------------------------
        # MODEL LOADING
        # -------------------------------------------------

        loader = ModelLoader()
        _ = loader.xgb
        loader.warmup()

        readiness.models_loaded = True
        readiness.schema_signature = loader.schema_signature
        readiness.model_version = loader.xgb_version
        readiness.artifact_hash = loader.artifact_hash
        readiness.dataset_hash = loader.dataset_hash
        readiness.training_code_hash = loader.training_code_hash

        if readiness.schema_signature != get_schema_signature():
            logger.warning("Runtime schema mismatch detected.")

        # -------------------------------------------------
        # REDIS (Soft)
        # -------------------------------------------------

        try:
            cache = RedisCache()
            readiness.redis_connected = cache.enabled
        except Exception:
            readiness.redis_connected = False
            logger.warning("Redis unavailable — continuing in degraded mode.")

        # -------------------------------------------------
        # DRIFT BASELINE (Soft)
        # -------------------------------------------------

        try:
            detector = DriftDetector()
            detector._load_verified_baseline()
            readiness.drift_baseline_loaded = True
        except Exception:
            readiness.drift_baseline_loaded = False
            logger.warning("Drift baseline not loaded (non-blocking).")

        # -------------------------------------------------
        # LLM STATE
        # -------------------------------------------------

        readiness.llm_enabled = get_bool("LLM_ENABLED", False)
        readiness.llm_model = get_env("OPENAI_MODEL", None)
        readiness.llm_rate_limit = get_int("LLM_RATE_LIMIT_PER_MIN", 30)
        readiness.llm_cache_enabled = get_bool("LLM_CACHE_ENABLED", True)

        # -------------------------------------------------
        # SYSTEM SNAPSHOT
        # -------------------------------------------------

        process = psutil.Process(os.getpid())
        readiness.boot_memory_mb = round(
            process.memory_info().rss / (1024 * 1024), 2
        )

        readiness.config_fingerprint = hashlib.sha256(
            str(sorted(os.environ.items())).encode()
        ).hexdigest()[:16]

        gc.collect()

        boot_time = round(time.time() - boot_start, 2)

        if boot_time > STARTUP_TIMEOUT_SEC:
            logger.warning("Startup exceeded expected timeout threshold.")

        logger.info("System ready in %.2fs", boot_time)

        yield

        logger.info("Shutting down MarketSentinel...")
        gc.collect()

    except Exception:
        logger.exception("CRITICAL STARTUP FAILURE")
        raise


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="MarketSentinel Portfolio API",
    version=APP_VERSION,
    lifespan=lifespan
)

app.add_exception_handler(Exception, global_exception_handler)


# =====================================================
# CORS MIDDLEWARE (React Frontend Support)
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# =====================================================
# REQUEST MIDDLEWARE
# =====================================================

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())[:12]
    start = time.time()

    try:
        response = await call_next(request)
        latency = time.time() - start

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(latency, 4))

        return response

    except Exception:
        logger.exception("Request failure | id=%s", request_id)
        raise


# =====================================================
# ROUTES
# =====================================================

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(universe.router)
app.include_router(model_info.router)
app.include_router(drift.router)
app.include_router(portfolio.router)
app.include_router(performance.router)
app.include_router(equity.router)
app.include_router(agent.router)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
async def root():
    return api_success({
        "service": "MarketSentinel Hybrid Portfolio Engine",
        "architecture": "multi-agent-ml-governed",
        "status": "running",
        "boot_id": readiness.boot_id,
        "model_version": readiness.model_version,
        "artifact_hash": readiness.artifact_hash,
        "version": APP_VERSION,
        "docs": "/docs",
        "metrics": "/metrics"
    })


# =====================================================
# METRICS
# =====================================================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# =====================================================
# READINESS
# =====================================================

@app.get("/ready")
def readiness_probe():

    if not readiness.ready:
        raise HTTPException(status_code=503, detail="service_not_ready")

    return api_success({
        "status": "ready",
        "boot_id": readiness.boot_id,
        "model_version": readiness.model_version,
        "artifact_hash": readiness.artifact_hash,
        "dataset_hash": readiness.dataset_hash,
        "schema_signature": readiness.schema_signature,
        "training_code_hash": readiness.training_code_hash,
        "redis_connected": readiness.redis_connected,
        "drift_baseline_loaded": readiness.drift_baseline_loaded,
        "memory_mb": readiness.boot_memory_mb,
        "config_fingerprint": readiness.config_fingerprint,
        "mode": "degraded" if not readiness.redis_connected else "normal",
        "llm": {
            "enabled": readiness.llm_enabled,
            "model": readiness.llm_model,
            "rate_limit_per_min": readiness.llm_rate_limit,
            "cache_enabled": readiness.llm_cache_enabled
        },
        "uptime_seconds": int(time.time()) - readiness.start_time
    })


# =====================================================
# LIVENESS
# =====================================================

@app.get("/live")
def liveness_probe():
    return api_success({
        "status": "alive",
        "boot_id": readiness.boot_id
    })