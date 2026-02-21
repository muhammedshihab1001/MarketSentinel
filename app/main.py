import logging
import time
import gc
import hashlib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest

from app.api.routes import predict
from app.api.routes import performance
from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("marketsentinel")


# =====================================================
# SYSTEM FINGERPRINT
# =====================================================

BOOT_ID = hashlib.sha256(
    str(time.time()).encode()
).hexdigest()[:12]

STARTUP_TIMEOUT_SEC = int(
    os.getenv("STARTUP_TIMEOUT_SEC", "120")
)

APP_VERSION = os.getenv("APP_VERSION", "3.1.0")


# =====================================================
# GLOBAL STATE
# =====================================================

class ReadinessState:

    def __init__(self):
        self.models_loaded = False
        self.redis_connected = False
        self.schema_signature = None
        self.model_version = None
        self.artifact_hash = None
        self.dataset_hash = None
        self.training_code_hash = None
        self.boot_id = BOOT_ID
        self.start_time = int(time.time())

    @property
    def ready(self):
        return self.models_loaded


readiness = ReadinessState()


# =====================================================
# GLOBAL EXCEPTION HANDLER
# =====================================================

async def global_exception_handler(request: Request, exc: Exception):

    if isinstance(exc, HTTPException):
        raise exc

    logger.exception("Unhandled API error")

    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error"}
    )


# =====================================================
# LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    start_time = time.time()

    try:

        logger.info("===================================")
        logger.info(" MarketSentinel Boot Sequence Start ")
        logger.info(" Boot ID: %s", BOOT_ID)
        logger.info("===================================")

        # -------------------------------------------------
        # MODEL LOADING
        # -------------------------------------------------

        loader = ModelLoader()

        model = loader.xgb  # Force strict load

        readiness.models_loaded = True
        readiness.schema_signature = loader._xgb_container.schema_signature
        readiness.model_version = loader.xgb_version
        readiness.artifact_hash = loader.artifact_hash
        readiness.dataset_hash = loader.dataset_hash
        readiness.training_code_hash = loader.training_code_hash

        if not readiness.schema_signature:
            raise RuntimeError("Schema signature missing at startup.")

        # -------------------------------------------------
        # REDIS CHECK
        # -------------------------------------------------

        cache = RedisCache()

        if cache.enabled:
            readiness.redis_connected = True
            logger.info("Redis connected.")
        else:
            logger.warning("Redis unavailable — degraded mode.")

        gc.collect()

        boot_time = round(time.time() - start_time, 2)

        if boot_time > STARTUP_TIMEOUT_SEC:
            raise RuntimeError("Startup exceeded timeout.")

        logger.info("System ready in %.2fs", boot_time)
        logger.info("Schema signature: %s", readiness.schema_signature)
        logger.info("Model version: %s", readiness.model_version)

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
# ROUTES
# =====================================================

app.include_router(predict.router, prefix="/v1")
app.include_router(performance.router,prefix="/v1")


# =====================================================
# ROOT
# =====================================================

@app.get("/")
async def root():
    return {
        "service": "MarketSentinel Portfolio Engine",
        "status": "running",
        "boot_id": readiness.boot_id,
        "model_version": readiness.model_version,
        "artifact_hash": readiness.artifact_hash,
        "docs": "/docs",
        "metrics": "/metrics"
    }


# =====================================================
# PROMETHEUS
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
        return Response("Service not ready", status_code=503)

    return {
        "status": "ready",
        "boot_id": readiness.boot_id,
        "model_version": readiness.model_version,
        "artifact_hash": readiness.artifact_hash,
        "dataset_hash": readiness.dataset_hash,
        "schema_signature": readiness.schema_signature,
        "training_code_hash": readiness.training_code_hash,
        "redis_connected": readiness.redis_connected,
        "mode": "degraded" if not readiness.redis_connected else "normal",
        "uptime_seconds": int(time.time()) - readiness.start_time
    }


# =====================================================
# LIVENESS
# =====================================================

@app.get("/live")
def liveness_probe():
    return {
        "status": "alive",
        "boot_id": readiness.boot_id
    }