import logging
import time
import gc
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest

from app.api.routes import health, predict
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
# GLOBAL INFERENCE LIMITER
# Prevent CPU death spirals
# =====================================================

MAX_CONCURRENT_INFERENCE = 4
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


# =====================================================
# READINESS
# =====================================================

class ReadinessState:

    def __init__(self):
        self.models_loaded = False
        self.redis_connected = False

    @property
    def ready(self):
        return self.models_loaded


readiness = ReadinessState()


# =====================================================
# GLOBAL EXCEPTION HANDLER
# =====================================================

async def global_exception_handler(request: Request, exc: Exception):

    logger.exception("Unhandled API error")

    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error"}
    )


# =====================================================
# LIFESPAN
# =====================================================

STARTUP_TIMEOUT_SEC = 120


@asynccontextmanager
async def lifespan(app: FastAPI):

    start_time = time.time()

    try:

        logger.info("===================================")
        logger.info(" MarketSentinel Boot Sequence Start ")
        logger.info("===================================")

        loader = ModelLoader()
        loader.warmup()

        readiness.models_loaded = True

        cache = RedisCache()

        if cache.enabled:
            readiness.redis_connected = True
            logger.info("Redis connected.")
        else:
            logger.warning("Redis unavailable — degraded mode.")

        # GC AFTER everything is loaded
        gc.collect()

        boot_time = round(time.time() - start_time, 2)

        if boot_time > STARTUP_TIMEOUT_SEC:
            raise RuntimeError("Startup exceeded timeout.")

        logger.info(f"System ready in {boot_time}s")

        yield

    except Exception:

        logger.exception("CRITICAL STARTUP FAILURE")

        raise


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="MarketSentinel API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_exception_handler(Exception, global_exception_handler)


# =====================================================
# ROUTES
# =====================================================

app.include_router(health.router, prefix="/health")
app.include_router(predict.router, prefix="/v1")


# =====================================================
# ROOT
# =====================================================

@app.get("/")
async def root():
    return {
        "service": "MarketSentinel API",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics"
    }


# =====================================================
# PROMETHEUS
# =====================================================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


# =====================================================
# READINESS
# =====================================================

@app.get("/ready")
def readiness_probe():

    if not readiness.ready:
        return Response("Service not ready", status_code=503)

    return {
        "status": "ready",
        "redis_connected": readiness.redis_connected,
        "mode": "degraded" if not readiness.redis_connected else "normal"
    }


# =====================================================
# LIVENESS
# =====================================================

@app.get("/live")
def liveness_probe():
    return {"status": "alive"}
