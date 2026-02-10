import logging
import time
import gc
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import generate_latest

from app.api.routes import health, predict
from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache


# =====================================================
# LOGGING (Institutional Baseline)
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("marketsentinel")


# =====================================================
# READINESS STATE
# =====================================================

class ReadinessState:
    """
    Thread-safe readiness tracker.
    """

    def __init__(self):
        self.models_loaded = False
        self.redis_connected = False

    @property
    def ready(self):
        # Redis is now treated as a degradation signal
        return self.models_loaded


readiness = ReadinessState()


# =====================================================
# LIFESPAN (Replaces deprecated startup event)
# =====================================================

STARTUP_TIMEOUT_SEC = 120


@asynccontextmanager
async def lifespan(app: FastAPI):

    start_time = time.time()

    try:

        logger.info("Starting MarketSentinel boot sequence...")

        # -----------------------------
        # Model Warmup
        # -----------------------------

        loader = ModelLoader()
        loader.warmup()

        readiness.models_loaded = True

        # Memory stabilization after large model loads
        gc.collect()

        # -----------------------------
        # Redis Validation
        # -----------------------------

        cache = RedisCache()

        if cache.enabled:
            readiness.redis_connected = True
            logger.info("Redis connection verified.")

        else:
            readiness.redis_connected = False
            logger.warning(
                "Redis unavailable — running in DEGRADED mode (no cache)."
            )

        boot_time = round(time.time() - start_time, 2)

        if boot_time > STARTUP_TIMEOUT_SEC:
            raise RuntimeError("Startup exceeded safety timeout.")

        logger.info(f"System ready. Boot time: {boot_time}s")

        yield

    except Exception:

        logger.exception("CRITICAL STARTUP FAILURE — refusing traffic.")

        # Fail fast → container restart
        raise


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="MarketSentinel API",
    description="Stock prediction & sentiment-based trading signals",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================
# ROUTERS
# =====================================================

app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    predict.router,
    prefix="/v1",
    tags=["Prediction"]
)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():
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
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# =====================================================
# READINESS PROBE
# =====================================================

@app.get("/ready")
def readiness_probe():

    if not readiness.ready:

        return Response(
            content="Service not ready",
            status_code=503
        )

    status = {
        "status": "ready",
        "models_loaded": readiness.models_loaded,
        "redis_connected": readiness.redis_connected
    }

    if not readiness.redis_connected:
        status["mode"] = "degraded"

    return status


# =====================================================
# LIVENESS PROBE
# =====================================================

@app.get("/live")
def liveness_probe():
    return {"status": "alive"}
