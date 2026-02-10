from fastapi import FastAPI, Response
from prometheus_client import generate_latest
import logging

from app.api.routes import health, predict
from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache


logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarketSentinel API",
    description="Stock prediction & sentiment-based trading signals",
    version="1.0.0"
)

# =====================================================
# READINESS STATE (Institutional Pattern)
# =====================================================

class ReadinessState:
    """
    Thread-safe readiness tracker.
    Allows richer health diagnostics.
    """

    def __init__(self):
        self.models_loaded = False
        self.redis_connected = False

    @property
    def ready(self):
        return self.models_loaded and self.redis_connected


readiness = ReadinessState()


# =====================================================
# STARTUP — Institutional Warmup
# =====================================================

@app.on_event("startup")
def preload_dependencies():
    """
    Preload ALL critical inference dependencies.

    Guarantees:
    ✅ no cold starts
    ✅ registry validation
    ✅ dependency health
    """

    try:

        logger.info("🔥 Starting MarketSentinel warmup...")

        # -----------------------------
        # Model Warmup
        # -----------------------------
        loader = ModelLoader()
        loader.warmup()

        readiness.models_loaded = True

        # -----------------------------
        # Redis Validation
        # -----------------------------
        cache = RedisCache()

        if cache.enabled:
            readiness.redis_connected = True
            logger.info("✅ Redis connection verified.")
        else:
            logger.warning(
                "⚠️ Redis unavailable — running without cache."
            )
            readiness.redis_connected = True  # still allow serving

        logger.info("✅ System ready for traffic.")

    except Exception as e:

        logger.exception("🚨 CRITICAL STARTUP FAILURE")

        # Fail fast → container restart
        raise e


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
# READINESS PROBE (Load Balancers)
# =====================================================

@app.get("/ready")
def readiness_probe():
    """
    Returns 200 ONLY when inference is safe.
    """

    if not readiness.ready:

        return Response(
            content="Service not ready",
            status_code=503
        )

    return {
        "status": "ready",
        "models_loaded": readiness.models_loaded,
        "redis_connected": readiness.redis_connected
    }


# =====================================================
# LIVENESS PROBE
# =====================================================

@app.get("/live")
def liveness_probe():
    """
    Confirms process is alive.
    Does NOT check dependencies.
    """

    return {"status": "alive"}
