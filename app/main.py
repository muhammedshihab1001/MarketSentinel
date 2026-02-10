from fastapi import FastAPI, Response
from prometheus_client import generate_latest
import logging

from app.api.routes import health, predict
from app.inference.model_loader import ModelLoader


logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarketSentinel API",
    description="Stock prediction & sentiment-based trading signals",
    version="1.0.0"
)

# ------------------------------------------------
# MODEL READINESS FLAG
# ------------------------------------------------

MODELS_READY = False


# ------------------------------------------------
# Startup Event — Institutional Preload
# ------------------------------------------------

@app.on_event("startup")
def preload_models():
    """
    Preload models BEFORE serving traffic.

    Benefits:
    ✅ copy-on-write memory sharing
    ✅ faster first request
    ✅ prevents cold inference spikes
    """

    global MODELS_READY

    try:
        loader = ModelLoader()

        # Force lazy properties to load
        _ = loader.xgb
        _ = loader.lstm
        _ = loader.scaler
        _ = loader.prophet

        MODELS_READY = True
        logger.info("Models successfully preloaded.")

    except Exception as e:
        logger.exception("Model preload failed!")
        raise e


# ---------------------------
# Routers
# ---------------------------

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

# ------------------------------------------------
# Root endpoint
# ------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "MarketSentinel API",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics"
    }


# ------------------------------------------------
# Prometheus metrics endpoint
# ------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# ------------------------------------------------
# Kubernetes / Container Readiness
# ------------------------------------------------

@app.get("/ready")
def readiness_probe():
    """
    Returns 200 ONLY when models are ready.

    Prevents traffic before inference is safe.
    """

    if not MODELS_READY:
        return Response(
            content="Models are still loading",
            status_code=503
        )

    return {"status": "ready"}


# ------------------------------------------------
# Liveness Probe
# ------------------------------------------------

@app.get("/live")
def liveness_probe():
    """
    Confirms the app is alive.

    Should NOT depend on models.
    """

    return {"status": "alive"}
