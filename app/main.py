from fastapi import FastAPI, Response
from prometheus_client import generate_latest

from app.api.routes import health, predict

app = FastAPI(
    title="MarketSentinel API",
    description="Stock prediction & sentiment-based trading signals",
    version="1.0.0"
)

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

# ---------------------------
# Root endpoint (optional but good)
# ---------------------------
@app.get("/")
def root():
    return {
        "service": "MarketSentinel API",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics"
    }

# ---------------------------
# Prometheus metrics endpoint
# ---------------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
