from fastapi import FastAPI
from app.api.routes import health, predict

app = FastAPI(
    title="MarketSentinel API",
    description="Stock prediction & sentiment-based trading signals",
    version="1.0.0"
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
