from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.concurrency import run_in_threadpool
import datetime
import time
import asyncio
import os
import re
import logging

from app.inference.pipeline import InferencePipeline
from app.inference.model_loader import ModelLoader
from core.schema.feature_schema import get_schema_signature
from core.signals.signal_engine import StrategyEngine

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.api")

# ------------------------------------------------
# LAZY SINGLETONS
# ------------------------------------------------

_pipeline = None
_strategy_engine = None
_model_loader = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


def get_strategy_engine():
    global _strategy_engine
    if _strategy_engine is None:
        _strategy_engine = StrategyEngine()
    return _strategy_engine


def get_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


# ------------------------------------------------
# CONCURRENCY GATE
# ------------------------------------------------

MAX_CONCURRENT_INFERENCES = int(
    os.getenv("MAX_CONCURRENT_INFERENCES", "4")
)

REQUEST_TIMEOUT = int(
    os.getenv("INFERENCE_TIMEOUT_SEC", "25")
)

inference_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_INFERENCES
)

MAX_BATCH_SIZE = int(
    os.getenv("MAX_BATCH_SIZE", "10")
)

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,10}$")


# ----------------------------------------
# REQUEST SCHEMAS
# ----------------------------------------

class PredictionRequest(BaseModel):
    ticker: str = Field(default="AAPL")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str):
        v = v.upper().strip()

        if not TICKER_REGEX.match(v):
            raise ValueError("Invalid ticker format")

        return v


class BatchPredictionRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=50)

    @field_validator("tickers")
    @classmethod
    def normalize(cls, tickers):

        cleaned = []

        for t in tickers:
            t = t.upper().strip()

            if not TICKER_REGEX.match(t):
                raise ValueError(f"Invalid ticker: {t}")

            cleaned.append(t)

        return list(set(cleaned))


# ----------------------------------------
# SINGLE INFERENCE
# ----------------------------------------

@router.post("/predict")
async def predict(req: PredictionRequest):

    endpoint = "/predict"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    try:

        async with inference_semaphore:

            result = await asyncio.wait_for(
                run_in_threadpool(
                    get_pipeline().run,
                    req.ticker
                ),
                timeout=REQUEST_TIMEOUT
            )

        # Attach model metadata
        loader = get_loader()

        return {
            "meta": {
                "model_version": loader.xgb_version if hasattr(loader, "xgb_version") else "latest",
                "schema_signature": get_schema_signature(),
                "timestamp": int(time.time())
            },
            "data": result
        }

    except asyncio.TimeoutError:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.error("Inference timeout")

        raise HTTPException(
            status_code=504,
            detail="Inference timeout"
        )

    except Exception:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Inference failure")

        raise HTTPException(
            status_code=500,
            detail="Inference failed"
        )

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# ----------------------------------------
# BATCH INFERENCE
# ----------------------------------------

@router.post("/predict/batch")
async def predict_batch(req: BatchPredictionRequest):

    endpoint = "/predict/batch"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    if len(req.tickers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE})"
        )

    results = []

    try:

        for ticker in req.tickers:

            async with inference_semaphore:

                try:

                    result = await asyncio.wait_for(
                        run_in_threadpool(
                            get_pipeline().run,
                            ticker
                        ),
                        timeout=REQUEST_TIMEOUT
                    )

                    results.append(result)

                except Exception:
                    logger.exception(f"Batch inference failed for {ticker}")
                    results.append({
                        "ticker": ticker,
                        "error": "inference_failed"
                    })

        return {
            "count": len(results),
            "results": results
        }

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# ----------------------------------------
# STRATEGY ENDPOINT
# ----------------------------------------

@router.post("/strategy/top")
async def top_opportunities(req: BatchPredictionRequest):

    endpoint = "/strategy/top"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    if len(req.tickers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE})"
        )

    predictions = []

    try:

        for ticker in req.tickers:

            async with inference_semaphore:

                try:

                    result = await asyncio.wait_for(
                        run_in_threadpool(
                            get_pipeline().run,
                            ticker
                        ),
                        timeout=REQUEST_TIMEOUT
                    )

                    predictions.append(result)

                except Exception:
                    logger.exception(f"Strategy inference failed for {ticker}")

        strategy_engine = get_strategy_engine()

        return {
            "top_buys": strategy_engine.top_opportunities(predictions),
            "sell_alerts": strategy_engine.sell_alerts(predictions),
            "signal_distribution": strategy_engine.signal_distribution(predictions),
            "analyzed": len(predictions)
        }

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# ----------------------------------------
# HEALTH ENDPOINT
# ----------------------------------------

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": int(time.time())
    }


# ----------------------------------------
# READINESS ENDPOINT
# ----------------------------------------

@router.get("/ready")
async def readiness():

    try:
        _ = get_loader().xgb
        return {
            "status": "ready",
            "schema_signature": get_schema_signature()
        }

    except Exception:
        raise HTTPException(status_code=503, detail="Model not ready")