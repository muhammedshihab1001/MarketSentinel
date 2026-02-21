from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.concurrency import run_in_threadpool
import time
import asyncio
import os
import re
import logging
import math

from app.inference.pipeline import InferencePipeline
from app.inference.model_loader import ModelLoader

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.api")

# ------------------------------------------------
# SINGLETONS
# ------------------------------------------------

_pipeline = None
_model_loader = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


def get_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.warmup()
    return _model_loader


# ------------------------------------------------
# CONCURRENCY CONTROL
# ------------------------------------------------

MAX_CONCURRENT_INFERENCES = int(
    os.getenv("MAX_CONCURRENT_INFERENCES", "4")
)

REQUEST_TIMEOUT = int(
    os.getenv("INFERENCE_TIMEOUT_SEC", "30")
)

MAX_BATCH_SIZE = int(
    os.getenv("MAX_BATCH_SIZE", "50")
)

inference_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_INFERENCES
)

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,12}$")


# ------------------------------------------------
# REQUEST SCHEMA
# ------------------------------------------------

class PortfolioRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=5, max_length=200)

    @field_validator("tickers")
    @classmethod
    def validate_and_normalize(cls, tickers):

        cleaned = []

        for t in tickers:
            t = t.upper().strip()

            if not TICKER_REGEX.match(t):
                raise ValueError(f"Invalid ticker: {t}")

            cleaned.append(t)

        return sorted(set(cleaned))


# ------------------------------------------------
# STRICT PORTFOLIO VALIDATION
# ------------------------------------------------

def validate_portfolio_output(result):

    if not isinstance(result, list):
        raise RuntimeError("Invalid portfolio output format.")

    for row in result:

        if not isinstance(row, dict):
            raise RuntimeError("Invalid portfolio row structure.")

        for k, v in row.items():

            if isinstance(v, float):
                if not math.isfinite(v):
                    raise RuntimeError(
                        f"Non-finite value in portfolio output: {k}"
                    )

    return result


# ------------------------------------------------
# PORTFOLIO ENDPOINT
# ------------------------------------------------

@router.post("/portfolio")
async def build_portfolio(req: PortfolioRequest):

    endpoint = "/portfolio"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    if len(req.tickers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE})"
        )

    try:

        async with inference_semaphore:

            result = await asyncio.wait_for(
                run_in_threadpool(
                    get_pipeline().run_batch,
                    req.tickers
                ),
                timeout=REQUEST_TIMEOUT
            )

        result = validate_portfolio_output(result)

        loader = get_loader()

        container = loader._xgb_container

        return {
            "meta": {
                "model_version": container.version,
                "schema_signature": container.schema_signature,
                "dataset_hash": container.dataset_hash,
                "training_code_hash": container.training_code_hash,
                "artifact_hash": container.artifact_hash,
                "timestamp": int(time.time())
            },
            "portfolio": result
        }

    except asyncio.TimeoutError:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.error("Portfolio inference timeout")

        raise HTTPException(
            status_code=504,
            detail="Portfolio inference timeout"
        )

    except Exception:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Portfolio inference failure")

        raise HTTPException(
            status_code=500,
            detail="Portfolio inference failed"
        )

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# ------------------------------------------------
# HEALTH
# ------------------------------------------------

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": int(time.time())
    }


# ------------------------------------------------
# READINESS
# ------------------------------------------------

@router.get("/ready")
async def readiness():

    try:
        loader = get_loader()
        container = loader._xgb_container or loader.xgb

        return {
            "status": "ready",
            "model_version": container.version,
            "schema_signature": container.schema_signature
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        ) 