from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.concurrency import run_in_threadpool
import time
import asyncio
import os
import re
import logging
import math
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.inference.pipeline import InferencePipeline
from app.inference.model_loader import ModelLoader

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.api")


# =========================================================
# SINGLETONS
# =========================================================

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


# =========================================================
# PRODUCTION LIMITS
# =========================================================

MAX_CONCURRENT_INFERENCES = int(
    os.getenv("MAX_CONCURRENT_INFERENCES", "4")
)

REQUEST_TIMEOUT = int(
    os.getenv("INFERENCE_TIMEOUT_SEC", "25")
)

MAX_BATCH_SIZE = int(
    os.getenv("MAX_BATCH_SIZE", "30")
)

MIN_BATCH_SIZE = 4

DEFAULT_USE_UNIVERSE = os.getenv(
    "DEFAULT_USE_UNIVERSE",
    "true"
).lower() == "true"

UNIVERSE_CONFIG_PATH = Path(
    os.getenv(
        "PRODUCTION_UNIVERSE_PATH",
        "config/universe_production.json"
    )
)

inference_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_INFERENCES
)

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,12}$")


# =========================================================
# DEFAULT UNIVERSE LOADER
# =========================================================

def load_default_universe() -> List[str]:

    if not UNIVERSE_CONFIG_PATH.exists():
        raise RuntimeError("Universe config missing.")

    with open(UNIVERSE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError("Universe config invalid format.")

    return sorted(set(data))


# =========================================================
# REQUEST SCHEMA
# =========================================================

class PortfolioRequest(BaseModel):
    tickers: Optional[List[str]] = Field(default=None)

    @field_validator("tickers")
    @classmethod
    def validate_and_normalize(cls, tickers):

        if tickers is None:
            return None

        cleaned = []

        for t in tickers:
            t = t.upper().strip()

            if not TICKER_REGEX.match(t):
                raise ValueError(f"Invalid ticker: {t}")

            cleaned.append(t)

        unique = sorted(set(cleaned))

        if len(unique) < MIN_BATCH_SIZE:
            raise ValueError(
                f"At least {MIN_BATCH_SIZE} tickers required."
            )

        return unique


# =========================================================
# OUTPUT VALIDATION
# =========================================================

def validate_portfolio_output(result: Any):

    if not isinstance(result, list):
        raise RuntimeError("Invalid portfolio output format.")

    for row in result:

        if not isinstance(row, dict):
            raise RuntimeError("Invalid portfolio row structure.")

        required = {"date", "ticker", "score", "signal", "weight"}

        if not required.issubset(row.keys()):
            raise RuntimeError("Portfolio row missing fields.")

        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise RuntimeError(
                    f"Non-finite value in portfolio output: {k}"
                )

    return result


# =========================================================
# RESPONSE MODELS
# =========================================================

class PortfolioResponse(BaseModel):
    meta: Dict[str, Any]
    portfolio: List[Dict[str, Any]]


class SnapshotResponse(BaseModel):
    meta: Dict[str, Any]
    snapshot: Dict[str, Any]


# =========================================================
# PORTFOLIO ENDPOINT
# =========================================================

@router.post("/portfolio", response_model=PortfolioResponse)
async def build_portfolio(req: PortfolioRequest):

    endpoint = "/portfolio"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        if req.tickers is None:

            if not DEFAULT_USE_UNIVERSE:
                raise HTTPException(
                    status_code=400,
                    detail="Tickers required."
                )

            tickers = load_default_universe()

        else:
            tickers = req.tickers

        if len(tickers) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE})"
            )

        loader = get_loader()

        if loader._xgb_container is None:
            raise HTTPException(
                status_code=503,
                detail="Model container unavailable"
            )

        async with inference_semaphore:

            result = await asyncio.wait_for(
                run_in_threadpool(
                    get_pipeline().run_batch,
                    tickers
                ),
                timeout=REQUEST_TIMEOUT
            )

        result = validate_portfolio_output(result)

        meta = {
            "model_version": loader.xgb_version,
            "schema_signature": loader._xgb_container.schema_signature,
            "dataset_hash": loader.dataset_hash,
            "training_code_hash": loader.training_code_hash,
            "artifact_hash": loader.artifact_hash,
            "inference_batch_size": len(tickers),
            "timestamp": int(time.time())
        }

        return PortfolioResponse(meta=meta, portfolio=result)

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(
            status_code=504,
            detail="Portfolio inference timeout"
        )

    except HTTPException:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise

    except RuntimeError as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(
            status_code=500,
            detail="Portfolio inference failed"
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# 🔥 LIVE SNAPSHOT (ML SHOWCASE ENDPOINT)
# =========================================================

@router.get("/live-snapshot", response_model=SnapshotResponse)
async def live_snapshot():

    endpoint = "/live-snapshot"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        tickers = load_default_universe()

        loader = get_loader()

        async with inference_semaphore:

            snapshot = await asyncio.wait_for(
                run_in_threadpool(
                    get_pipeline().run_snapshot,
                    tickers
                ),
                timeout=REQUEST_TIMEOUT
            )

        meta = {
            "model_version": loader.xgb_version,
            "schema_signature": loader._xgb_container.schema_signature,
            "dataset_hash": loader.dataset_hash,
            "training_code_hash": loader.training_code_hash,
            "artifact_hash": loader.artifact_hash,
            "universe_size": len(tickers),
            "timestamp": int(time.time())
        }

        return SnapshotResponse(meta=meta, snapshot=snapshot)

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(
            status_code=504,
            detail="Snapshot inference timeout"
        )

    except Exception:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Live snapshot failure")
        raise HTTPException(
            status_code=500,
            detail="Live snapshot failed"
        )

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )