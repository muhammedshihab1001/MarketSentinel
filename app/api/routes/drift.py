import logging
import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline, get_shared_model_loader
from core.market.universe import MarketUniverse
from core.schema.feature_schema import MODEL_FEATURES
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.drift")

REQUEST_TIMEOUT = 30
MAX_CONCURRENT = 2

drift_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

_pipeline: InferencePipeline | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


@router.get("/drift-status")
async def drift_status():

    endpoint = "/drift-status"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        async with drift_semaphore:

            result = await asyncio.wait_for(
                run_in_threadpool(_drift_status_sync),
                timeout=REQUEST_TIMEOUT
            )

        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Drift check timeout")

    except Exception:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Drift status check failed")
        raise HTTPException(status_code=500, detail="Drift status failure")

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _drift_status_sync():

    pipeline = get_pipeline()
    loader = get_shared_model_loader()

    tickers = MarketUniverse.get_universe()

    df = pipeline._build_cross_sectional_frame(tickers)
    latest_df = pipeline._select_latest_snapshot(df)

    feature_df = latest_df.loc[:, MODEL_FEATURES]

    drift_result = pipeline.drift_detector.detect(feature_df)

    return {
        "drift_detected": drift_result.get("drift_detected", False),
        "severity_score": drift_result.get("severity_score", 0),
        "drift_state": drift_result.get("drift_state", "unknown"),
        "model_version": loader.xgb_version,
    }