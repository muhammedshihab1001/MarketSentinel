import logging
import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline, get_shared_model_loader
from core.market.universe import MarketUniverse
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    DTYPE,
)
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.drift")

REQUEST_TIMEOUT = 30
MAX_CONCURRENT = 2
MIN_UNIVERSE_WIDTH = 10

drift_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

_pipeline: InferencePipeline | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


# =========================================================
# DRIFT STATUS
# =========================================================

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

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Drift status check failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# SYNC DRIFT LOGIC
# =========================================================

def _drift_status_sync():

    start_time = time.time()

    pipeline = get_pipeline()
    loader = get_shared_model_loader()

    tickers = MarketUniverse.get_universe()

    if not tickers or len(tickers) < MIN_UNIVERSE_WIDTH:
        raise RuntimeError("Universe too small for drift detection.")

    # Build full cross-sectional frame
    df = pipeline._build_cross_sectional_frame(tickers)
    latest_df = pipeline._select_latest_snapshot(df)

    if latest_df.empty:
        raise RuntimeError("No latest snapshot available for drift.")

    # Validate schema before drift
    feature_df = validate_feature_schema(
        latest_df.loc[:, MODEL_FEATURES],
        mode="inference"
    ).astype(DTYPE)

    drift_result = pipeline.drift_detector.detect(feature_df)

    return {
        "drift_detected": drift_result.get("drift_detected", False),
        "severity_score": drift_result.get("severity_score", 0),
        "drift_state": drift_result.get("drift_state", "unknown"),
        "exposure_scale": drift_result.get("exposure_scale", 1.0),

        # Governance
        "model_version": loader.xgb_version,
        "schema_signature": loader.schema_signature,
        "artifact_hash": loader.artifact_hash,
        "dataset_hash": loader.dataset_hash,

        # Context
        "universe_size": len(tickers),
        "snapshot_date": str(latest_df["date"].iloc[0]),
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time())
    }