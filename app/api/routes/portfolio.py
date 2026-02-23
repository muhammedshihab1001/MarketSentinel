import logging
import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline
from core.market.universe import MarketUniverse
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.portfolio")

REQUEST_TIMEOUT = 30
MAX_CONCURRENT = 3

portfolio_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

_pipeline: InferencePipeline | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


@router.get("/portfolio-summary")
async def portfolio_summary():

    endpoint = "/portfolio-summary"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        async with portfolio_semaphore:

            snapshot = await asyncio.wait_for(
                run_in_threadpool(_portfolio_summary_sync),
                timeout=REQUEST_TIMEOUT
            )

        return snapshot

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Portfolio summary timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Portfolio summary failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _portfolio_summary_sync():

    pipeline = get_pipeline()
    universe = MarketUniverse.get_universe()

    snapshot = pipeline.run_snapshot(universe)

    if not isinstance(snapshot, dict) or "signals" not in snapshot:
        raise RuntimeError("Invalid snapshot structure.")

    results = snapshot["signals"]

    if not results:
        raise RuntimeError("No signals generated.")

    # ===============================
    # SIGNAL COUNTS
    # ===============================

    long_count = sum(1 for r in results if r.get("signal") == "LONG")
    short_count = sum(1 for r in results if r.get("signal") == "SHORT")
    neutral_count = sum(1 for r in results if r.get("signal") == "NEUTRAL")

    # ===============================
    # EXPOSURE
    # ===============================

    gross_exposure = sum(abs(r.get("weight", 0.0)) for r in results)
    net_exposure = sum(r.get("weight", 0.0) for r in results)

    # ===============================
    # AGENT METRICS
    # ===============================

    high_conviction_count = sum(
        1 for r in results
        if r.get("agent", {}).get("strength_score", 0.0) >= 75
    )

    elevated_risk_count = sum(
        1 for r in results
        if r.get("agent", {}).get("risk_level") == "elevated"
    )

    return {
        "snapshot_date": snapshot.get("snapshot_date"),
        "universe_size": snapshot.get("universe_size"),
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,
        "gross_exposure": round(gross_exposure, 6),
        "net_exposure": round(net_exposure, 6),
        "high_conviction_count": high_conviction_count,
        "elevated_risk_count": elevated_risk_count,
    }