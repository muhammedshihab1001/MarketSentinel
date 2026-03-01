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


# =========================================================
# SYNC EXECUTION
# =========================================================

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

    long_count = sum(1 for r in results if r.get("weight", 0.0) > 0)
    short_count = sum(1 for r in results if r.get("weight", 0.0) < 0)
    neutral_count = sum(1 for r in results if r.get("weight", 0.0) == 0)

    # ===============================
    # EXPOSURE
    # ===============================

    gross_exposure = sum(abs(r.get("weight", 0.0)) for r in results)
    net_exposure = sum(r.get("weight", 0.0) for r in results)

    # ===============================
    # AGENT METRICS
    # ===============================

    approved_trades = sum(
        1 for r in results
        if r.get("agent", {}).get("trade_approved", False)
    )

    rejected_trades = len(results) - approved_trades

    strength_scores = [
        r.get("agent", {}).get("strength_score", 0.0)
        for r in results
    ]

    avg_strength = float(sum(strength_scores) / len(strength_scores))

    confidence_scores = [
        r.get("agent", {}).get("confidence_numeric", 0.0)
        for r in results
    ]

    avg_confidence = float(sum(confidence_scores) / len(confidence_scores))

    high_conviction_count = sum(
        1 for r in results
        if r.get("agent", {}).get("strength_score", 0.0) >= 75
    )

    elevated_risk_count = sum(
        1 for r in results
        if r.get("agent", {}).get("risk_level") == "elevated"
    )

    drift_detected = snapshot.get("drift", {}).get("drift_detected", False)
    drift_state = snapshot.get("drift", {}).get("drift_state", "unknown")

    # ===============================
    # PORTFOLIO HEALTH SCORE
    # ===============================

    health_score = 100

    if drift_detected:
        health_score -= 15

    if elevated_risk_count > len(results) * 0.3:
        health_score -= 10

    if avg_confidence < 0.4:
        health_score -= 10

    health_score = max(0, min(100, health_score))

    # ===============================
    # TOP 5 SUMMARY (FOR DASHBOARD)
    # ===============================

    top_5_preview = [
        {
            "ticker": r["ticker"],
            "score": round(r["score"], 4),
            "weight": round(r["weight"], 4),
            "confidence": r.get("agent", {}).get("confidence"),
            "approved": r.get("agent", {}).get("trade_approved")
        }
        for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    ]

    return {
        "snapshot_date": snapshot.get("snapshot_date"),
        "universe_size": snapshot.get("universe_size"),

        # Exposure
        "gross_exposure": round(gross_exposure, 6),
        "net_exposure": round(net_exposure, 6),

        # Signal breakdown
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,

        # Agent metrics
        "approved_trades": approved_trades,
        "rejected_trades": rejected_trades,
        "avg_strength_score": round(avg_strength, 2),
        "avg_confidence": round(avg_confidence, 3),
        "high_conviction_count": high_conviction_count,
        "elevated_risk_count": elevated_risk_count,

        # Drift
        "drift_detected": drift_detected,
        "drift_state": drift_state,

        # Portfolio health
        "portfolio_health_score": health_score,

        # Preview
        "top_5_preview": top_5_preview,
    }