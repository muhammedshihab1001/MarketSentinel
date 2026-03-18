# =========================================================
# PORTFOLIO SUMMARY ROUTE v2.3
# FIX: reads from Redis snapshot cache — was 37s first call, now <100ms
# Falls back to full inference only if cache is empty
# =========================================================

import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.cache import RedisCache
from app.inference.pipeline import get_shared_model_loader
from core.market.universe import MarketUniverse
from core.logging.logger import get_logger

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

router = APIRouter()
logger = get_logger("marketsentinel.portfolio")

REQUEST_TIMEOUT = 180
MAX_CONCURRENT = 3

portfolio_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# GET SNAPSHOT — cache-first, inference fallback
# =========================================================

def _get_snapshot():
    """
    Fast path: read from background pre-computed snapshot in Redis.
    Slow path: run full inference if cache is cold (first boot only).
    """

    cache = RedisCache()

    if cache.enabled:
        # Try background snapshot key first (set by main.py every 120s)
        bg_key = cache.build_key({"type": "background_snapshot"})
        snapshot = cache.get(bg_key)

        if snapshot and isinstance(snapshot, dict) and "signals" in snapshot:
            logger.info("Portfolio served from background snapshot cache")
            return snapshot, True

    # Cache miss — run full inference
    logger.info("Portfolio cache miss — running full inference")
    from app.api.routes.predict import get_pipeline
    universe = MarketUniverse.get_universe()
    pipeline = get_pipeline()
    snapshot = pipeline.run_snapshot(universe)
    return snapshot, False


# =========================================================
# SHARED SYNC LOGIC
# =========================================================

def _portfolio_summary_sync():

    start_time = time.time()

    loader = get_shared_model_loader()
    universe = MarketUniverse.get_universe()

    snapshot, from_cache = _get_snapshot()

    if not isinstance(snapshot, dict) or "signals" not in snapshot:
        raise RuntimeError("Invalid snapshot structure.")

    results = snapshot["signals"]

    if not results:
        raise RuntimeError("No signals generated.")

    long_count = sum(1 for r in results if r.get("weight", 0.0) > 0)
    short_count = sum(1 for r in results if r.get("weight", 0.0) < 0)
    neutral_count = sum(1 for r in results if r.get("weight", 0.0) == 0)

    gross_exposure = sum(abs(r.get("weight", 0.0)) for r in results)
    net_exposure = sum(r.get("weight", 0.0) for r in results)

    def _get_signal_agent(r):
        return r.get("agents", {}).get("signal_agent", {})

    approved_trades = sum(
        1 for r in results
        if _get_signal_agent(r).get("trade_approved", False)
    )
    rejected_trades = len(results) - approved_trades

    agent_scores = [
        _get_signal_agent(r).get("agent_score", 0.0) or 0.0
        for r in results
    ]
    confidence_scores = [
        _get_signal_agent(r).get("confidence_numeric", 0.0) or 0.0
        for r in results
    ]

    avg_strength = sum(agent_scores) / len(agent_scores) if agent_scores else 0.0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    high_conviction_count = sum(
        1 for r in results
        if (_get_signal_agent(r).get("agent_score") or 0.0) >= 0.75
    )
    elevated_risk_count = sum(
        1 for r in results
        if _get_signal_agent(r).get("risk_level") == "elevated"
    )

    drift_info = snapshot.get("drift", {})
    drift_detected = drift_info.get("drift_detected", False)
    drift_state = drift_info.get("drift_state", "unknown")

    health_score = 100
    if drift_detected:
        health_score -= 15
    if elevated_risk_count > len(results) * 0.3:
        health_score -= 10
    if avg_confidence < 0.4:
        health_score -= 10
    health_score = max(0, min(100, health_score))

    positions = [
        {
            "ticker": r["ticker"],
            "weight": round(r.get("weight", 0.0), 4),
            "signal": _get_signal_agent(r).get("signal", "NEUTRAL"),
        }
        for r in results
        if r.get("weight", 0.0) != 0.0
    ]

    top_5_preview = [
        {
            "ticker": r["ticker"],
            "score": round(r.get("raw_model_score", 0.0), 4),
            "weight": round(r.get("weight", 0.0), 4),
            "confidence": _get_signal_agent(r).get("confidence_numeric"),
            "approved": _get_signal_agent(r).get("trade_approved"),
        }
        for r in sorted(
            results,
            key=lambda x: x.get("raw_model_score", 0.0),
            reverse=True,
        )[:5]
    ]

    return {
        "snapshot_date": snapshot.get("snapshot_date"),
        "universe_size": len(universe),
        "model_version": loader.xgb_version,
        "schema_signature": loader.schema_signature,
        "gross_exposure": round(gross_exposure, 6),
        "net_exposure": round(net_exposure, 6),
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,
        "approved_trades": approved_trades,
        "rejected_trades": rejected_trades,
        "avg_strength_score": round(avg_strength, 3),
        "avg_confidence": round(avg_confidence, 3),
        "high_conviction_count": high_conviction_count,
        "elevated_risk_count": elevated_risk_count,
        "drift_detected": drift_detected,
        "drift_state": drift_state,
        "portfolio_health_score": health_score,
        "positions": positions,
        "top_5_preview": top_5_preview,
        "served_from_cache": from_cache,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# GET /portfolio  (frontend calls this)
# =========================================================

@router.get("/portfolio")
async def portfolio():

    endpoint = "/portfolio"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with portfolio_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_portfolio_summary_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Portfolio timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Portfolio failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /portfolio-summary  (backward compatibility)
# =========================================================

@router.get("/portfolio-summary")
async def portfolio_summary():

    endpoint = "/portfolio-summary"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with portfolio_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_portfolio_summary_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Portfolio summary timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Portfolio summary failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)