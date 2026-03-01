import time
import asyncio
import os
import re
import logging
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline, get_shared_model_loader
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

from app.api.schemas import (
    SignalExplanationEnvelope,
    SignalExplanationMeta,
    SignalExplanationResponse,
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.api")

_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing InferencePipeline (singleton)")
        _pipeline = InferencePipeline()
    return _pipeline


MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
REQUEST_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT_SEC", "25"))
MIN_BATCH_SIZE = 4

PRIMARY_UNIVERSE_PATH = Path(
    os.getenv("PRODUCTION_UNIVERSE_PATH", "config/universe.json")
)

FALLBACK_UNIVERSE_PATH = Path("config/universe.json")

inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,12}$")


def safe_attr(obj, attr, default=None):
    return getattr(obj, attr, default)


def load_default_universe() -> List[str]:

    universe_path = None

    if PRIMARY_UNIVERSE_PATH.exists():
        universe_path = PRIMARY_UNIVERSE_PATH
    elif FALLBACK_UNIVERSE_PATH.exists():
        universe_path = FALLBACK_UNIVERSE_PATH
    else:
        raise RuntimeError("No universe configuration file found.")

    with open(universe_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        tickers = data
    elif isinstance(data, dict) and "tickers" in data:
        tickers = data["tickers"]
    else:
        raise RuntimeError("Universe config invalid format.")

    cleaned = [
        str(t).upper().strip()
        for t in tickers
        if TICKER_REGEX.match(str(t).upper().strip())
    ]

    unique = sorted(set(cleaned))

    if len(unique) < MIN_BATCH_SIZE:
        raise RuntimeError("Universe size below minimum batch size.")

    return unique


# =========================================================
# LIVE SNAPSHOT
# =========================================================

@router.get("/live-snapshot")
async def live_snapshot():

    endpoint = "/live-snapshot"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        tickers = load_default_universe()
        pipeline = get_pipeline()
        loader = get_shared_model_loader()

        async with inference_semaphore:
            snapshot = await asyncio.wait_for(
                run_in_threadpool(pipeline.run_snapshot, tickers),
                timeout=REQUEST_TIMEOUT
            )

        if not snapshot or "signals" not in snapshot:
            raise RuntimeError("Invalid snapshot structure.")

        signals = snapshot["signals"]

        long_count = sum(1 for s in signals if s.get("weight", 0.0) > 0)
        short_count = sum(1 for s in signals if s.get("weight", 0.0) < 0)

        strength_scores = [
            s.get("agent", {}).get("agent_score", 0.0) * 100
            for s in signals
        ]

        avg_strength = (
            round(sum(strength_scores) / len(strength_scores), 2)
            if strength_scores else 0.0
        )

        meta = {
            "model_version": safe_attr(loader, "xgb_version"),
            "schema_signature": safe_attr(loader, "schema_signature"),
            "dataset_hash": safe_attr(loader, "dataset_hash"),
            "artifact_hash": safe_attr(loader, "artifact_hash"),
            "feature_checksum": safe_attr(loader, "feature_checksum"),
            "universe_size": len(tickers),
            "long_signals": long_count,
            "short_signals": short_count,
            "avg_agent_score": avg_strength,
            "gross_exposure": snapshot.get("gross_exposure"),
            "net_exposure": snapshot.get("net_exposure"),
            "drift_state": snapshot.get("drift", {}).get("drift_state"),
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": int(time.time())
        }

        return {"meta": meta, "snapshot": snapshot}

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Snapshot inference timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Live snapshot failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# SIGNAL EXPLANATION
# =========================================================

@router.get(
    "/signal-explanation/{ticker}",
    response_model=SignalExplanationEnvelope
)
async def signal_explanation(ticker: str):

    endpoint = "/signal-explanation"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    ticker = ticker.upper().strip()

    if not TICKER_REGEX.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format.")

    try:
        pipeline = get_pipeline()
        loader = get_shared_model_loader()
        universe_tickers = load_default_universe()

        if ticker not in universe_tickers:
            raise HTTPException(
                status_code=404,
                detail="Ticker not in production universe."
            )

        async with inference_semaphore:
            snapshot = await asyncio.wait_for(
                run_in_threadpool(pipeline.run_snapshot, universe_tickers),
                timeout=REQUEST_TIMEOUT
            )

        signals = snapshot.get("signals", [])

        row = next((s for s in signals if s["ticker"] == ticker), None)

        if row is None:
            raise HTTPException(status_code=404, detail="Signal not found.")

        agent_data = row.get("agent", {})

        direction = row.get("signal", "NEUTRAL")

        explanation = SignalExplanationResponse(
            ticker=row["ticker"],
            score=row.get("raw_model_score", 0.0),
            signal=direction,
            agent_score=agent_data.get("agent_score", 0.0),
            alpha_strength=agent_data.get("alpha_strength", 0.0),
            confidence_numeric=agent_data.get("confidence_numeric", 0.0),
            governance_score=agent_data.get("governance_score", 0),
            risk_level=agent_data.get("risk_level", "unknown"),
            volatility_regime=agent_data.get("volatility_regime", "unknown"),
            drift_flag=agent_data.get("drift_flag", False),
            warnings=agent_data.get("warnings", []),
            explanation=agent_data.get("explanation", "")
        )

        meta = SignalExplanationMeta(
            model_version=safe_attr(loader, "xgb_version"),
            schema_signature=safe_attr(loader, "schema_signature"),
            dataset_hash=safe_attr(loader, "dataset_hash"),
            artifact_hash=safe_attr(loader, "artifact_hash"),
            latency_ms=int((time.time() - start_time) * 1000),
            timestamp=int(time.time())
        )

        return SignalExplanationEnvelope(
            meta=meta,
            explanation=explanation
        )

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Signal explanation timeout")

    except HTTPException:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Signal explanation failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)