# =========================================================
# PREDICTION & SNAPSHOT ROUTES v3.4
# Hybrid Multi-Agent Compatible
# CV-Optimized | Decision-Aware | Governance-Aware
# =========================================================

import time
import asyncio
import os
import re
import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline, get_shared_model_loader
from core.data.market_data_service import MarketDataService

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

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = logging.getLogger("marketsentinel.api")

_pipeline: Optional[InferencePipeline] = None
_universe_cache: Optional[List[str]] = None


# =========================================================
# PIPELINE SINGLETON
# =========================================================

def get_pipeline() -> InferencePipeline:

    global _pipeline

    if _pipeline is None:

        logger.info("Initializing InferencePipeline (singleton)")
        _pipeline = InferencePipeline()

    return _pipeline


# =========================================================
# CONFIG
# =========================================================

MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))

REQUEST_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT_SEC", "180"))

MIN_BATCH_SIZE = 4

PRIMARY_UNIVERSE_PATH = Path(
    os.getenv("PRODUCTION_UNIVERSE_PATH", "config/universe.json")
)

FALLBACK_UNIVERSE_PATH = Path("config/universe.json")

inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,12}$")


# =========================================================
# UTILS
# =========================================================

def load_default_universe() -> List[str]:

    global _universe_cache

    if _universe_cache is not None:
        return _universe_cache

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

    _universe_cache = unique

    return unique


# =========================================================
# LIVE SNAPSHOT
# =========================================================

@router.get("/live-snapshot")
async def live_snapshot():

    endpoint = "/predict/live-snapshot"

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    try:

        tickers = load_default_universe()

        pipeline = get_pipeline()

        loader = get_shared_model_loader()
        container = loader._xgb_container

        async with inference_semaphore:

            snapshot = await asyncio.wait_for(
                run_in_threadpool(pipeline.run_snapshot, tickers),
                timeout=REQUEST_TIMEOUT
            )

        if not isinstance(snapshot, dict) or "signals" not in snapshot:

            raise RuntimeError("Invalid snapshot structure.")

        signals = snapshot["signals"]

        long_count = sum(1 for s in signals if s.get("weight", 0.0) > 0)
        short_count = sum(1 for s in signals if s.get("weight", 0.0) < 0)

        hybrid_scores = [
            s.get("hybrid_consensus_score", 0.0) * 100
            for s in signals
        ]

        avg_strength = (
            round(sum(hybrid_scores) / len(hybrid_scores), 2)
            if hybrid_scores else 0.0
        )

        top_5 = snapshot.get("top_5", [])

        decision_report = snapshot.get("decision_report", {})

        drift = snapshot.get("drift", {})

        executive_summary = {

            "top_5_tickers": [t["ticker"] for t in top_5],

            "portfolio_bias": decision_report.get("portfolio_bias"),

            "risk_regime": drift.get("drift_state"),

            "gross_exposure": snapshot.get("gross_exposure"),

            "net_exposure": snapshot.get("net_exposure")
        }

        meta = {

            "model_version": container.version if container else None,

            "schema_signature": container.schema_signature if container else None,

            "dataset_hash": container.dataset_hash if container else None,

            "artifact_hash": container.artifact_hash if container else None,

            "feature_checksum": container.feature_checksum if container else None,

            "universe_size": len(tickers),

            "long_signals": long_count,

            "short_signals": short_count,

            "avg_hybrid_score": avg_strength,

            "drift_state": drift.get("drift_state"),

            "latency_ms": int((time.time() - start_time) * 1000),

            "timestamp": int(time.time())
        }

        return {

            "meta": meta,

            "executive_summary": executive_summary,

            "snapshot": snapshot
        }

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

    endpoint = "/predict/signal-explanation"

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

        agents = row.get("agents", {})

        signal_agent = agents.get("signal_agent", {})

        explanation = SignalExplanationResponse(

            ticker=row["ticker"],

            score=row.get("raw_model_score", 0.0),

            signal=signal_agent.get("signal", "NEUTRAL"),

            agent_score=row.get(
                "hybrid_consensus_score",
                row.get("agent_score", 0.0)
            ),

            alpha_strength=signal_agent.get("alpha_strength", 0.0),

            confidence_numeric=signal_agent.get("confidence_numeric", 0.0),

            governance_score=signal_agent.get("governance_score", 0),

            risk_level=signal_agent.get("risk_level", "unknown"),

            volatility_regime=signal_agent.get("volatility_regime", "unknown"),

            drift_flag=signal_agent.get("drift_flag", False),

            warnings=signal_agent.get("warnings", []),

            explanation=signal_agent.get("explanation", "")
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


# =========================================================
# PRICE HISTORY
# =========================================================

@router.get("/price-history/{ticker}")
async def price_history(ticker: str, days: int = 365):

    endpoint = "/predict/price-history"

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    ticker = ticker.upper().strip()

    if not TICKER_REGEX.match(ticker):

        raise HTTPException(status_code=400, detail="Invalid ticker format.")

    if days > 2000:
        days = 2000

    try:

        service = MarketDataService()

        end_date = pd.Timestamp.utcnow().normalize()

        start_date = end_date - pd.Timedelta(days=days)

        data, failures = await run_in_threadpool(
            service.get_price_data_batch,
            [ticker],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        df = data.get(ticker)

        if df is None or df.empty:

            raise HTTPException(status_code=404, detail="Price history unavailable")

        prices = [

            {
                "date": str(row["date"]),
                "close": float(row["close"])
            }

            for _, row in df.iterrows()
        ]

        return {

            "ticker": ticker,

            "days": days,

            "prices": prices,

            "latency_ms": int((time.time() - start_time) * 1000),

            "timestamp": int(time.time())
        }

    except HTTPException:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        raise

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        logger.exception("Price history endpoint failed")

        raise HTTPException(status_code=500, detail=str(e))

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)