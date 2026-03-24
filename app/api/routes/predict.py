# =========================================================
# PREDICTION & SNAPSHOT ROUTES v4.0
#
# FIX (issue 1): get_pipeline() now initializes InferencePipeline
#   properly. Added init_pipeline() called from main.py startup
#   so model_loader and cache are available before first request.
# FIX (issue 3): pipeline.run_snapshot() called WITHOUT tickers arg.
#   Tickers are loaded inside pipeline from MarketUniverse.
#   Old: pipeline.run_snapshot(tickers) → tickers went to snapshot_date param → WRONG
#   New: pipeline.run_snapshot()        → uses MarketUniverse internally → CORRECT
# FIX: Background snapshot uses set_background_snapshot() not set()
# =========================================================

import time
import asyncio
import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline
from app.inference.model_loader import get_model_loader
from app.inference.cache import RedisCache
from core.data.market_data_service import MarketDataService
from core.logging.logger import get_logger

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

from app.api.schemas import (
    SignalExplanationEnvelope,
    SignalExplanationMeta,
    SignalExplanationResponse,
)

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = get_logger("marketsentinel.api")

# =========================================================
# MODULE-LEVEL SINGLETONS
# Set by init_pipeline() during app startup.
# get_pipeline() uses these to construct InferencePipeline.
# =========================================================

_pipeline: Optional[InferencePipeline] = None
_model_loader = None     # ModelLoader instance — used by pipeline.py fallback
_cache_instance = None   # RedisCache instance

_universe_cache: Optional[List[str]] = None

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"


# =========================================================
# PIPELINE INIT — called from main.py during startup
# =========================================================

def init_pipeline(model_loader, cache):
    """
    Initialize the InferencePipeline singleton with model and cache.
    Called from main.py lifespan after model and cache are loaded.

    Args:
        model_loader: ModelLoader instance (has .predict(), .version, etc.)
        cache:        RedisCache instance
    """
    global _pipeline, _model_loader, _cache_instance
    _model_loader = model_loader
    _cache_instance = cache
    _pipeline = InferencePipeline(model=model_loader, cache=cache)
    logger.info(
        "InferencePipeline initialized | model=%s",
        getattr(model_loader, "version", "unknown"),
    )


# =========================================================
# PIPELINE SINGLETON ACCESSOR
# =========================================================

def get_pipeline() -> InferencePipeline:
    """
    Return the InferencePipeline singleton.
    Raises RuntimeError if init_pipeline() was not called.
    """
    global _pipeline
    if _pipeline is None:
        # Fallback: try to init lazily from get_model_loader()
        try:
            loader = get_model_loader()
            cache = RedisCache()
            init_pipeline(loader, cache)
            logger.warning(
                "InferencePipeline lazily initialized (init_pipeline not called at startup)"
            )
        except Exception as e:
            raise RuntimeError(
                f"InferencePipeline not initialized. Call init_pipeline() during startup. Error: {e}"
            )
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


def _date_window(days: int):
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days + 30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# =========================================================
# LIVE SNAPSHOT
# =========================================================

@router.get("/live-snapshot")
@router.post("/live-snapshot")
async def live_snapshot():

    endpoint = "/predict/live-snapshot"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        pipeline = get_pipeline()
        loader = get_model_loader()
        meta_info = loader.metadata or {}
        cache = RedisCache()

        # Try background cache first
        snapshot = None
        cached = cache.get(BACKGROUND_SNAPSHOT_KEY)
        if cached and isinstance(cached, dict) and "snapshot" in cached:
            snapshot = cached
            logger.info("live-snapshot served from background cache")

        if snapshot is None:
            # FIX: run_snapshot() with NO args — tickers loaded internally
            async with inference_semaphore:
                snapshot = await asyncio.wait_for(
                    run_in_threadpool(pipeline.run_snapshot),
                    timeout=REQUEST_TIMEOUT,
                )

        if not isinstance(snapshot, dict):
            raise RuntimeError("Invalid snapshot structure.")

        return {
            "meta": snapshot.get("meta", {}),
            "executive_summary": snapshot.get("executive_summary", {}),
            "snapshot": snapshot.get("snapshot", {}),
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
    response_model=SignalExplanationEnvelope,
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
        loader = get_model_loader()
        universe_tickers = load_default_universe()

        if ticker not in universe_tickers:
            raise HTTPException(
                status_code=404,
                detail="Ticker not in production universe.",
            )

        # FIX: run_snapshot() with NO args
        async with inference_semaphore:
            snapshot = await asyncio.wait_for(
                run_in_threadpool(pipeline.run_snapshot),
                timeout=REQUEST_TIMEOUT,
            )

        signals = snapshot.get("snapshot", {}).get("signals", [])
        row = next((s for s in signals if s["ticker"] == ticker), None)

        if row is None:
            raise HTTPException(status_code=404, detail="Signal not found.")

        signal_details = snapshot.get("_signal_details", {})
        agents = signal_details.get(ticker, {})
        signal_agent = agents.get("signal_agent", {})

        weight = row.get("weight", 0.0)
        derived_signal = "LONG" if weight > 0 else ("SHORT" if weight < 0 else "NEUTRAL")

        explanation = SignalExplanationResponse(
            ticker=row["ticker"],
            score=row.get("raw_model_score", 0.0),
            signal=signal_agent.get("signal", derived_signal),
            agent_score=row.get("hybrid_consensus_score", 0.0),
            alpha_strength=signal_agent.get("alpha_strength", 0.0),
            confidence_numeric=signal_agent.get("confidence_numeric", 0.0),
            governance_score=signal_agent.get("governance_score", 0),
            risk_level=signal_agent.get("risk_level", "unknown"),
            volatility_regime=signal_agent.get("volatility_regime", "unknown"),
            drift_flag=signal_agent.get("drift_flag", False),
            warnings=signal_agent.get("warnings", []),
            explanation=signal_agent.get("explanation", ""),
        )

        meta_info = loader.metadata or {}
        meta = SignalExplanationMeta(
            model_version=loader.version,
            schema_signature=loader.schema_signature,
            dataset_hash=meta_info.get("dataset_hash"),
            artifact_hash=loader.artifact_hash,
            latency_ms=int((time.time() - start_time) * 1000),
            timestamp=int(time.time()),
        )

        return SignalExplanationEnvelope(meta=meta, explanation=explanation)

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

    days = min(days, 2000)

    try:
        service = MarketDataService()
        start_date, end_date = _date_window(days)

        df = await run_in_threadpool(
            service.get_price_data,
            ticker,
            start_date,
            end_date,
            "1d",
            days,
        )

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Price history unavailable")

        df = df.tail(days)

        prices = []
        for _, row in df.iterrows():
            date_val = str(pd.Timestamp(row["date"]).date())
            prices.append({
                "date": date_val,
                "open": round(float(row.get("open", 0)), 4),
                "high": round(float(row.get("high", 0)), 4),
                "low": round(float(row.get("low", 0)), 4),
                "close": round(float(row.get("close", 0)), 4),
                "volume": int(row.get("volume", 0)),
            })

        return {
            "ticker": ticker,
            "days": days,
            "rows": len(prices),
            "data_source": "postgresql",
            "prices": prices,
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": int(time.time()),
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