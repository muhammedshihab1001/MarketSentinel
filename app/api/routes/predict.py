# (FULL FILE STARTS HERE)

import time
import asyncio
import os
import re
import logging
import math
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.concurrency import run_in_threadpool

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

MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
REQUEST_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT_SEC", "25"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "30"))
MIN_BATCH_SIZE = 4

DEFAULT_USE_UNIVERSE = os.getenv(
    "DEFAULT_USE_UNIVERSE",
    "true"
).lower() == "true"

PRIMARY_UNIVERSE_PATH = Path(
    os.getenv("PRODUCTION_UNIVERSE_PATH", "config/universe_production.json")
)

FALLBACK_UNIVERSE_PATH = Path("config/universe.json")

inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,12}$")


# =========================================================
# DEFAULT UNIVERSE LOADER
# =========================================================

def load_default_universe() -> List[str]:

    universe_path = None

    if PRIMARY_UNIVERSE_PATH.exists():
        universe_path = PRIMARY_UNIVERSE_PATH
        logger.info("Using production universe file.")
    elif FALLBACK_UNIVERSE_PATH.exists():
        universe_path = FALLBACK_UNIVERSE_PATH
        logger.warning("Production universe missing. Using fallback universe.json")
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

    cleaned = []

    for t in tickers:
        t = str(t).upper().strip()
        if TICKER_REGEX.match(t):
            cleaned.append(t)

    unique = sorted(set(cleaned))

    if len(unique) < MIN_BATCH_SIZE:
        raise RuntimeError("Universe size below minimum batch size.")

    return unique


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
# LIVE SNAPSHOT (ENHANCED)
# =========================================================

@router.get("/live-snapshot", response_model=SnapshotResponse)
async def live_snapshot():

    endpoint = "/live-snapshot"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:

        tickers = load_default_universe()
        loader = get_loader()
        pipeline = get_pipeline()

        async with inference_semaphore:

            snapshot = await asyncio.wait_for(
                run_in_threadpool(
                    pipeline.run_snapshot,
                    tickers
                ),
                timeout=REQUEST_TIMEOUT
            )

        signals = snapshot.get("signals", [])

        long_count = sum(1 for s in signals if s.get("signal") == "LONG")
        short_count = sum(1 for s in signals if s.get("signal") == "SHORT")

        # 🔥 NEW AGGREGATE METRICS
        strength_scores = [
            s.get("agent", {}).get("strength_score", 0.0)
            for s in signals
        ]

        avg_strength = round(sum(strength_scores) / len(strength_scores), 2) if strength_scores else 0.0
        max_strength = max(strength_scores) if strength_scores else 0.0
        min_strength = min(strength_scores) if strength_scores else 0.0

        high_conviction_count = sum(
            1 for s in signals
            if s.get("agent", {}).get("strength_score", 0.0) >= 75
        )

        elevated_risk_count = sum(
            1 for s in signals
            if s.get("agent", {}).get("risk_level") == "elevated"
        )

        meta = {
            "model_version": loader.xgb_version,
            "schema_signature": loader.schema_signature,
            "dataset_hash": loader.dataset_hash,
            "training_code_hash": loader.training_code_hash,
            "artifact_hash": loader.artifact_hash,
            "universe_size": len(tickers),
            "long_signals": long_count,
            "short_signals": short_count,
            "avg_strength_score": avg_strength,
            "max_strength_score": max_strength,
            "min_strength_score": min_strength,
            "high_conviction_count": high_conviction_count,
            "elevated_risk_count": elevated_risk_count,
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": int(time.time())
        }

        return SnapshotResponse(meta=meta, snapshot=snapshot)

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Snapshot inference timeout")

    except Exception:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Live snapshot failure")
        raise HTTPException(status_code=500, detail="Live snapshot failed")

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)