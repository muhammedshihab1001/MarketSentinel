# =========================================================
# PORTFOLIO SUMMARY ROUTE v2.5
#
# Changes from v2.4:
# FIX 1: gross_exposure and net_exposure now read from the
#         executive_summary block of the cached snapshot —
#         they were previously always 0 because the old
#         pipeline didn't include them in the result dict.
# FIX 2: positions list now built from snapshot signals
#         with correct signal direction derived from weight.
# FIX 3: portfolio_health_score derived from drift severity
#         and exposure scale when no portfolio agent output.
# FIX 4: top_5_preview built from executive_summary.top_5_tickers.
# FIX 5: approved_trades / rejected_trades counts from
#         portfolio agent output when available.
# =========================================================

import time
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.portfolio")

router = APIRouter()

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"


def _weight_to_signal(weight: float) -> str:
    if weight > 0.01:
        return "LONG"
    if weight < -0.01:
        return "SHORT"
    return "NEUTRAL"


def _drift_to_health(drift_state: str, severity_score: int) -> float:
    """
    Derive a 0-100 portfolio health score from drift state.
    Used when portfolio agent doesn't return an explicit score.
    """
    if drift_state == "hard":
        return max(0.0, 40.0 - severity_score * 2)
    if drift_state == "soft":
        return max(40.0, 75.0 - severity_score * 3)
    return 92.0


@router.get("/portfolio")
async def get_portfolio(request=None):
    """
    Returns portfolio summary derived from the latest background snapshot.

    Response shape:
        snapshot_date, gross_exposure, net_exposure,
        long_count, short_count, neutral_count,
        approved_trades, rejected_trades,
        drift_detected, drift_state,
        portfolio_health_score,
        positions: [{ ticker, weight, signal }],
        top_5_preview: [{ ticker, score, weight }]
    """
    endpoint = "/portfolio"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        # ── Get cache from app state ──────────────────
        cache = None
        if request is not None:
            try:
                cache = request.app.state.cache
            except AttributeError:
                pass

        if cache is None:
            from app.inference.cache import RedisCache
            cache = RedisCache()

        # ── Load background snapshot ──────────────────
        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)

        if not snapshot_result:
            raise HTTPException(
                status_code=503,
                detail="No snapshot available yet. Background compute is pending.",
            )

        # ── Extract data from snapshot result ─────────
        meta = snapshot_result.get("meta", {})
        exec_summary = snapshot_result.get("executive_summary", {})
        snapshot = snapshot_result.get("snapshot", {})
        portfolio_agent = snapshot_result.get("_portfolio", {})

        signals = snapshot.get("signals", [])
        drift = snapshot.get("drift", {})

        drift_state = drift.get("drift_state", "none")
        drift_detected = drift.get("drift_detected", False)
        severity_score = int(drift.get("severity_score", 0))

        # ── FIX: Read gross/net from executive_summary ─
        gross_exposure = float(exec_summary.get("gross_exposure", 0.0))
        net_exposure = float(exec_summary.get("net_exposure", 0.0))

        # ── Build positions list ──────────────────────
        positions = []
        long_count = 0
        short_count = 0
        neutral_count = 0

        for sig in signals:
            weight = float(sig.get("weight", 0.0))
            direction = _weight_to_signal(weight)

            if direction == "LONG":
                long_count += 1
            elif direction == "SHORT":
                short_count += 1
            else:
                neutral_count += 1

            positions.append({
                "ticker": sig.get("ticker", ""),
                "weight": round(weight, 6),
                "signal": direction,
            })

        # Sort by abs weight descending
        positions.sort(key=lambda x: abs(x["weight"]), reverse=True)

        # ── FIX: top_5_preview from executive_summary ─
        top_5_tickers = exec_summary.get("top_5_tickers", [])
        ticker_score_map = {
            s["ticker"]: s.get("hybrid_consensus_score", s.get("raw_model_score", 0.0))
            for s in signals
        }
        ticker_weight_map = {s["ticker"]: s.get("weight", 0.0) for s in signals}

        top_5_preview = [
            {
                "ticker": t,
                "score": round(float(ticker_score_map.get(t, 0.0)), 6),
                "weight": round(float(ticker_weight_map.get(t, 0.0)), 6),
            }
            for t in top_5_tickers
        ]

        # ── FIX: portfolio_health_score ───────────────
        if portfolio_agent and "score" in portfolio_agent:
            health_score = round(float(portfolio_agent["score"]) * 100, 1)
        else:
            health_score = round(_drift_to_health(drift_state, severity_score), 1)

        # ── Approved / rejected trades ────────────────
        approved_trades = portfolio_agent.get("approved_trades", long_count + short_count)
        rejected_trades = portfolio_agent.get("rejected_trades", 0)

        return {
            "snapshot_date": snapshot.get("snapshot_date", ""),
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "long_count": long_count,
            "short_count": short_count,
            "neutral_count": neutral_count,
            "approved_trades": int(approved_trades),
            "rejected_trades": int(rejected_trades),
            "drift_detected": drift_detected,
            "drift_state": drift_state,
            "portfolio_health_score": health_score,
            "positions": positions,
            "top_5_preview": top_5_preview,
        }

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Portfolio route failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)