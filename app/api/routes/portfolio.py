# =========================================================
# PORTFOLIO SUMMARY ROUTE v2.6
#
# SWAGGER FIX v2.6:
# BUG FIX: get_portfolio(request=None) was using optional
#   Request which caused AttributeError when FastAPI
#   injected the real Request object. Changed to proper
#   FastAPI dependency injection: get_portfolio(request: Request).
# SWAGGER: Added summary, description, response_description
#   so the endpoint is fully documented in /docs.
# =========================================================

import time
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.portfolio")

router = APIRouter(tags=["portfolio"])

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"


def _weight_to_signal(weight: float) -> str:
    if weight > 0.01:
        return "LONG"
    if weight < -0.01:
        return "SHORT"
    return "NEUTRAL"


def _drift_to_health(drift_state: str, severity_score: int) -> float:
    if drift_state == "hard":
        return max(0.0, 40.0 - severity_score * 2)
    if drift_state == "soft":
        return max(40.0, 75.0 - severity_score * 3)
    return 92.0


@router.get(
    "/portfolio",
    summary="Portfolio Summary",
    description="""
Returns the current portfolio state derived from the latest background snapshot.

**Requires authentication** (owner or demo — demo gets 3 requests before lock).

Returns:
- `positions`: all 100 tickers with weight and signal direction (LONG/SHORT/NEUTRAL)
- `top_5_preview`: highest-scoring tickers this snapshot
- `gross_exposure` / `net_exposure`: portfolio exposure metrics
- `drift_state`: current model drift (none/low/moderate/high/critical)
- `portfolio_health_score`: 0-100 score derived from drift + exposure

**503** = background snapshot is still computing (~90s on first load).
Wait and retry — the snapshot cache refreshes every 300s.
""",
    response_description="Portfolio summary with positions, exposure, and health score.",
)
async def get_portfolio(request: Request):
    """
    Returns portfolio summary derived from the latest background snapshot.
    Returns 503 if no snapshot is cached yet (computing takes ~90s on first load).
    """
    endpoint = "/portfolio"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        # ── Get cache from app state ──────────────────
        try:
            cache = request.app.state.cache
        except AttributeError:
            from app.inference.cache import RedisCache
            cache = RedisCache()

        # ── Load background snapshot ──────────────────
        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)

        if not snapshot_result:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No snapshot available yet. "
                    "Background compute is pending (~90s on first load). "
                    "Retry in 30 seconds."
                ),
            )

        # ── Extract data from snapshot result ─────────
        exec_summary = snapshot_result.get("executive_summary", {})
        snapshot = snapshot_result.get("snapshot", {})
        portfolio_agent = snapshot_result.get("_portfolio", {})

        signals = snapshot.get("signals", [])
        drift = snapshot.get("drift", {})

        drift_state = drift.get("drift_state", "none")
        drift_detected = drift.get("drift_detected", False)
        severity_score = int(drift.get("severity_score", 0))

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

        positions.sort(key=lambda x: abs(x["weight"]), reverse=True)

        # ── top_5_preview ─────────────────────────────
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

        # ── portfolio_health_score ────────────────────
        if portfolio_agent and "score" in portfolio_agent:
            health_score = round(float(portfolio_agent["score"]) * 100, 1)
        else:
            health_score = round(_drift_to_health(drift_state, severity_score), 1)

        approved_trades = portfolio_agent.get(
            "approved_trades", long_count + short_count
        )
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