# =========================================================
# HYBRID AGENT EXPLANATION ROUTE v3.3
#
# Changes from v3.2:
# FIX 1: /agent/explain now reads per-ticker agent details
#         from snapshot cache _signal_details dict written
#         by pipeline.py v5.6. Previously always returned
#         empty agent data because the cache key didn't
#         match what the pipeline stored.
# FIX 2: Response shape aligned to AgentExplainResponse
#         in frontend types/index.ts — uses data wrapper.
# FIX 3: /agent/political-risk reads from political agent
#         cache stored in snapshot _political key.
# FIX 4: /agent/agents returns agent descriptions without
#         running inference — was crashing on model call.
# =========================================================

import logging
import asyncio
import time
import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.agent")

router = APIRouter()

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"


def _derive_signal(weight: float) -> str:
    if weight > 0.01:
        return "LONG"
    if weight < -0.01:
        return "SHORT"
    return "NEUTRAL"


def _derive_risk_level(score: float) -> str:
    abs_score = abs(score)
    if abs_score >= 0.8:
        return "high"
    if abs_score >= 0.5:
        return "medium"
    return "low"


def _derive_volatility_regime(technical_output: dict) -> str:
    component_scores = technical_output.get("component_scores", {})
    signals = technical_output.get("signals", {})
    regime = signals.get("volatility_regime", "normal")
    if regime in ("high_volatility", "low_volatility", "normal"):
        return regime
    return "normal"


# =========================================================
# GET /agent/explain?ticker=X
# =========================================================

@router.get("/agent/explain")
@router.post("/agent/explain")
async def explain_signal(
    request: Request,
    ticker: str = Query(None),
):
    """
    Returns signal explanation for a ticker.
    Reads from the background snapshot cache where per-ticker
    agent details were stored by the inference pipeline.
    """
    endpoint = "/agent/explain"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    # Support both GET ?ticker=X and POST body
    if ticker is None:
        try:
            body = await request.json()
            ticker = body.get("ticker")
        except Exception:
            pass

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker parameter is required")

    ticker = ticker.upper().strip()

    try:
        cache = request.app.state.cache
    except AttributeError:
        from app.inference.cache import RedisCache
        cache = RedisCache()

    try:
        # ── Load snapshot from cache ──────────────────
        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)

        if not snapshot_result:
            raise HTTPException(
                status_code=503,
                detail="No snapshot available. Background compute is pending.",
            )

        # ── Find ticker in signals ────────────────────
        signals = snapshot_result.get("snapshot", {}).get("signals", [])
        signal_row = next((s for s in signals if s["ticker"] == ticker), None)

        if signal_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"{ticker} not found in current snapshot. Check universe.",
            )

        # ── FIX: Read agent details from _signal_details ─
        signal_details = snapshot_result.get("_signal_details", {})
        agents = signal_details.get(ticker, {})

        signal_agent_output = agents.get("signal_agent", {})
        technical_output = agents.get("technical_agent", {})

        # Derive values from agent outputs
        raw_score = float(signal_row.get("raw_model_score", 0.0))
        hybrid_score = float(signal_row.get("hybrid_consensus_score", 0.0))
        weight = float(signal_row.get("weight", 0.0))

        signal_direction = (
            signal_agent_output.get("signals", {}).get("signal")
            or _derive_signal(weight)
        )

        confidence_numeric = signal_agent_output.get("confidence")
        if confidence_numeric is not None:
            confidence_numeric = round(float(confidence_numeric), 4)

        governance_score = signal_agent_output.get("governance_score")
        if governance_score is not None:
            governance_score = int(governance_score)

        risk_level = (
            signal_agent_output.get("risk_level")
            or _derive_risk_level(raw_score)
        )

        volatility_regime = _derive_volatility_regime(technical_output)

        technical_bias = (
            technical_output.get("bias")
            or technical_output.get("signals", {}).get("bias", "neutral")
        )

        warnings = signal_agent_output.get("warnings", [])
        explanation = signal_agent_output.get("explanation", "")

        drift_state = (
            snapshot_result.get("snapshot", {})
            .get("drift", {})
            .get("drift_state", "none")
        )

        # ── LLM section (disabled unless LLM_ENABLED=true) ─
        llm_output = None
        if os.getenv("LLM_ENABLED", "false").lower() == "true":
            try:
                from app.agent.llm_explainer import LLMExplainer
                explainer = LLMExplainer()
                if explainer._enabled:
                    llm_result = await asyncio.to_thread(
                        explainer.explain,
                        ticker=ticker,
                        signal=signal_direction,
                        score=raw_score,
                        context={
                            "risk_level": risk_level,
                            "warnings": warnings,
                            "drift_state": drift_state,
                        },
                    )
                    llm_output = llm_result
            except Exception as e:
                logger.debug("LLM explain failed: %s", e)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # ── FIX: Response wrapped in data key ────────
        return {
            "success": True,
            "data": {
                "ticker": ticker,
                "snapshot_date": signal_row.get("date", ""),
                "raw_model_score": round(raw_score, 6),
                "weight": round(weight, 6),
                "hybrid_consensus_score": round(hybrid_score, 6),
                "signal": signal_direction,
                "confidence_numeric": confidence_numeric,
                "governance_score": governance_score,
                "risk_level": risk_level,
                "volatility_regime": volatility_regime,
                "technical_bias": technical_bias,
                "drift_state": drift_state,
                "warnings": warnings,
                "explanation": explanation,
                "llm": llm_output,
                "latency_ms": latency_ms,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Agent explain failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /agent/political-risk?ticker=X
# =========================================================

@router.get("/agent/political-risk")
async def political_risk(
    request: Request,
    ticker: str = Query(...),
):
    """
    Returns political risk for a ticker's country (US default).
    Reads from snapshot cache _political key — no extra API call.
    """
    endpoint = "/agent/political-risk"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    ticker = ticker.upper().strip()

    try:
        cache = request.app.state.cache
    except AttributeError:
        from app.inference.cache import RedisCache
        cache = RedisCache()

    try:
        # FIX: Read from snapshot cache _political key
        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)
        political = {}

        if snapshot_result:
            political = snapshot_result.get("_political", {})

        if not political:
            # Live fallback
            from core.agent.political_risk_agent import PoliticalRiskAgent
            agent = PoliticalRiskAgent()
            political = agent.get_political_risk(ticker, country="US")

        return {
            "ticker": ticker,
            "political_risk_score": float(political.get("political_risk_score", 0.0)),
            "political_risk_label": political.get("political_risk_label", "LOW"),
            "top_events": political.get("top_events", [])[:5],
            "source": political.get("source", "gdelt"),
            "served_from_cache": bool(snapshot_result),
            "latency_ms": round((time.time() - start_time) * 1000, 1),
        }

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Political risk failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /agent/agents
# FIX: Returns static descriptions — no inference call
# =========================================================

@router.get("/agent/agents")
async def list_agents():
    """
    Returns descriptions of all agents in the pipeline.
    Static response — no inference required.
    """
    return {
        "agents": {
            "signal_agent": {
                "name": "SignalAgent",
                "description": "Interprets XGBoost model output into LONG/SHORT/NEUTRAL signal with confidence and risk level.",
                "weight": 0.5,
            },
            "technical_risk_agent": {
                "name": "TechnicalRiskAgent",
                "description": "Evaluates momentum, EMA structure, RSI, and volatility regime for technical quality score.",
                "weight": 0.2,
            },
            "portfolio_decision_agent": {
                "name": "PortfolioDecisionAgent",
                "description": "Aggregates per-ticker signals into portfolio-level decisions with exposure control.",
                "weight": 0.2,
            },
            "political_risk_agent": {
                "name": "PoliticalRiskAgent",
                "description": "Detects geopolitical and macro risk events via GDELT headlines. CRITICAL label overrides trading signals.",
                "weight": 0.1,
            },
        }
    }