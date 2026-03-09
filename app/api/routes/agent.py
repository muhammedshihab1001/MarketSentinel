# =========================================================
# HYBRID AGENT EXPLANATION ROUTE v3.0
# Multi-Agent + Hybrid Consensus Compatible
# =========================================================

import logging
import asyncio
import time
import os
import re
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.api.routes.predict import get_pipeline
from app.agent.llm_explainer import LLMExplainer

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

# political risk agent (lazy import to avoid startup penalty)
try:
    from core.agent.political_risk_agent import PoliticalRiskAgent
except Exception:
    PoliticalRiskAgent = None


router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger("marketsentinel.agent")

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\-]{1,12}$")


# =========================================================
# RESPONSE HELPERS
# =========================================================

def success(data):
    return {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def error(message):
    return {
        "success": False,
        "data": None,
        "error": message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =========================================================
# LAZY LLM EXPLAINER
# =========================================================

_explainer_instance = None


def get_explainer():

    global _explainer_instance

    if _explainer_instance is None:
        logger.info("Initializing LLMExplainer (lazy)")
        _explainer_instance = LLMExplainer()

    return _explainer_instance


# =========================================================
# AGENT REGISTRY
# =========================================================

AVAILABLE_AGENTS = {
    "signal_agent": True,
    "technical_risk_agent": True,
    "portfolio_decision_agent": True,
    "political_risk_agent": PoliticalRiskAgent is not None
}


# =========================================================
# LIST AVAILABLE AGENTS
# =========================================================

@router.get("/agents")
def list_agents():

    endpoint = "/agent/agents"
    start_time = time.time()

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    try:

        return success({
            "agents": AVAILABLE_AGENTS
        })

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Agent listing failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# POLITICAL RISK AGENT
# =========================================================

@router.get("/political-risk")
async def political_risk(
    ticker: str = Query(..., description="Stock ticker symbol"),
    country: str = Query("US", description="Country code")
):

    endpoint = "/agent/political-risk"
    start_time = time.time()

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    try:

        if PoliticalRiskAgent is None:
            raise HTTPException(
                status_code=503,
                detail="political_risk_agent_not_available"
            )

        ticker = ticker.upper().strip()

        if not TICKER_REGEX.match(ticker):
            raise HTTPException(
                status_code=400,
                detail="invalid_ticker"
            )

        agent = PoliticalRiskAgent()

        result = await asyncio.wait_for(
            run_in_threadpool(
                agent.get_political_risk,
                ticker,
                country
            ),
            timeout=30
        )

        return success(result)

    except asyncio.TimeoutError:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        raise HTTPException(
            status_code=504,
            detail="political_risk_timeout"
        )

    except HTTPException:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Political risk agent failure")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )


# =========================================================
# HYBRID AGENT EXPLANATION
# =========================================================

@router.post("/explain")
async def explain_signal(
    ticker: str = Query(..., description="Stock ticker symbol"),
    include_llm: bool = Query(True, description="Include LLM explanation")
):

    endpoint = "/agent/explain"
    start_time = time.time()

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    try:

        if not ticker or not isinstance(ticker, str):
            raise HTTPException(status_code=400, detail="invalid_ticker")

        ticker = ticker.upper().strip()

        if not TICKER_REGEX.match(ticker):
            raise HTTPException(status_code=400, detail="invalid_ticker")

        pipeline = get_pipeline()

        snapshot = await asyncio.wait_for(
            run_in_threadpool(
                pipeline.run_snapshot,
                [ticker]
            ),
            timeout=180
        )

        if not isinstance(snapshot, dict):
            raise HTTPException(
                status_code=500,
                detail="invalid_snapshot_response"
            )

        signals = snapshot.get("signals", [])

        if not signals:
            raise HTTPException(
                status_code=404,
                detail="ticker_not_found"
            )

        row = next(
            (s for s in signals if s.get("ticker") == ticker),
            None
        )

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="ticker_not_in_snapshot"
            )

        agents = row.get("agents", {})

        signal_output = agents.get("signal_agent", {})
        technical_output = agents.get("technical_agent", {})

        drift_info = snapshot.get("drift", {})

        # -------------------------------------------------
        # Optional LLM Explanation
        # -------------------------------------------------

        llm_output = None

        if include_llm and os.getenv("LLM_ENABLED", "0") == "1":

            explainer = get_explainer()

            try:

                llm_output = await explainer.explain(
                    signal_row=row,
                    signal_output=signal_output,
                    technical_output=technical_output,
                    drift_stats=drift_info
                )

            except Exception:

                logger.exception("LLM explanation failed (non-blocking)")

                llm_output = {
                    "llm_enabled": True,
                    "error": "llm_runtime_failure"
                }

        # -------------------------------------------------
        # Structured Response
        # -------------------------------------------------

        response = {

            "ticker": ticker,
            "snapshot_date": snapshot.get("snapshot_date"),

            "raw_model_score": row.get("raw_model_score"),
            "weight": row.get("weight"),

            "agent_score": row.get("agent_score"),
            "technical_score": row.get("technical_score"),
            "hybrid_consensus_score": row.get("hybrid_consensus_score"),

            "signal": signal_output.get("signal"),
            "confidence_numeric": signal_output.get("confidence_numeric"),
            "governance_score": signal_output.get("governance_score"),
            "risk_level": signal_output.get("risk_level"),
            "volatility_regime": signal_output.get("volatility_regime"),

            "technical_bias": technical_output.get("bias"),
            "technical_confidence": technical_output.get("confidence"),
            "technical_component_scores": technical_output.get("component_scores"),

            "drift_state": drift_info.get("drift_state"),
            "drift_severity": drift_info.get("severity_score"),
            "exposure_scale": drift_info.get("exposure_scale"),

            "warnings": signal_output.get("warnings", []),
            "explanation": signal_output.get("explanation", ""),

            "llm": llm_output,

            "latency_ms": int((time.time() - start_time) * 1000),
        }

        return success(response)

    except asyncio.TimeoutError:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        raise HTTPException(
            status_code=504,
            detail="agent_timeout"
        )

    except HTTPException:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Agent explanation failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )