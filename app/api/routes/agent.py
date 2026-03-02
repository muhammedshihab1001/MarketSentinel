# =========================================================
# HYBRID AGENT EXPLANATION ROUTE v2.0
# Multi-Agent + Hybrid Consensus Compatible
# =========================================================

import logging
import asyncio
import time
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline
from app.agent.llm_explainer import LLMExplainer

router = APIRouter()
logger = logging.getLogger("marketsentinel.agent")

# =========================================================
# SINGLETON PIPELINE
# =========================================================

_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing InferencePipeline (singleton)")
        _pipeline = InferencePipeline()
    return _pipeline


# =========================================================
# LAZY LLM EXPLAINER
# =========================================================

_explainer_instance: LLMExplainer | None = None


def get_explainer() -> LLMExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        logger.info("Initializing LLMExplainer (lazy)")
        _explainer_instance = LLMExplainer()
    return _explainer_instance


# =========================================================
# HYBRID AGENT EXPLANATION ROUTE
# =========================================================

@router.post("/agent/explain")
async def explain_signal(
    ticker: str = Query(..., description="Stock ticker symbol"),
    include_llm: bool = Query(True, description="Include LLM explanation")
):

    start_time = time.time()

    try:

        if not ticker or not isinstance(ticker, str):
            raise HTTPException(status_code=400, detail="Invalid ticker")

        ticker = ticker.upper().strip()

        pipeline = get_pipeline()

        snapshot = await asyncio.wait_for(
            run_in_threadpool(
                pipeline.run_snapshot,
                [ticker]
            ),
            timeout=180
        )

        if not isinstance(snapshot, dict):
            raise HTTPException(status_code=500, detail="Invalid snapshot response")

        signals = snapshot.get("signals", [])

        if not signals:
            raise HTTPException(status_code=404, detail="Ticker not found")

        row = next(
            (s for s in signals if s.get("ticker") == ticker),
            None
        )

        if row is None:
            raise HTTPException(status_code=404, detail="Ticker not in snapshot")

        # -------------------------------------------------
        # Extract Agent Outputs (Hybrid Compatible)
        # -------------------------------------------------

        agents = row.get("agents", {})

        signal_output = agents.get("signal_agent", {})
        technical_output = agents.get("technical_agent", {})

        drift_info = snapshot.get("drift", {})

        # -------------------------------------------------
        # Optional LLM Explanation
        # -------------------------------------------------

        llm_output = None

        if include_llm:
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
        # Structured Response (Hybrid-Aware)
        # -------------------------------------------------

        response = {
            "ticker": ticker,
            "snapshot_date": snapshot.get("snapshot_date"),

            # ----------------------------
            # Core Model Output
            # ----------------------------
            "raw_model_score": row.get("raw_model_score"),
            "weight": row.get("weight"),

            # ----------------------------
            # Hybrid Scores
            # ----------------------------
            "agent_score": row.get("agent_score"),
            "technical_score": row.get("technical_score"),
            "hybrid_consensus_score": row.get("hybrid_consensus_score"),

            # ----------------------------
            # Signal Agent Details
            # ----------------------------
            "signal": signal_output.get("signal"),
            "confidence_numeric": signal_output.get("confidence_numeric"),
            "governance_score": signal_output.get("governance_score"),
            "risk_level": signal_output.get("risk_level"),
            "volatility_regime": signal_output.get("volatility_regime"),

            # ----------------------------
            # Technical Agent Details
            # ----------------------------
            "technical_bias": technical_output.get("bias"),
            "technical_confidence": technical_output.get("confidence"),
            "technical_component_scores": technical_output.get("component_scores"),

            # ----------------------------
            # Drift Governance
            # ----------------------------
            "drift_state": drift_info.get("drift_state"),
            "drift_severity": drift_info.get("severity_score"),
            "exposure_scale": drift_info.get("exposure_scale"),

            # ----------------------------
            # Explanation Layer
            # ----------------------------
            "warnings": signal_output.get("warnings", []),
            "explanation": signal_output.get("explanation", ""),

            # ----------------------------
            # Optional LLM Narrative
            # ----------------------------
            "llm": llm_output,

            # ----------------------------
            # Observability
            # ----------------------------
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": int(time.time())
        }

        return response

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent explanation timeout")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Agent explanation failed")
        raise HTTPException(status_code=500, detail=str(e))