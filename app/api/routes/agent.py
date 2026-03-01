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

        # -------------------------------------------------
        # Run snapshot safely in threadpool
        # -------------------------------------------------

        snapshot = await asyncio.wait_for(
            run_in_threadpool(
                pipeline.run_snapshot,
                [ticker]
            ),
            timeout=25
        )

        if not isinstance(snapshot, dict):
            raise HTTPException(status_code=500, detail="Invalid snapshot response")

        signals = snapshot.get("signals")

        if not signals or not isinstance(signals, list):
            raise HTTPException(status_code=404, detail="Ticker not found")

        # -------------------------------------------------
        # Find correct ticker row
        # -------------------------------------------------

        row = next(
            (s for s in signals if s.get("ticker") == ticker),
            None
        )

        if row is None:
            raise HTTPException(status_code=404, detail="Ticker not in snapshot")

        agent_output = row.get("agent", {})
        drift_info = snapshot.get("drift", {})

        # -------------------------------------------------
        # Optional LLM Explanation
        # -------------------------------------------------

        llm_output = None

        if include_llm:
            explainer = get_explainer()

            llm_output = await explainer.explain(
                signal_row=row,
                agent_output=agent_output,
                probability_stats={
                    "mean_score": snapshot.get("score_mean"),
                    "std_score": snapshot.get("score_std"),
                }
            )

        # -------------------------------------------------
        # Structured Response (Aligned With Current Schema)
        # -------------------------------------------------

        response = {
            "ticker": ticker,
            "snapshot_date": snapshot.get("snapshot_date"),

            # Core signal
            "signal": agent_output.get("signal"),
            "raw_model_score": row.get("raw_model_score"),
            "weight": row.get("weight"),

            # Agent metrics
            "agent_score": agent_output.get("agent_score"),
            "confidence_numeric": agent_output.get("confidence_numeric"),
            "governance_score": agent_output.get("governance_score"),
            "risk_level": agent_output.get("risk_level"),
            "volatility_regime": agent_output.get("volatility_regime"),

            # Drift governance
            "drift_state": drift_info.get("drift_state"),
            "drift_severity": drift_info.get("severity_score"),
            "exposure_scale": drift_info.get("exposure_scale"),

            # Reasoning
            "warnings": agent_output.get("warnings", []),
            "explanation": agent_output.get("explanation", ""),

            # Optional LLM narrative
            "llm": llm_output,

            # Observability
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