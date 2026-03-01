import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline
from app.agent.llm_explainer import LLMExplainer

router = APIRouter()
logger = logging.getLogger("marketsentinel.agent")

# =========================================================
# LAZY LLM EXPLAINER (SAFE INIT)
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

    try:
        if not ticker or not isinstance(ticker, str):
            raise HTTPException(status_code=400, detail="Invalid ticker")

        ticker = ticker.upper().strip()

        pipeline = InferencePipeline()

        # -------------------------------------------------
        # Run snapshot safely in threadpool
        # -------------------------------------------------

        snapshot = await run_in_threadpool(
            pipeline.run_snapshot,
            [ticker]
        )

        if not isinstance(snapshot, dict):
            raise HTTPException(status_code=500, detail="Invalid snapshot response")

        signals = snapshot.get("signals")

        if not signals or not isinstance(signals, list):
            raise HTTPException(status_code=404, detail="Ticker not found")

        # -------------------------------------------------
        # Find correct ticker row (safe)
        # -------------------------------------------------

        row = None
        for s in signals:
            if s.get("ticker") == ticker:
                row = s
                break

        if row is None:
            raise HTTPException(status_code=404, detail="Ticker not in snapshot")

        agent_output = row.get("agent", {})
        drift_info = snapshot.get("drift", {})

        probability_stats = {
            "mean_score": snapshot.get("score_mean"),
            "std_score": snapshot.get("score_std"),
        }

        # -------------------------------------------------
        # Optional LLM EXPLANATION
        # -------------------------------------------------

        llm_output = None

        if include_llm:
            explainer = get_explainer()

            llm_output = await explainer.explain(
                signal_row=row,
                agent_output=agent_output,
                probability_stats=probability_stats
            )

        # -------------------------------------------------
        # Structured Institutional Response
        # -------------------------------------------------

        response = {
            "ticker": ticker,
            "snapshot_date": snapshot.get("snapshot_date"),
            "signal": agent_output.get("signal"),
            "score": row.get("score"),
            "weight": row.get("weight"),

            # Hybrid breakdown
            "alpha_confidence": agent_output.get("confidence"),
            "strength_score": agent_output.get("strength_score"),
            "risk_level": agent_output.get("risk_level"),
            "trend": agent_output.get("trend"),
            "momentum": agent_output.get("momentum_state"),
            "macro_regime": agent_output.get("macro_regime"),
            "volatility_regime": agent_output.get("volatility_regime"),

            # Governance
            "drift_state": drift_info.get("drift_state"),
            "drift_severity": drift_info.get("severity_score"),
            "exposure_scale": drift_info.get("exposure_scale"),

            # Structured reasoning
            "warnings": agent_output.get("warnings", []),
            "reasoning": agent_output.get("reasoning", []),

            # Optional LLM narrative
            "llm": llm_output
        }

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Agent explanation failed")
        raise HTTPException(status_code=500, detail=str(e))