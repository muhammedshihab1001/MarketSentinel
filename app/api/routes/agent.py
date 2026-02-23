import logging
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline
from app.agent.llm_explainer import LLMExplainer

router = APIRouter()
logger = logging.getLogger("marketsentinel.agent")

explainer = LLMExplainer()


@router.post("/agent/explain")
async def explain_signal(ticker: str):

    try:
        pipeline = InferencePipeline()

        # Run snapshot safely in threadpool
        snapshot = await run_in_threadpool(
            pipeline.run_snapshot,
            [ticker]
        )

        if not isinstance(snapshot, dict):
            raise HTTPException(status_code=500, detail="Invalid snapshot response")

        signals = snapshot.get("signals")

        if not signals or not isinstance(signals, list):
            raise HTTPException(status_code=404, detail="Ticker not found")

        row = signals[0]

        agent_output = row.get("agent", {})
        probability_stats = snapshot.get("probability_stats", {})

        # LLM explanation
        explanation = await explainer.explain(
            signal_row=row,
            agent_output=agent_output,
            probability_stats=probability_stats
        )

        return {
            "ticker": ticker,
            "signal": row.get("signal"),
            "score": row.get("score"),
            "agent": agent_output,
            "llm": explanation
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Agent explanation failed")
        raise HTTPException(status_code=500, detail=str(e))