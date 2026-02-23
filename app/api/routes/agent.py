import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline
from app.agent.llm_explainer import LLMExplainer

router = APIRouter()
pipeline = InferencePipeline()
explainer = LLMExplainer()


@router.post("/agent/explain")
async def explain_signal(ticker: str):

    try:

        snapshot = await run_in_threadpool(
            pipeline.run_snapshot,
            [ticker]
        )

        signals = snapshot.get("signals", [])

        if not signals:
            raise HTTPException(status_code=404, detail="Ticker not found")

        row = signals[0]
        agent_output = row.get("agent", {})

        explanation = await explainer.explain(
            signal_row=row,
            agent_output=agent_output,
            probability_stats=snapshot.get("probability_stats", {})
        )

        return {
            "ticker": ticker,
            "signal": row.get("signal"),
            "score": row.get("score"),
            "agent": agent_output,
            "llm": explanation
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Explanation failed")