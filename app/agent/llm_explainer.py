import os
import logging
import asyncio
from typing import Dict, Any

from openai import AsyncOpenAI

logger = logging.getLogger("marketsentinel.llm")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "12"))
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMExplainer:

    def __init__(self):
        self.enabled = LLM_ENABLED

    async def explain(
        self,
        signal_row: Dict[str, Any],
        agent_output: Dict[str, Any],
        probability_stats: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self.enabled:
            return {
                "llm_enabled": False,
                "message": "LLM explanation disabled"
            }

        prompt = self._build_prompt(signal_row, agent_output, probability_stats)

        try:

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a financial risk explanation engine. Do NOT change signals. Only explain."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                ),
                timeout=OPENAI_TIMEOUT
            )

            content = response.choices[0].message.content

            return {
                "llm_enabled": True,
                "narrative": content
            }

        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)

            return {
                "llm_enabled": True,
                "error": "llm_unavailable"
            }

    def _build_prompt(self, row, agent, stats):

        return f"""
Signal Summary:
Ticker: {row.get("ticker")}
Signal: {row.get("signal")}
Score: {row.get("score")}
Rank Percentile: {row.get("rank_pct")}

Agent Assessment:
Confidence: {agent.get("confidence")}
Trend: {agent.get("trend")}
Volatility Regime: {agent.get("volatility_regime")}
Momentum State: {agent.get("momentum_state")}
Warnings: {agent.get("warnings")}

Market Context:
Probability Mean: {stats.get("mean")}
Probability Std: {stats.get("std")}

Explain this signal in clear institutional language.
Do not invent new numbers.
Do not change the signal.
"""