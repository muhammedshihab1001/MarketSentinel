import logging
import asyncio
from typing import Dict, Any

from openai import AsyncOpenAI

from core.config.env_loader import (
    get_env,
    get_bool,
    get_int,
)

logger = logging.getLogger("marketsentinel.llm")


class LLMExplainer:

    def __init__(self):

        # 🔐 Use centralized env loader
        self.enabled = get_bool("LLM_ENABLED", False)
        self.model_name = get_env("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = get_int("OPENAI_TIMEOUT", 12)

        api_key = get_env("OPENAI_API_KEY")

        if self.enabled and not api_key:
            logger.warning("LLM enabled but OPENAI_API_KEY missing.")
            self.enabled = False

        self.client = AsyncOpenAI(api_key=api_key) if self.enabled else None

    ########################################################
    # PUBLIC API
    ########################################################

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

        prompt = self._build_prompt(
            signal_row,
            agent_output,
            probability_stats
        )

        try:

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a financial explanation engine. "
                                "You MUST NOT change trading signals. "
                                "You only explain the provided signal."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                ),
                timeout=self.timeout
            )

            content = response.choices[0].message.content

            return {
                "llm_enabled": True,
                "model": self.model_name,
                "narrative": content
            }

        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)

            return {
                "llm_enabled": True,
                "error": "llm_unavailable"
            }

    ########################################################
    # PROMPT BUILDER
    ########################################################

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

Explain this signal in professional institutional language.
Do not invent new numbers.
Do not modify the signal.
"""