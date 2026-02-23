import logging
import asyncio
import time
import threading
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

        self.enabled = get_bool("LLM_ENABLED", False)
        self.model_name = get_env("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = get_int("OPENAI_TIMEOUT", 12)

        # 🔐 Rate limiting config
        self.rate_limit_per_minute = get_int("LLM_RATE_LIMIT_PER_MIN", 30)

        self._request_times = []
        self._rate_lock = threading.Lock()

        api_key = get_env("OPENAI_API_KEY")

        if self.enabled and not api_key:
            logger.warning("LLM enabled but OPENAI_API_KEY missing.")
            self.enabled = False

        self.client = AsyncOpenAI(api_key=api_key) if self.enabled else None

    ########################################################
    # RATE LIMIT CHECK
    ########################################################

    def _check_rate_limit(self):

        if self.rate_limit_per_minute <= 0:
            return True

        now = time.time()

        with self._rate_lock:

            # Remove timestamps older than 60 seconds
            self._request_times = [
                t for t in self._request_times
                if now - t < 60
            ]

            if len(self._request_times) >= self.rate_limit_per_minute:
                return False

            self._request_times.append(now)
            return True

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

        if not self._check_rate_limit():
            logger.warning("LLM rate limit exceeded.")

            return {
                "llm_enabled": True,
                "error": "rate_limit_exceeded"
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