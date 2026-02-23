import logging
import asyncio
import time
import threading
import hashlib
import json
from typing import Dict, Any

from openai import AsyncOpenAI

from core.config.env_loader import (
    get_env,
    get_bool,
    get_int,
)

logger = logging.getLogger("marketsentinel.llm")


class LLMExplainer:

    ########################################################
    # INIT
    ########################################################

    def __init__(self):

        self.enabled = get_bool("LLM_ENABLED", False)
        self.model_name = get_env("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = get_int("OPENAI_TIMEOUT", 12)

        # 🔐 Rate limiting
        self.rate_limit_per_minute = get_int("LLM_RATE_LIMIT_PER_MIN", 30)

        # 📦 Cache
        self.cache_enabled = get_bool("LLM_CACHE_ENABLED", True)
        self.cache_ttl_seconds = get_int("LLM_CACHE_TTL_SEC", 120)

        self._request_times = []
        self._rate_lock = threading.Lock()

        self._cache = {}
        self._cache_lock = threading.Lock()

        api_key = get_env("OPENAI_API_KEY")

        if self.enabled and not api_key:
            logger.warning("LLM enabled but OPENAI_API_KEY missing.")
            self.enabled = False

        self.client = AsyncOpenAI(api_key=api_key) if self.enabled else None

    ########################################################
    # RATE LIMIT
    ########################################################

    def _check_rate_limit(self):

        if self.rate_limit_per_minute <= 0:
            return True

        now = time.time()

        with self._rate_lock:

            self._request_times = [
                t for t in self._request_times
                if now - t < 60
            ]

            if len(self._request_times) >= self.rate_limit_per_minute:
                return False

            self._request_times.append(now)
            return True

    ########################################################
    # CACHE
    ########################################################

    def _cache_key(self, row, agent, stats):

        payload = {
            "ticker": row.get("ticker"),
            "signal": row.get("signal"),
            "score": row.get("score"),
            "rank_pct": row.get("rank_pct"),
            "confidence": agent.get("confidence"),
            "trend": agent.get("trend"),
            "volatility_regime": agent.get("volatility_regime"),
            "momentum_state": agent.get("momentum_state"),
            "prob_mean": stats.get("mean"),
            "prob_std": stats.get("std"),
        }

        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _get_cached(self, key):

        if not self.cache_enabled:
            return None

        with self._cache_lock:

            entry = self._cache.get(key)

            if not entry:
                return None

            data, ts = entry

            if time.time() - ts > self.cache_ttl_seconds:
                del self._cache[key]
                return None

            return data

    def _set_cache(self, key, value):

        if not self.cache_enabled:
            return

        with self._cache_lock:
            self._cache[key] = (value, time.time())

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

        cache_key = self._cache_key(
            signal_row,
            agent_output,
            probability_stats
        )

        cached = self._get_cached(cache_key)
        if cached:
            return {
                **cached,
                "cached": True
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
                                "You are a financial explanation engine.\n"
                                "You MUST return ONLY valid JSON.\n"
                                "You MUST NOT modify signals.\n"
                                "Return format:\n"
                                "{\n"
                                '  "summary": "...",\n'
                                '  "rationale": "...",\n'
                                '  "risk_commentary": "...",\n'
                                '  "outlook": "..."\n'
                                "}"
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

            raw_content = response.choices[0].message.content.strip()

            parsed = self._safe_parse_json(raw_content)

            result = {
                "llm_enabled": True,
                "model": self.model_name,
                "structured": parsed,
                "cached": False
            }

            self._set_cache(cache_key, result)

            return result

        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)
            return {
                "llm_enabled": True,
                "error": "llm_unavailable"
            }

    ########################################################
    # SAFE JSON PARSER
    ########################################################

    def _safe_parse_json(self, content: str):

        try:
            data = json.loads(content)

            required = {
                "summary",
                "rationale",
                "risk_commentary",
                "outlook"
            }

            if not required.issubset(data.keys()):
                raise ValueError("Missing required keys")

            return data

        except Exception:
            logger.warning("Malformed LLM JSON response.")
            return {
                "summary": "Explanation unavailable.",
                "rationale": "Model response could not be parsed.",
                "risk_commentary": "Unavailable.",
                "outlook": "Unavailable."
            }

    ########################################################
    # PROMPT BUILDER
    ########################################################

    def _build_prompt(self, row, agent, stats):

        return f"""
Ticker: {row.get("ticker")}
Signal: {row.get("signal")}
Score: {row.get("score")}
Rank Percentile: {row.get("rank_pct")}

Confidence: {agent.get("confidence")}
Trend: {agent.get("trend")}
Volatility Regime: {agent.get("volatility_regime")}
Momentum State: {agent.get("momentum_state")}
Warnings: {agent.get("warnings")}

Probability Mean: {stats.get("mean")}
Probability Std: {stats.get("std")}

Explain this signal clearly and professionally.
"""