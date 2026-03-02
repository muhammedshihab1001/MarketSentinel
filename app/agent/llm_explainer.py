# =========================================================
# LLM EXPLAINER v2.0
# Hybrid Multi-Agent Compatible
# CV-Ready | Clean | Backward Safe
# =========================================================

import logging
import asyncio
import time
import threading
import hashlib
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

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

        self.rate_limit_per_minute = get_int("LLM_RATE_LIMIT_PER_MIN", 30)

        self.cache_enabled = get_bool("LLM_CACHE_ENABLED", True)
        self.cache_ttl_seconds = get_int("LLM_CACHE_TTL_SEC", 180)

        self.audit_enabled = get_bool("LLM_AUDIT_ENABLED", True)

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

    def _cache_key(self, row, signal_output, technical_output, stats):

        payload = {
            "ticker": row.get("ticker"),
            "raw_model_score": row.get("raw_model_score"),
            "agent_score": row.get("agent_score"),
            "technical_score": row.get("technical_score"),
            "hybrid_consensus_score": row.get("hybrid_consensus_score"),
            "signal": signal_output.get("signal"),
            "confidence": signal_output.get("confidence_numeric"),
            "risk_level": signal_output.get("risk_level"),
            "technical_bias": technical_output.get("bias"),
            "technical_score": technical_output.get("score"),
            "drift_state": stats.get("drift_state"),
            "severity_score": stats.get("severity_score"),
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
    # AUDIT LOGGING
    ########################################################

    def _audit_log(
        self,
        ticker: str,
        signal: str,
        result: Dict[str, Any],
        cached: bool,
    ):

        if not self.audit_enabled:
            return

        try:

            response_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()

            audit_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "ticker": ticker,
                "signal": signal,
                "model": self.model_name,
                "cached": cached,
                "status": "success" if "error" not in result else "error",
                "response_hash": response_hash,
            }

            logger.info("LLM_AUDIT | %s", json.dumps(audit_record))

        except Exception:
            logger.warning("Failed to write LLM audit log.")

    ########################################################
    # PUBLIC API
    ########################################################

    async def explain(
        self,
        signal_row: Dict[str, Any],
        signal_output: Dict[str, Any],
        technical_output: Dict[str, Any],
        drift_stats: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self.enabled:
            return {
                "llm_enabled": False,
                "message": "LLM explanation disabled"
            }

        ticker = signal_row.get("ticker")
        signal = signal_output.get("signal")

        cache_key = self._cache_key(
            signal_row,
            signal_output,
            technical_output,
            drift_stats
        )

        cached = self._get_cached(cache_key)
        if cached:
            self._audit_log(ticker, signal, cached, cached=True)
            return {**cached, "cached": True}

        if not self._check_rate_limit():
            result = {
                "llm_enabled": True,
                "error": "rate_limit_exceeded"
            }
            self._audit_log(ticker, signal, result, cached=False)
            return result

        prompt = self._build_prompt(
            signal_row,
            signal_output,
            technical_output,
            drift_stats
        )

        try:

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an institutional equity strategist. "
                                "Return ONLY valid JSON with keys:\n"
                                "summary, rationale, risk_commentary, outlook."
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
            self._audit_log(ticker, signal, result, cached=False)

            return result

        except Exception:
            result = {
                "llm_enabled": True,
                "error": "llm_unavailable"
            }
            self._audit_log(ticker, signal, result, cached=False)
            return result

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
                raise ValueError

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
    # PROMPT BUILDER (HYBRID AWARE)
    ########################################################

    def _build_prompt(self, row, signal_output, technical_output, drift_stats):

        return f"""
Ticker: {row.get("ticker")}

Raw Model Score: {row.get("raw_model_score")}
Signal Agent Score: {row.get("agent_score")}
Technical Agent Score: {row.get("technical_score")}
Hybrid Consensus Score: {row.get("hybrid_consensus_score")}

Final Signal: {signal_output.get("signal")}
Confidence: {signal_output.get("confidence_numeric")}
Risk Level: {signal_output.get("risk_level")}
Volatility Regime: {signal_output.get("volatility_regime")}

Technical Bias: {technical_output.get("bias")}
Technical Strength: {technical_output.get("score")}

Drift State: {drift_stats.get("drift_state")}
Drift Severity: {drift_stats.get("severity_score")}

Warnings: {signal_output.get("warnings")}

Provide a professional institutional-style explanation suitable for an investment committee.
"""