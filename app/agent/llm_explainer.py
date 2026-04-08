"""
MarketSentinel v4.1.0

LLM Explainer — generates institutional-grade explanations for
trading signals and portfolio decisions via OpenAI.

Features:
    - Per-stock and portfolio-level explanations
    - In-memory LRU-style cache with TTL
    - Sliding-window rate limiter
    - Audit logging (hash-based, no PII)
    - Graceful degradation when LLM is disabled or unavailable
    - Separate JSON validators for stock vs portfolio responses
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from openai import AsyncOpenAI

from core.config.env_loader import get_bool, get_env, get_int

logger = logging.getLogger("marketsentinel.llm")

# ── Expected keys per response type ──────────────────────────────────────────
_STOCK_REQUIRED_KEYS:     Set[str] = {"summary", "rationale", "risk_commentary", "outlook"}
_PORTFOLIO_REQUIRED_KEYS: Set[str] = {"portfolio_summary", "macro_risk_view", "allocation_comment"}

# ── Token limits ──────────────────────────────────────────────────────────────
_STOCK_MAX_TOKENS     = 400
_PORTFOLIO_MAX_TOKENS = 600

# ── Cache batch eviction size ─────────────────────────────────────────────────
_CACHE_EVICT_BATCH = 10   # evict oldest 10 at once when cache is full


class LLMExplainer:
    """
    Generates LLM-powered explanations for trading signals and portfolio decisions.

    All public methods are async.
    Disable LLM entirely with LLM_ENABLED=0 (default).
    """

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.enabled               = get_bool("LLM_ENABLED",          False)
        self.model_name            = get_env("OPENAI_MODEL",           "gpt-4o-mini")
        self.timeout               = get_int("OPENAI_TIMEOUT",         12)
        self.rate_limit_per_minute = get_int("LLM_RATE_LIMIT_PER_MIN", 30)
        self.cache_enabled         = get_bool("LLM_CACHE_ENABLED",     True)
        self.cache_ttl_seconds     = get_int("LLM_CACHE_TTL_SEC",      180)
        self.audit_enabled         = get_bool("LLM_AUDIT_ENABLED",     True)
        self.max_cache_items       = get_int("LLM_CACHE_MAX_ITEMS",    500)

        self._request_times: List[float] = []
        self._rate_lock  = threading.Lock()

        self._cache:      Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        api_key = get_env("OPENAI_API_KEY")

        if self.enabled and not api_key:
            logger.warning(
                "LLM_ENABLED=1 but OPENAI_API_KEY is missing — disabling LLM."
            )
            self.enabled = False

        self.client: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=api_key) if self.enabled else None
        )

        logger.info(
            "LLMExplainer ready | enabled=%s | model=%s",
            self.enabled, self.model_name,
        )

    # ────────────────────────────────────────────────────────────────────────
    # RATE LIMITER  (sliding window)
    # ────────────────────────────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Returns True if a request is allowed, False if rate limit exceeded."""
        if self.rate_limit_per_minute <= 0:
            return True

        now = time.time()
        with self._rate_lock:
            self._request_times = [
                t for t in self._request_times if now - t < 60
            ]
            if len(self._request_times) >= self.rate_limit_per_minute:
                return False
            self._request_times.append(now)
        return True

    # ────────────────────────────────────────────────────────────────────────
    # IN-MEMORY CACHE  (TTL + batch eviction)
    # ────────────────────────────────────────────────────────────────────────

    def _cache_key(
        self,
        row:              Dict[str, Any],
        signal_output:    Dict[str, Any],
        technical_output: Dict[str, Any],
        stats:            Dict[str, Any],
    ) -> str:
        payload = {
            "ticker":      row.get("ticker"),
            "score":       row.get("hybrid_consensus_score"),
            "confidence":  signal_output.get("confidence_numeric"),
            "risk":        signal_output.get("risk_level"),
            "vol_regime":  signal_output.get("volatility_regime"),
            "tech_bias":   technical_output.get("bias"),
            "tech_score":  technical_output.get("score"),
            "drift_state": stats.get("drift_state"),
            "severity":    stats.get("severity_score"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def _portfolio_cache_key(self, decision_report: Dict[str, Any]) -> str:
        payload = {
            "portfolio_bias":    decision_report.get("portfolio_findings", {}).get("portfolio_bias"),
            "avg_confidence":    decision_report.get("portfolio_findings", {}).get("average_confidence"),
            "drift_state":       decision_report.get("portfolio_findings", {}).get("drift_state"),
            "concentration":     decision_report.get("portfolio_findings", {}).get("concentration_risk"),
            "snapshot_date":     decision_report.get("snapshot_date"),
        }
        return "portfolio:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
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

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        with self._cache_lock:
            # Batch eviction: remove oldest _CACHE_EVICT_BATCH items when full
            if len(self._cache) >= self.max_cache_items:
                oldest_keys = sorted(
                    self._cache, key=lambda k: self._cache[k][1]
                )[:_CACHE_EVICT_BATCH]
                for k in oldest_keys:
                    del self._cache[k]
            self._cache[key] = (value, time.time())

    # ────────────────────────────────────────────────────────────────────────
    # AUDIT LOGGING
    # ────────────────────────────────────────────────────────────────────────

    def _audit_log(
        self,
        ticker: str,
        signal: str,
        result: Dict[str, Any],
        cached: bool,
    ) -> None:
        if not self.audit_enabled:
            return
        try:
            response_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()
            record = {
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "ticker":        ticker,
                "signal":        signal,
                "model":         self.model_name,
                "cached":        cached,
                "status":        "error" if "error" in result else "success",
                "response_hash": response_hash,
            }
            logger.info("LLM_AUDIT | %s", json.dumps(record))
        except Exception as exc:
            logger.warning("Failed to write LLM audit log: %s", exc)

    # ────────────────────────────────────────────────────────────────────────
    # JSON PARSERS  (separate per response type)
    # ────────────────────────────────────────────────────────────────────────

    def _parse_stock_json(self, content: str) -> Dict[str, Any]:
        """
        Parse and validate stock-level LLM response.
        Expected keys: summary, rationale, risk_commentary, outlook
        """
        return self._parse_json(
            content,
            required_keys=_STOCK_REQUIRED_KEYS,
            fallback={
                "summary":          "Explanation unavailable.",
                "rationale":        "Model response could not be parsed.",
                "risk_commentary":  "Unavailable.",
                "outlook":          "Unavailable.",
            },
        )

    def _parse_portfolio_json(self, content: str) -> Dict[str, Any]:
        """
        Parse and validate portfolio-level LLM response.
        Expected keys: portfolio_summary, macro_risk_view, allocation_comment
        """
        return self._parse_json(
            content,
            required_keys=_PORTFOLIO_REQUIRED_KEYS,
            fallback={
                "portfolio_summary":  "Portfolio explanation unavailable.",
                "macro_risk_view":    "Unavailable.",
                "allocation_comment": "Unavailable.",
            },
        )

    @staticmethod
    def _parse_json(
        content:       str,
        required_keys: Set[str],
        fallback:      Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generic JSON parser with key validation and fallback."""
        try:
            content = content.replace("```json", "").replace("```", "").strip()
            data    = json.loads(content)
            if not required_keys.issubset(data.keys()):
                missing = required_keys - set(data.keys())
                logger.warning("LLM response missing keys: %s", missing)
                return fallback
            return data
        except json.JSONDecodeError as exc:
            logger.warning("LLM JSON parse error: %s", exc)
            return fallback
        except Exception as exc:
            logger.warning("LLM response processing error: %s", exc)
            return fallback

    # ────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDERS
    # ────────────────────────────────────────────────────────────────────────

    def _build_stock_prompt(
        self,
        row:              Dict[str, Any],
        signal_output:    Dict[str, Any],
        technical_output: Dict[str, Any],
        drift_stats:      Dict[str, Any],
    ) -> str:
        warnings = signal_output.get("warnings", [])[:5]
        return (
            f"Ticker: {row.get('ticker')}\n"
            f"Hybrid Score: {row.get('hybrid_consensus_score')}\n"
            f"Model Confidence: {signal_output.get('confidence_numeric')}\n"
            f"Signal Direction: {signal_output.get('signal')}\n"
            f"Risk Level: {signal_output.get('risk_level')}\n"
            f"Volatility Regime: {signal_output.get('volatility_regime')}\n"
            f"Technical Bias: {technical_output.get('bias')}\n"
            f"Technical Strength: {technical_output.get('score')}\n"
            f"Drift State: {drift_stats.get('drift_state')}\n"
            f"Drift Severity: {drift_stats.get('severity_score')}\n"
            f"Warnings: {warnings}\n\n"
            "Provide professional institutional commentary."
        )

    def _build_portfolio_prompt(self, decision_report: Dict[str, Any]) -> str:
        findings = json.dumps(
            decision_report.get("portfolio_findings", {}), indent=2
        )
        summary  = decision_report.get("executive_summary", "")
        return (
            f"Portfolio Findings:\n{findings}\n\n"
            f"Executive Summary:\n{summary}\n\n"
            "Provide institutional investment committee commentary.\n\n"
            "Return JSON with keys: portfolio_summary, macro_risk_view, allocation_comment."
        )

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC API — SINGLE STOCK
    # ────────────────────────────────────────────────────────────────────────

    async def explain(
        self,
        signal_row:       Dict[str, Any],
        signal_output:    Dict[str, Any],
        technical_output: Dict[str, Any],
        drift_stats:      Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate LLM explanation for a single stock signal.
        Returns structured JSON with: summary, rationale, risk_commentary, outlook.
        """
        if not self.enabled:
            return {"llm_enabled": False, "message": "LLM explanation disabled"}

        ticker = signal_row.get("ticker", "UNKNOWN")
        signal = signal_output.get("signal", "NEUTRAL")

        # ── Cache check ───────────────────────────────────────────────────────
        cache_key = self._cache_key(
            signal_row, signal_output, technical_output, drift_stats
        )
        cached = self._get_cached(cache_key)
        if cached:
            self._audit_log(ticker, signal, cached, cached=True)
            return {**cached, "cached": True}

        # ── Rate limit check ──────────────────────────────────────────────────
        if not self._check_rate_limit():
            result = {"llm_enabled": True, "error": "rate_limit_exceeded"}
            self._audit_log(ticker, signal, result, cached=False)
            return result

        # ── OpenAI call ───────────────────────────────────────────────────────
        prompt = self._build_stock_prompt(
            signal_row, signal_output, technical_output, drift_stats
        )
        start = time.time()

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role":    "system",
                            "content": (
                                "You are an institutional equity strategist. "
                                "Return ONLY valid JSON with keys: "
                                "summary, rationale, risk_commentary, outlook."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=_STOCK_MAX_TOKENS,
                ),
                timeout=self.timeout,
            )

            latency     = time.time() - start
            raw_content = response.choices[0].message.content.strip()
            parsed      = self._parse_stock_json(raw_content)

            result = {
                "llm_enabled": True,
                "model":       self.model_name,
                "latency":     round(latency, 3),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "structured":  parsed,
                "cached":      False,
            }

            self._set_cache(cache_key, result)
            self._audit_log(ticker, signal, result, cached=False)
            return result

        except asyncio.TimeoutError:
            logger.warning("LLM explain timeout for ticker=%s", ticker)
            result = {"llm_enabled": True, "error": "llm_timeout"}
            self._audit_log(ticker, signal, result, cached=False)
            return result

        except Exception as exc:
            logger.warning("LLM explain failed for ticker=%s: %s", ticker, exc)
            result = {"llm_enabled": True, "error": "llm_unavailable"}
            self._audit_log(ticker, signal, result, cached=False)
            return result

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC API — PORTFOLIO LEVEL
    # ────────────────────────────────────────────────────────────────────────

    async def explain_portfolio(
        self,
        decision_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate LLM explanation for full portfolio decision report.
        Returns structured JSON with:
            portfolio_summary, macro_risk_view, allocation_comment.
        """
        if not self.enabled:
            return {"llm_enabled": False, "message": "LLM explanation disabled"}

        # ── Cache check (portfolio has its own key) ───────────────────────────
        cache_key = self._portfolio_cache_key(decision_report)
        cached    = self._get_cached(cache_key)
        if cached:
            logger.debug("Portfolio LLM cache hit.")
            return {**cached, "cached": True}

        # ── Rate limit check ──────────────────────────────────────────────────
        if not self._check_rate_limit():
            return {"llm_enabled": True, "error": "rate_limit_exceeded"}

        # ── OpenAI call ───────────────────────────────────────────────────────
        prompt = self._build_portfolio_prompt(decision_report)
        start  = time.time()

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role":    "system",
                            "content": "Return valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=_PORTFOLIO_MAX_TOKENS,
                ),
                timeout=self.timeout,
            )

            latency     = time.time() - start
            raw_content = response.choices[0].message.content.strip()
            parsed      = self._parse_portfolio_json(raw_content)   # ← correct parser

            result = {
                "llm_enabled": True,
                "model":       self.model_name,
                "latency":     round(latency, 3),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "structured":  parsed,
                "cached":      False,
            }

            self._set_cache(cache_key, result)
            return result

        except asyncio.TimeoutError:
            logger.warning("LLM explain_portfolio timeout.")
            return {"llm_enabled": True, "error": "portfolio_llm_timeout"}

        except Exception as exc:
            logger.warning("LLM explain_portfolio failed: %s", exc)
            return {"llm_enabled": True, "error": "portfolio_llm_unavailable"}
