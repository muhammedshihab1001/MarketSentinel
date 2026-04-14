# =========================================================
# HYBRID AGENT EXPLANATION ROUTE v3.7
# FIXES:
#   1. explainer._enabled → explainer.enabled
#   2. asyncio.to_thread(async) → await async directly
#   3. Wrong argument names fixed
#   4. Full LLM failure handling — agent always responds
#      even if LLM times out, rate limit hit, API expired,
#      or any other LLM error occurs.
#   5. LLM Singleton — cache preserved between requests
#      (was creating new instance per request, killing cache)
#   6. confidence_numeric key fix — tries confidence_numeric
#      first, falls back to confidence (covers both schemas)
# =========================================================

import asyncio
import time
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.agent")

router = APIRouter(prefix="/agent", tags=["agent"])

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"

# =========================================================
# LLM SINGLETON — preserves in-memory cache between requests
# Without this: new LLMExplainer() per request = cache lost
# =========================================================

_llm_explainer_instance = None


def _get_llm_explainer():
    """
    Return singleton LLMExplainer instance.
    Cache inside LLMExplainer is preserved between requests.
    Same ticker → second call returns from cache instantly.
    """
    global _llm_explainer_instance
    if _llm_explainer_instance is None:
        from app.agent.llm_explainer import LLMExplainer
        _llm_explainer_instance = LLMExplainer()
        logger.info("LLM singleton initialised | model=%s", _llm_explainer_instance.model_name)
    return _llm_explainer_instance


def _ts():
    return datetime.now(timezone.utc).isoformat()


def _success(data):
    return {"success": True, "data": data, "error": None, "timestamp": _ts()}


def _derive_signal(weight: float) -> str:
    if weight > 0.01:
        return "LONG"
    if weight < -0.01:
        return "SHORT"
    return "NEUTRAL"


def _get_cache(request: Request):
    try:
        return request.app.state.cache
    except AttributeError:
        from app.inference.cache import RedisCache
        return RedisCache()


# =========================================================
# LLM HELPER — always returns something, never raises
# =========================================================

async def _safe_llm_explain(
    signal_row: dict,
    signal_agent_output: dict,
    technical_output: dict,
    drift_state: str,
    severity_score: int,
) -> dict:
    """
    Calls LLM explain with full failure isolation.

    Behavior by scenario:
      LLM disabled          → {llm_enabled: false, message: ...}
      No API key            → {llm_enabled: true, error: no_api_key}
      API key expired       → {llm_enabled: true, error: llm_error}
      Rate limit exceeded   → {llm_enabled: true, error: rate_limit_exceeded}
      Timeout (any)         → {llm_enabled: true, error: llm_timeout}
      Network error         → {llm_enabled: true, error: llm_error}
      Parse error           → {llm_enabled: true, error: llm_error}
      Success               → full LLM structured response

    Agent endpoint ALWAYS returns complete signal data.
    LLM is additive — its failure never affects core response.
    """

    # ── LLM disabled ──────────────────────────────────────
    llm_enabled = os.getenv("LLM_ENABLED", "false").lower() in ("1", "true")
    if not llm_enabled:
        return {
            "llm_enabled": False,
            "message": "LLM disabled. Set LLM_ENABLED=true to enable."
        }

    # ── API key missing ───────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "llm_enabled": True,
            "error": "no_api_key",
            "message": "OPENAI_API_KEY not configured. Agent response complete."
        }

    # ── Full LLM call with isolation ──────────────────────
    try:
        # FIX 5: Use singleton — cache preserved between requests
        explainer = _get_llm_explainer()

        # FIX 1: correct attribute name (no underscore prefix)
        if not explainer.enabled:
            return {
                "llm_enabled": False,
                "message": "LLM disabled by configuration."
            }

        # FIX 2: await directly — explain() is async, not sync
        # FIX 3: correct argument names matching llm_explainer.py signature
        result = await asyncio.wait_for(
            explainer.explain(
                signal_row=signal_row,
                signal_output=signal_agent_output,
                technical_output=technical_output,
                drift_stats={
                    "drift_state": drift_state,
                    "severity_score": severity_score,
                },
            ),
            timeout=15,  # outer safety net (LLM internal = 12s)
        )

        # Handle error dicts returned by LLMExplainer itself
        if isinstance(result, dict) and "error" in result:
            error_code = result["error"]
            messages = {
                "rate_limit_exceeded": (
                    "LLM rate limit reached. "
                    "Resets in 60s. Agent response is complete."
                ),
                "llm_timeout": (
                    "LLM response timed out. "
                    "Agent response is complete."
                ),
                "llm_unavailable": (
                    "LLM service unavailable. "
                    "Agent response is complete."
                ),
            }
            return {
                "llm_enabled": True,
                "error": error_code,
                "message": messages.get(
                    error_code,
                    "LLM unavailable. Agent response is complete."
                )
            }

        return result

    except asyncio.TimeoutError:
        logger.warning("LLM outer timeout — agent response unaffected")
        return {
            "llm_enabled": True,
            "error": "llm_timeout",
            "message": "LLM took too long. Agent response is complete."
        }

    except Exception as exc:
        logger.debug("LLM explain non-blocking failure: %s", exc)
        return {
            "llm_enabled": True,
            "error": "llm_error",
            "message": "LLM unavailable. Agent response is complete."
        }


# =========================================================
# GET /agent/explain?ticker=X
# =========================================================

@router.get(
    "/explain",
    summary="Signal Explanation for Ticker",
    description="""
Returns signal explanation for a specific ticker from the background snapshot cache.

**ticker** (required): Any ticker in the universe. See GET /universe for the full list.

**Response includes:**
- `signal`: LONG / SHORT / NEUTRAL
- `raw_model_score`: raw XGBoost output
- `hybrid_consensus_score`: agent-weighted consensus
- `confidence_numeric`: signal confidence (0-1)
- `risk_level`: low / medium / high
- `volatility_regime`: normal / high_volatility / low_volatility
- `technical_bias`: bullish / bearish / neutral
- `warnings`: list of risk flags from the signal agent
- `explanation`: natural language explanation
- `llm`: LLM rationale (null if disabled, error message if LLM fails)

**LLM failure guarantee:** Signal data, confidence, risk level and all
agent fields are ALWAYS returned regardless of LLM status.

**503** = snapshot not yet computed. Wait ~90s and retry.
""",
    response_description="Signal explanation from background snapshot cache.",
)
@router.post("/explain", include_in_schema=False)
async def explain_signal(
    request: Request,
    ticker: str = Query(
        None,
        description="Ticker symbol (e.g. AAPL, NVDA, MSFT)",
        example="AAPL",
    ),
):
    endpoint = "/agent/explain"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    if ticker is None:
        try:
            body = await request.json()
            ticker = body.get("ticker")
        except Exception:
            pass

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker parameter is required")

    ticker = ticker.upper().strip()

    try:
        cache = _get_cache(request)
        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)

        if not snapshot_result:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No snapshot available. "
                    "Background compute pending (~90s on first load)."
                ),
            )

        signals = snapshot_result.get("snapshot", {}).get("signals", [])
        signal_row = next((s for s in signals if s["ticker"] == ticker), None)

        if signal_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"{ticker} not found in snapshot. Check GET /universe.",
            )

        signal_details = snapshot_result.get("_signal_details", {})
        agents = signal_details.get(ticker, {})

        signal_agent_output = agents.get("signal_agent", {})
        technical_output    = agents.get("technical_agent", {})

        raw_score    = float(signal_row.get("raw_model_score", 0.0))
        hybrid_score = float(signal_row.get("hybrid_consensus_score", 0.0))
        weight       = float(signal_row.get("weight", 0.0))

        signal_direction = (
            signal_agent_output.get("signals", {}).get("signal")
            or _derive_signal(weight)
        )

        # FIX 6: Try confidence_numeric first, fall back to confidence
        # Signal agent may store under either key depending on version
        raw_confidence = (
            signal_agent_output.get("confidence_numeric")
            or signal_agent_output.get("confidence")
        )
        confidence_numeric = round(float(raw_confidence), 4) if raw_confidence is not None else None

        governance_score = signal_agent_output.get("governance_score")
        if governance_score is not None:
            governance_score = int(governance_score)

        risk_level        = signal_agent_output.get("risk_level", "low")
        volatility_regime = (
            technical_output.get("signals", {}).get("volatility_regime", "normal")
        )
        technical_bias = (
            technical_output.get("bias")
            or technical_output.get("signals", {}).get("bias", "neutral")
        )

        drift_info     = snapshot_result.get("snapshot", {}).get("drift", {})
        drift_state    = drift_info.get("drift_state", "none")
        severity_score = drift_info.get("severity_score", 0)

        warnings    = signal_agent_output.get("warnings", [])
        explanation = signal_agent_output.get("explanation", "")

        # ── LLM — fully isolated, never affects response ───────────────────
        # Scenarios handled:
        #   disabled        → {llm_enabled: false}
        #   no key          → {error: no_api_key}
        #   expired key     → {error: llm_error}
        #   rate limited    → {error: rate_limit_exceeded}
        #   timeout         → {error: llm_timeout}
        #   network error   → {error: llm_error}
        #   success         → full structured LLM response
        llm_output = await _safe_llm_explain(
            signal_row=signal_row,
            signal_agent_output=signal_agent_output,
            technical_output=technical_output,
            drift_state=drift_state,
            severity_score=severity_score,
        )

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # Top-5 rationale lookup
        rationale_list = (
            snapshot_result.get("executive_summary", {})
            .get("top_5_rationale", [])
        )
        rationale = next(
            (r for r in rationale_list if r.get("ticker") == ticker), {}
        )

        return _success({
            "ticker":                 ticker,
            "snapshot_date":          signal_row.get("date", ""),
            "raw_model_score":        round(raw_score, 6),
            "weight":                 round(weight, 6),
            "hybrid_consensus_score": round(hybrid_score, 6),
            "signal":                 signal_direction,
            "confidence_numeric":     confidence_numeric,
            "governance_score":       governance_score,
            "risk_level":             risk_level,
            "volatility_regime":      volatility_regime,
            "technical_bias":         technical_bias,
            "drift_state":            drift_state,
            "warnings":               warnings,
            "explanation":            explanation,
            "llm":                    llm_output,
            # Rationale — populated for top-5 only
            "rank":             rationale.get("rank"),
            "agents_approved":  rationale.get("agents_approved", []),
            "agents_flagged":   rationale.get("agents_flagged", []),
            "selection_reason": rationale.get("selection_reason", ""),
            "agent_scores":     rationale.get("agent_scores", {}),
            "in_top_5":         bool(rationale),
            "latency_ms":       latency_ms,
        })

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Agent explain failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /agent/political-risk?ticker=X
# =========================================================

@router.get(
    "/political-risk",
    summary="Political Risk Score for Ticker",
    description="""
Returns geopolitical risk score for a ticker's country (US default).

Reads from snapshot cache first. Falls back to live GDELT query.

**Requires:** Owner or Demo authentication (counts against `agent` quota).
""",
    response_description="Political risk score and top geopolitical events.",
)
async def political_risk(
    request: Request,
    ticker: str = Query(
        ...,
        description="Ticker symbol (e.g. AAPL).",
        example="AAPL",
    ),
):
    endpoint = "/agent/political-risk"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    ticker = ticker.upper().strip()

    try:
        cache = _get_cache(request)
        snapshot_result = cache.get(BACKGROUND_SNAPSHOT_KEY)
        political = {}

        if snapshot_result:
            political = snapshot_result.get("_political", {})

        if not political:
            from core.agent.political_risk_agent import PoliticalRiskAgent
            agent = PoliticalRiskAgent()
            political = agent.get_political_risk(ticker, country="US")

        return _success({
            "ticker":               ticker,
            "political_risk_score": float(political.get("political_risk_score", 0.0)),
            "political_risk_label": political.get("political_risk_label", "LOW"),
            "top_events":           political.get("top_events", [])[:5],
            "source":               political.get("source", "gdelt"),
            "gdelt_status":         political.get("gdelt_status", "unknown"),
            "served_from_cache":    bool(snapshot_result),
            "latency_ms":           round((time.time() - start_time) * 1000, 1),
        })

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Political risk failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /agent/agents
# =========================================================

@router.get(
    "/agents",
    summary="Agent Pipeline Descriptions",
    description="Returns static descriptions of all agents. No auth required.",
    response_description="Agent descriptions with weights.",
)
async def list_agents():
    return _success({
        "agents": {
            "signal_agent": {
                "name":        "SignalAgent",
                "description": "Interprets XGBoost output into LONG/SHORT/NEUTRAL with confidence and risk level.",
                "weight":      0.5,
            },
            "technical_risk_agent": {
                "name":        "TechnicalRiskAgent",
                "description": "Evaluates momentum, EMA structure, RSI, and volatility regime.",
                "weight":      0.2,
            },
            "portfolio_decision_agent": {
                "name":        "PortfolioDecisionAgent",
                "description": "Aggregates per-ticker signals into portfolio decisions with exposure control.",
                "weight":      0.2,
            },
            "political_risk_agent": {
                "name":        "PoliticalRiskAgent",
                "description": "Detects geopolitical risk via GDELT. CRITICAL label overrides all signals.",
                "weight":      0.1,
            },
        }
    })
