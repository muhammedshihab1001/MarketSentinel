# =========================================================
# HYBRID AGENT EXPLANATION ROUTE v3.8
#
# Changes from v3.7:
# NEW (Issue #29): Added /agent/performance endpoints
#   - GET /agent/performance       → all agents latest metrics
#   - GET /agent/performance/{name} → single agent history
#   Both read from agent_performance table via repository.
#
# FIX (Issue #29/30): /agent/agents reads weights dynamically
#   from config/agent_weights.json instead of hardcoded values.
#
# All v3.7 fixes retained:
#   1. explainer._enabled → explainer.enabled
#   2. asyncio.to_thread(async) → await async directly
#   3. Wrong argument names fixed
#   4. Full LLM failure handling
#   5. LLM Singleton preserved between requests
#   6. confidence_numeric key fix
# =========================================================

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

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
# =========================================================

_llm_explainer_instance = None


def _get_llm_explainer():
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
# WEIGHT CONFIG HELPER (Issue #29/30)
# Reads dynamic weights from agent_weights.json
# Falls back to defaults if file missing
# =========================================================

def _load_agent_weights_config() -> dict:
    """
    Load agent weights from config/agent_weights.json.
    Returns full config dict including weights, bounds, notes.
    """
    _DEFAULTS = {
        "weights": {
            "signal_agent": 0.30,
            "technical_agent": 0.20,
            "raw_model": 0.50,
        },
        "bounds": {"min_weight": 0.10, "max_weight": 0.60},
    }
    try:
        # Try project root relative path
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "agent_weights.json"
        if not config_path.exists():
            # Docker path
            config_path = Path("/app/config/agent_weights.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Could not load agent_weights.json | error=%s", e)
    return _DEFAULTS


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
    llm_enabled = os.getenv("LLM_ENABLED", "false").lower() in ("1", "true")
    if not llm_enabled:
        return {"llm_enabled": False, "message": "LLM disabled. Set LLM_ENABLED=true to enable."}

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"llm_enabled": True, "error": "no_api_key",
                "message": "OPENAI_API_KEY not configured. Agent response complete."}

    try:
        explainer = _get_llm_explainer()
        if not explainer.enabled:
            return {"llm_enabled": False, "message": "LLM disabled by configuration."}

        result = await asyncio.wait_for(
            explainer.explain(
                signal_row=signal_row,
                signal_output=signal_agent_output,
                technical_output=technical_output,
                drift_stats={"drift_state": drift_state, "severity_score": severity_score},
            ),
            timeout=15,
        )

        if isinstance(result, dict) and "error" in result:
            error_code = result["error"]
            messages = {
                "rate_limit_exceeded": "LLM rate limit reached. Resets in 60s.",
                "llm_timeout": "LLM response timed out. Agent response is complete.",
                "llm_unavailable": "LLM service unavailable. Agent response is complete.",
            }
            return {"llm_enabled": True, "error": error_code,
                    "message": messages.get(error_code, "LLM unavailable.")}

        return result

    except asyncio.TimeoutError:
        logger.warning("LLM outer timeout — agent response unaffected")
        return {"llm_enabled": True, "error": "llm_timeout",
                "message": "LLM took too long. Agent response is complete."}
    except Exception as exc:
        logger.debug("LLM explain non-blocking failure: %s", exc)
        return {"llm_enabled": True, "error": "llm_error",
                "message": "LLM unavailable. Agent response is complete."}


# =========================================================
# GET /agent/explain?ticker=X
# =========================================================

@router.get("/explain", summary="Signal Explanation for Ticker")
@router.post("/explain", include_in_schema=False)
async def explain_signal(
    request: Request,
    ticker: str = Query(None, description="Ticker symbol (e.g. AAPL)", example="AAPL"),
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
            raise HTTPException(status_code=503,
                detail="No snapshot available. Background compute pending (~90s on first load).")

        signals = snapshot_result.get("snapshot", {}).get("signals", [])
        signal_row = next((s for s in signals if s["ticker"] == ticker), None)

        if signal_row is None:
            raise HTTPException(status_code=404,
                detail=f"{ticker} not found in snapshot. Check GET /universe.")

        signal_details = snapshot_result.get("_signal_details", {})
        agents = signal_details.get(ticker, {})
        signal_agent_output = agents.get("signal_agent", {})
        technical_output = agents.get("technical_agent", {})

        raw_score   = float(signal_row.get("raw_model_score", 0.0))
        hybrid_score = float(signal_row.get("hybrid_consensus_score", 0.0))
        weight       = float(signal_row.get("weight", 0.0))

        signal_direction = signal_agent_output.get("signals", {}).get("signal") or _derive_signal(weight)

        raw_confidence = (signal_agent_output.get("confidence_numeric")
                         or signal_agent_output.get("confidence"))
        confidence_numeric = round(float(raw_confidence), 4) if raw_confidence is not None else None

        governance_score = signal_agent_output.get("governance_score")
        if governance_score is not None:
            governance_score = int(governance_score)

        risk_level       = signal_agent_output.get("risk_level", "low")
        volatility_regime = technical_output.get("signals", {}).get("volatility_regime", "normal")
        technical_bias   = (technical_output.get("bias")
                           or technical_output.get("signals", {}).get("bias", "neutral"))

        drift_info     = snapshot_result.get("snapshot", {}).get("drift", {})
        drift_state    = drift_info.get("drift_state", "none")
        severity_score = drift_info.get("severity_score", 0)

        warnings    = signal_agent_output.get("warnings", [])
        explanation = signal_agent_output.get("explanation", "")

        llm_output = await _safe_llm_explain(
            signal_row=signal_row,
            signal_agent_output=signal_agent_output,
            technical_output=technical_output,
            drift_state=drift_state,
            severity_score=severity_score,
        )

        latency_ms = round((time.time() - start_time) * 1000, 1)

        rationale_list = snapshot_result.get("executive_summary", {}).get("top_5_rationale", [])
        rationale = next((r for r in rationale_list if r.get("ticker") == ticker), {})

        return _success({
            "ticker": ticker,
            "snapshot_date": signal_row.get("date", ""),
            "raw_model_score": round(raw_score, 6),
            "weight": round(weight, 6),
            "hybrid_consensus_score": round(hybrid_score, 6),
            "signal": signal_direction,
            "confidence_numeric": confidence_numeric,
            "governance_score": governance_score,
            "risk_level": risk_level,
            "volatility_regime": volatility_regime,
            "technical_bias": technical_bias,
            "drift_state": drift_state,
            "warnings": warnings,
            "explanation": explanation,
            "llm": llm_output,
            "rank": rationale.get("rank"),
            "agents_approved": rationale.get("agents_approved", []),
            "agents_flagged": rationale.get("agents_flagged", []),
            "selection_reason": rationale.get("selection_reason", ""),
            "agent_scores": rationale.get("agent_scores", {}),
            "in_top_5": bool(rationale),
            "latency_ms": latency_ms,
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

@router.get("/political-risk", summary="Political Risk Score for Ticker")
async def political_risk(
    request: Request,
    ticker: str = Query(..., description="Ticker symbol (e.g. AAPL)", example="AAPL"),
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
            "ticker": ticker,
            "political_risk_score": float(political.get("political_risk_score", 0.0)),
            "political_risk_label": political.get("political_risk_label", "LOW"),
            "top_events": political.get("top_events", [])[:5],
            "source": political.get("source", "gdelt"),
            "gdelt_status": political.get("gdelt_status", "unknown"),
            "served_from_cache": bool(snapshot_result),
            "latency_ms": round((time.time() - start_time) * 1000, 1),
        })

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Political risk failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /agent/agents
# FIX (Issue #29/30): reads weights from agent_weights.json
# =========================================================

@router.get("/agents", summary="Agent Pipeline Descriptions")
async def list_agents():
    # FIX: was hardcoded 0.5/0.2/0.2/0.1 — now reads from config
    cfg = _load_agent_weights_config()
    w = cfg.get("weights", {})

    return _success({
        "agents": {
            "signal_agent": {
                "name": "SignalAgent",
                "description": "Interprets XGBoost output into LONG/SHORT/NEUTRAL with confidence and risk level.",
                "weight": round(w.get("signal_agent", 0.30), 4),
            },
            "technical_risk_agent": {
                "name": "TechnicalRiskAgent",
                "description": "Evaluates momentum, EMA structure, RSI, and volatility regime.",
                "weight": round(w.get("technical_agent", 0.20), 4),
            },
            "raw_model": {
                "name": "XGBoostModel",
                "description": "Raw XGBoost regression score. Primary signal source.",
                "weight": round(w.get("raw_model", 0.50), 4),
            },
            "portfolio_decision_agent": {
                "name": "PortfolioDecisionAgent",
                "description": "Aggregates per-ticker signals into portfolio decisions with exposure control.",
                "weight": 0.0,   # orchestrator — not in hybrid score
            },
            "political_risk_agent": {
                "name": "PoliticalRiskAgent",
                "description": "Detects geopolitical risk via GDELT. CRITICAL label overrides all signals.",
                "weight": 0.0,   # overlay — not in hybrid score
            },
        },
        "last_updated": cfg.get("last_updated", "unknown"),
        "bounds": cfg.get("bounds", {}),
    })


# =========================================================
# GET /agent/performance          — Issue #29
# Returns latest evaluation metrics for all agents
# =========================================================

@router.get(
    "/performance",
    summary="Agent Performance Metrics",
    description="""
Returns the latest evaluated performance metrics for all agents.

Reads from the `agent_performance` table populated by `evaluate_agents.py`.

**Response includes per agent:**
- `direction_accuracy` — fraction of correct LONG/SHORT predictions
- `sharpe_ratio` — annualized Sharpe ratio
- `num_predictions` — number of predictions evaluated
- `avg_score` — average agent score in the period
- `confidence_calibration` — how well confidence matched accuracy (signal_agent only)
- `mean_absolute_error` — prediction error (model_only only)

**503** = No performance data yet. Run `evaluate_agents.py` first.
""",
)
async def get_agent_performance(
    days: int = Query(30, description="Lookback period in days (default: 30)"),
):
    endpoint = "/agent/performance"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        from core.db.repository import PredictionRepository

        perf_df = PredictionRepository.get_agent_performance(days=days)

        if perf_df is None or perf_df.empty:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No agent performance data found for the last {days} days. "
                    "Run: docker-compose exec api python scripts/evaluate_agents.py"
                ),
            )

        # Build per-agent summary (latest record per agent)
        latest = (
            perf_df.sort_values("evaluation_date")
            .groupby("agent_name")
            .tail(1)
        )

        agents_summary = {}
        for _, row in latest.iterrows():
            agent_name = row["agent_name"]
            agents_summary[agent_name] = {
                "direction_accuracy": row.get("direction_accuracy"),
                "sharpe_ratio": row.get("sharpe_ratio"),
                "num_predictions": int(row.get("num_predictions", 0)),
                "avg_score": row.get("avg_score"),
                "confidence_calibration": row.get("confidence_calibration"),
                "mean_absolute_error": row.get("mean_absolute_error"),
                "evaluation_date": row.get("evaluation_date"),
            }

        # Load current weights for comparison
        cfg = _load_agent_weights_config()
        current_weights = cfg.get("weights", {})

        return _success({
            "agents": agents_summary,
            "current_weights": current_weights,
            "period_days": days,
            "num_agents_evaluated": len(agents_summary),
            "latency_ms": round((time.time() - start_time) * 1000, 1),
        })

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Agent performance fetch failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /agent/performance/{agent_name}  — Issue #29
# Returns full history for a single agent
# =========================================================

@router.get(
    "/performance/{agent_name}",
    summary="Single Agent Performance History",
    description="""
Returns full evaluation history for a single agent.

**agent_name options:** `signal_agent`, `technical_agent`, `model_only`

**Response includes:** full time-series of evaluation metrics.
""",
)
async def get_single_agent_performance(
    agent_name: str,
    days: int = Query(30, description="Lookback period in days (default: 30)"),
):
    endpoint = f"/agent/performance/{agent_name}"
    API_REQUEST_COUNT.labels(endpoint="/agent/performance/{agent_name}").inc()
    start_time = time.time()

    valid_agents = {"signal_agent", "technical_agent", "model_only"}
    if agent_name not in valid_agents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent_name '{agent_name}'. Must be one of: {sorted(valid_agents)}",
        )

    try:
        from core.db.repository import PredictionRepository

        perf_df = PredictionRepository.get_agent_performance(
            agent_name=agent_name,
            days=days,
        )

        if perf_df is None or perf_df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No performance data found for agent '{agent_name}' "
                    f"in the last {days} days."
                ),
            )

        # Convert to list of records sorted by date
        records = perf_df.sort_values("evaluation_date").to_dict("records")

        # Summary stats over the period
        latest = records[-1] if records else {}

        return _success({
            "agent_name": agent_name,
            "period_days": days,
            "num_evaluations": len(records),
            "latest": {
                "direction_accuracy": latest.get("direction_accuracy"),
                "sharpe_ratio": latest.get("sharpe_ratio"),
                "num_predictions": latest.get("num_predictions"),
                "avg_score": latest.get("avg_score"),
                "evaluation_date": latest.get("evaluation_date"),
            },
            "history": records,
            "latency_ms": round((time.time() - start_time) * 1000, 1),
        })

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint="/agent/performance/{agent_name}").inc()
        logger.exception("Single agent performance fetch failed | agent=%s", agent_name)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        API_LATENCY.labels(endpoint="/agent/performance/{agent_name}").observe(
            time.time() - start_time
        )