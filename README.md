# MarketSentinel

**Institutional-Grade ML Trading Signal Platform**

[![CI](https://github.com/muhammedshihab1001/MarketSentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammedshihab1001/MarketSentinel/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## What Is MarketSentinel

MarketSentinel is a production-ready machine learning platform that transforms raw market price data into **risk-aware trading signals with full explainability**. It is built for reliability, auditability, and operational safety — the same principles used inside quantitative trading firms.

The system does not just predict. It **governs** predictions through a multi-agent decision pipeline, drift enforcement, political risk overlay, and per-request quota management — ensuring every signal issued is traceable, explainable, and safe.

---

## Live System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MarketSentinel Platform                       │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Market   │    │ Feature  │    │ XGBoost  │    │ Agent    │  │
│  │ Data     │───▶│ Pipeline │───▶│ Inference│───▶│ Pipeline │  │
│  │ (OHLCV)  │    │ (64 feat)│    │ (v5.9)   │    │ (4 agents│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                        │         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │         │
│  │ Redis    │◀───│ Snapshot │◀───│ Portfolio│◀────────┘         │
│  │ Cache    │    │ Engine   │    │ Decision │                    │
│  └──────────┘    └──────────┘    └──────────┘                   │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FastAPI REST API (v3.7)                     │   │
│  │  Auth · Snapshot · Portfolio · Drift · Performance       │   │
│  │  Agent Explain · Political Risk · Equity · Model Info    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

| Capability | Detail |
|---|---|
| **Universe** | 100 S&P 500 large-cap tickers |
| **Model** | XGBoost trained on 64 engineered features |
| **Signals** | LONG / SHORT / NEUTRAL with confidence scores |
| **Agents** | 4-agent decision pipeline with approval/flagging |
| **Drift** | Real-time model drift detection (0–15 severity scale) |
| **Political Risk** | GDELT + NewsAPI 6-provider fallback chain |
| **Demo System** | Per-feature quota (10 requests/week) with Redis tracking |
| **Security** | JWT cookies + API key + per-endpoint rate limiting |
| **Observability** | Prometheus metrics + structured logging |

---

## System Architecture

### Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE v5.9                      │
│                                                                       │
│  PostgreSQL        Feature Engineering      XGBoost Model            │
│  ┌─────────┐       ┌─────────────────┐     ┌──────────────┐         │
│  │ OHLCV   │──────▶│ 64 Features     │────▶│ raw_score[]  │         │
│  │ Daily   │       │ - momentum      │     │ per ticker   │         │
│  │ 100 tkr │       │ - RSI / EMA     │     └──────┬───────┘         │
│  └─────────┘       │ - volatility    │            │                  │
│                    │ - regime        │            ▼                  │
│                    │ - reversal      │     ┌──────────────┐         │
│                    └─────────────────┘     │ Per-Ticker   │         │
│                                            │ Agent Loop   │         │
│                                            └──────┬───────┘         │
│                                                   │                  │
│              ┌────────────────────────────────────┤                  │
│              │                                    │                  │
│              ▼                ▼                   ▼                  │
│  ┌──────────────┐  ┌──────────────┐   ┌──────────────────┐         │
│  │ SignalAgent  │  │TechnicalRisk │   │ PoliticalRisk    │         │
│  │ (weight 0.5) │  │Agent (0.2)   │   │ Agent (0.1)      │         │
│  │              │  │              │   │ GDELT + NewsAPI  │         │
│  │ - confidence │  │ - momentum_z │   │ - risk_score     │         │
│  │ - risk_level │  │ - EMA ratio  │   │ - label          │         │
│  │ - governance │  │ - RSI        │   │ - top_events     │         │
│  └──────┬───────┘  └──────┬───────┘   └────────┬─────────┘         │
│         │                 │                     │                    │
│         └─────────────────┴─────────────────────┘                   │
│                                   │                                  │
│                                   ▼                                  │
│                    ┌──────────────────────────┐                     │
│                    │ Hybrid Consensus Score   │                     │
│                    │ 0.5×raw + 0.3×signal     │                     │
│                    │ + 0.2×technical          │                     │
│                    │ × political_overlay      │                     │
│                    │ × exposure_scale (drift) │                     │
│                    └──────────────┬───────────┘                     │
│                                   │                                  │
│                                   ▼                                  │
│                    ┌──────────────────────────┐                     │
│                    │ Portfolio Decision Agent │                     │
│                    │ (weight 0.2)             │                     │
│                    │ - position sizing        │                     │
│                    │ - exposure control       │                     │
│                    │ - top-5 rationale        │                     │
│                    └──────────────┬───────────┘                     │
│                                   │                                  │
│                                   ▼                                  │
│                    ┌──────────────────────────┐                     │
│                    │     Redis Cache          │                     │
│                    │  ms:background_snapshot  │                     │
│                    │  TTL: 360 seconds        │                     │
│                    └──────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Auth & Security Flow

```
┌──────────────────────────────────────────────────────────┐
│                  Request Lifecycle                        │
│                                                          │
│  Incoming Request                                        │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────────────────────┐                    │
│  │  main.py — request_middleware   │                    │
│  │                                 │                    │
│  │  Is path in PUBLIC_PATHS?       │                    │
│  │  YES → pass through             │                    │
│  │  NO  → check JWT or API key     │                    │
│  │         No auth → 401           │                    │
│  └──────────────┬──────────────────┘                    │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │  Per-Endpoint Rate Limit Check  │                    │
│  │  Redis key: ms:ratelimit:{ip}   │                    │
│  │  Exceeded → 429 + Retry-After   │                    │
│  └──────────────┬──────────────────┘                    │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │  AuthMiddleware v2.4            │                    │
│  │                                 │                    │
│  │  Valid X-API-KEY? → owner role  │                    │
│  │  Valid JWT owner? → full access │                    │
│  │  Valid JWT demo?  → quota check │                    │
│  │  No token?        → 401         │                    │
│  └──────────────┬──────────────────┘                    │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │  Demo Quota (DemoTracker)       │                    │
│  │  Redis key:                     │                    │
│  │  demo:usage:{fingerprint}:{feat}│                    │
│  │                                 │                    │
│  │  Locked? → 200 demo_locked:true │                    │
│  │  OK?     → increment + proceed  │                    │
│  └──────────────┬──────────────────┘                    │
│                 │                                        │
│                 ▼                                        │
│            Route Handler                                 │
└──────────────────────────────────────────────────────────┘
```

### Drift Detection Flow

```
┌──────────────────────────────────────────────────────────┐
│                  Drift Detection System                   │
│                                                          │
│  Every Snapshot Run                                      │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────┐     ┌─────────────────────────┐   │
│  │ Current Feature │     │ Baseline Feature         │   │
│  │ Distribution    │────▶│ Distribution (training)  │   │
│  │ (100 tickers)   │     │ (stored in artifacts/)   │   │
│  └─────────────────┘     └──────────────┬───────────┘   │
│                                          │               │
│                                          ▼               │
│                           ┌─────────────────────────┐   │
│                           │ Statistical Tests        │   │
│                           │ - KS test per feature    │   │
│                           │ - PSI calculation        │   │
│                           │ - Schema validation      │   │
│                           └──────────────┬───────────┘   │
│                                          │               │
│                                          ▼               │
│                           ┌─────────────────────────┐   │
│                           │ Severity Score (0-15)    │   │
│                           │                          │   │
│                           │  0-4:  none (normal)     │   │
│                           │  5-9:  soft (reduce pos) │   │
│                           │ 10-15: hard (block trade)│   │
│                           └──────────────┬───────────┘   │
│                                          │               │
│                                          ▼               │
│                           ┌─────────────────────────┐   │
│                           │ exposure_scale           │   │
│                           │ none → 1.0 (full size)  │   │
│                           │ soft → 0.6 (60% of pos) │   │
│                           │ hard → 0.2 (20% of pos) │   │
│                           └─────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## API Reference

### Authentication

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/auth/owner-login` | POST | Public | Login as owner with password |
| `/auth/demo-login` | POST | Public | One-click demo access |
| `/auth/me` | GET | Public | Get current session + usage quota |
| `/auth/logout` | POST | Public | Clear session cookie |

**Owner login request:**
```json
POST /auth/owner-login
{ "username": "shihab", "password": "your_password" }
```

**Demo login response:**
```json
{
  "authenticated": true,
  "role": "demo",
  "usage": {
    "features": {
      "snapshot":    { "used": 0, "limit": 10, "remaining": 10, "locked": false },
      "portfolio":   { "used": 2, "limit": 10, "remaining": 8,  "locked": false },
      "drift":       { "used": 0, "limit": 10, "remaining": 10, "locked": false },
      "performance": { "used": 0, "limit": 10, "remaining": 10, "locked": false },
      "agent":       { "used": 0, "limit": 10, "remaining": 10, "locked": false },
      "signals":     { "used": 0, "limit": 10, "remaining": 10, "locked": false }
    },
    "fully_locked": false,
    "reset_in_seconds": 604800,
    "limit_per_feature": 10
  }
}
```

---

### Snapshot (Core Signal Engine)

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/snapshot` | POST | Owner/Demo | Full inference — 100 signals + top-5 rationale |

**Response structure:**
```json
{
  "meta": {
    "model_version": "xgb_20260407_215046",
    "drift_state": "soft",
    "long_signals": 66,
    "short_signals": 34,
    "avg_hybrid_score": 0.039,
    "latency_ms": 15894
  },
  "executive_summary": {
    "top_5_tickers": ["MU", "SBUX", "UNH", "QCOM", "AMZN"],
    "top_5_rationale": [
      {
        "rank": 1,
        "ticker": "MU",
        "signal": "LONG",
        "hybrid_score": 0.3971,
        "raw_model_score": 1.4201,
        "confidence": 0.53,
        "risk_level": "low",
        "governance_score": 26,
        "agents_approved": ["TechnicalRiskAgent"],
        "agents_flagged": ["SignalAgent (score=0.00)", "PoliticalRiskAgent (HIGH)"],
        "selection_reason": "MU ranked #1 with hybrid consensus score 0.3971..."
      }
    ]
  },
  "snapshot": {
    "drift": { "severity_score": 7, "drift_state": "soft", "exposure_scale": 0.6 },
    "signals": [
      { "ticker": "MU", "raw_model_score": 1.4201, "hybrid_consensus_score": 0.3971, "weight": 0.2382 }
    ]
  }
}
```

---

### Agent Explain

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/agent/explain?ticker=X` | GET | Owner/Demo | Full signal explanation for any ticker |
| `/agent/political-risk?ticker=X` | GET | Owner/Demo | Geopolitical risk score |
| `/agent/agents` | GET | Public | Agent pipeline descriptions |

**Explain response (top-5 ticker):**
```json
{
  "data": {
    "ticker": "MU",
    "signal": "LONG",
    "raw_model_score": 1.4201,
    "hybrid_consensus_score": 0.3971,
    "confidence_numeric": null,
    "governance_score": 26,
    "risk_level": "low",
    "volatility_regime": "normal",
    "technical_bias": "bearish",
    "drift_state": "soft",
    "explanation": "NEUTRAL | score=1.42 | conf=0.53 | tech=0.00 | risk=low",
    "rank": 1,
    "in_top_5": true,
    "agents_approved": ["TechnicalRiskAgent"],
    "agents_flagged": ["SignalAgent (score=0.00, risk=low)"],
    "selection_reason": "MU ranked #1 with hybrid consensus score 0.3971...",
    "agent_scores": {
      "signal_agent": 0.0,
      "technical_agent": 0.4204,
      "raw_model": 1.4201
    }
  }
}
```

---

### Portfolio, Drift, Performance

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/portfolio` | GET | Owner/Demo | 100 positions with weights and signals |
| `/drift` | GET | Owner/Demo | Model drift metrics (severity 0–15) |
| `/performance?days=252` | GET | Owner/Demo | Institutional strategy metrics |
| `/equity/{ticker}` | GET | Owner/Demo | Latest OHLCV + returns |
| `/equity/{ticker}/history?days=90` | GET | Owner/Demo | Historical price data |

**Performance response (key metrics):**
```json
{
  "metrics": {
    "sharpe_ratio": 2.085,
    "sortino_ratio": 3.110,
    "calmar_ratio": 3.544,
    "cumulative_return": 0.177,
    "max_drawdown": -0.061,
    "hit_rate": 0.564,
    "annual_return": 0.215,
    "annual_volatility": 0.096
  }
}
```

---

### Model & Health

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/model/info` | GET | Public | Model version, hashes, feature count |
| `/model/feature-importance` | GET | Owner/Demo | Top 62 features by XGBoost gain |
| `/model/ic-stats?days=30` | GET | Owner | IC statistics (Spearman correlation) |
| `/health/live` | GET | Public | Liveness probe |
| `/health/ready` | GET | Public | Readiness: DB + Redis + model loaded |
| `/health/db` | GET | Public | Database connectivity check |
| `/health/model` | GET | Public | Model artifact integrity check |
| `/universe` | GET | Public | 100-ticker universe list |

---

## Rate Limits

| Endpoint | Limit | Window |
|---|---|---|
| `/auth/owner-login` | 5 requests | 60 seconds |
| `/auth/demo-login` | 10 requests | 60 seconds |
| `/snapshot` | 10 requests | 60 seconds |
| `/agent/explain` | 20 requests | 60 seconds |
| `/agent/political-risk` | 20 requests | 60 seconds |
| `/performance` | 20 requests | 60 seconds |
| `/health/live` | 60 requests | 60 seconds |
| All other paths | 60 requests | 60 seconds |

Rate limits are enforced per IP via Redis. When Redis is unavailable, the system **fails open** — no requests are blocked during Redis downtime.

---

## Demo Quota System

Demo users receive **10 free requests per feature group per week**. Quotas are tracked in Redis per browser fingerprint (IP + User-Agent hash).

```
Feature Groups:
  snapshot    → POST /snapshot, POST /predict/live-snapshot
  portfolio   → GET /portfolio
  drift       → GET /drift
  performance → GET /performance
  agent       → GET /agent/explain, GET /agent/political-risk
  signals     → GET /equity/*, GET /model/feature-importance

When limit is reached:
  Backend returns HTTP 200 with body:
  { "demo_locked": true, "feature": "portfolio", "reset_in_seconds": 604800 }

Redis keys:
  demo:usage:{fingerprint}:{feature}   → request count (TTL: 7 days)
  demo:reg:{fingerprint}               → registration timestamp
```

---

## Feature Engineering Pipeline

The model uses 64 engineered features derived from raw OHLCV data:

```
Price Features          Momentum Features       Volatility Features
─────────────────       ─────────────────       ─────────────────
close_rank              momentum_5              volatility_20
return_1d               momentum_20             atr_14
return_5d               momentum_20_z           regime_feature
return_20d              reversal_5_rank         vol_ratio
log_return              rsi_14                  high_low_range

Moving Averages         EMA Structure           Cross-Sectional
─────────────────       ─────────────────       ─────────────────
ema_20                  ema_ratio               price_rank
ema_50                  trend_strength          volume_rank
sma_20                  ema_slope               return_rank
ema_20_slope            price_above_ema         momentum_rank
                        golden_cross            relative_strength
```

All features pass through schema validation before inference. The schema signature (SHA256) is embedded in every model artifact to detect silent feature drift.

---

## Agent Pipeline

```
┌───────────────────────────────────────────────────────────────┐
│                    4-Agent Decision Pipeline                   │
│                                                               │
│  Input: raw XGBoost score per ticker                         │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Agent 1: SignalAgent (weight: 0.5)                     │ │
│  │  ─────────────────────────────────                      │ │
│  │  Interprets XGBoost output into LONG/SHORT/NEUTRAL      │ │
│  │  Computes: confidence, governance_score, risk_level     │ │
│  │  Applies: drift penalty, political risk override        │ │
│  │  Output: signal direction + position size hint          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Agent 2: TechnicalRiskAgent (weight: 0.2)              │ │
│  │  ─────────────────────────────────                      │ │
│  │  Evaluates: RSI, EMA structure, momentum, volatility    │ │
│  │  Output: technical_bias, volatility_regime, tech_score  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Agent 3: PoliticalRiskAgent (weight: 0.1)              │ │
│  │  ─────────────────────────────────                      │ │
│  │  Sources: GDELT → NewsAPI → GNews (6-provider chain)    │ │
│  │  Output: political_risk_score (0-1), label, top_events  │ │
│  │  HIGH label: hybrid_score × 0.5                         │ │
│  │  CRITICAL label: signal forced to NEUTRAL               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Agent 4: PortfolioDecisionAgent (weight: 0.2)          │ │
│  │  ─────────────────────────────────                      │ │
│  │  Aggregates all signals into portfolio decisions        │ │
│  │  Enforces: gross exposure ≤ 1.0, net exposure control   │ │
│  │  Produces: top-5 rationale with agent approval status   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Hybrid Score Formula:                                        │
│  hybrid = (0.5 × raw_model + 0.3 × signal_score             │
│             + 0.2 × technical_score)                         │
│           × political_overlay                                 │
│           × exposure_scale (from drift)                      │
└───────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
MarketSentinel/
│
├── app/                         ← FastAPI inference API
│   ├── api/
│   │   ├── routes/
│   │   │   ├── agent.py         ← /agent/* endpoints (explain, political-risk)
│   │   │   ├── auth.py          ← /auth/* endpoints (login, logout, me)
│   │   │   ├── drift.py         ← /drift endpoint
│   │   │   ├── equity.py        ← /equity/* endpoints
│   │   │   ├── health.py        ← /health/* probes
│   │   │   ├── model_info.py    ← /model/* endpoints
│   │   │   ├── performance.py   ← /performance endpoint
│   │   │   ├── portfolio.py     ← /portfolio endpoint
│   │   │   ├── predict.py       ← /predict/* inference endpoints
│   │   │   └── universe.py      ← /universe endpoint
│   │   └── schemas.py           ← Pydantic request/response models
│   │
│   ├── agent/
│   │   └── llm_explainer.py     ← LLM-powered explanation (optional)
│   │
│   ├── core/
│   │   └── auth/
│   │       ├── demo_tracker.py  ← Redis-based quota tracking
│   │       ├── jwt_handler.py   ← JWT encode/decode
│   │       └── middleware.py    ← Auth middleware v2.4
│   │
│   ├── inference/
│   │   ├── cache.py             ← Redis cache with memory fallback
│   │   ├── model_loader.py      ← Thread-safe model loading singleton
│   │   └── pipeline.py          ← Inference pipeline v5.9
│   │
│   ├── monitoring/
│   │   └── metrics.py           ← Prometheus counters and histograms
│   │
│   └── main.py                  ← FastAPI app entrypoint v3.7
│
├── core/                        ← Business domain (model-agnostic)
│   ├── agent/
│   │   ├── base_agent.py        ← Abstract agent interface
│   │   ├── signal_agent.py      ← XGBoost signal interpreter
│   │   ├── technical_risk_agent.py ← Technical indicator evaluator
│   │   ├── portfolio_decision_agent.py ← Portfolio-level decisions
│   │   └── political_risk_agent.py ← GDELT geopolitical risk
│   │
│   ├── analytics/
│   │   └── performance_engine.py ← Sharpe, Sortino, Calmar, IC
│   │
│   ├── artifacts/
│   │   ├── metadata_manager.py  ← Dataset fingerprinting
│   │   └── model_registry.py    ← Artifact versioning and governance
│   │
│   ├── data/
│   │   ├── data_fetcher.py      ← Historical OHLCV retrieval
│   │   ├── data_sync.py         ← Automated daily sync
│   │   ├── market_data_service.py ← Batch price data orchestrator
│   │   └── providers/
│   │       └── market/
│   │           ├── yahoo_provider.py
│   │           ├── twelvedata_provider.py
│   │           ├── base.py
│   │           └── router.py
│   │
│   ├── db/
│   │   ├── engine.py            ← SQLAlchemy engine + pool
│   │   ├── models.py            ← ORM models (OHLCV, Features, Predictions)
│   │   └── repository.py        ← Data access layer (OHLCV, Features, Predictions)
│   │
│   ├── features/
│   │   └── feature_engineering.py ← 64-feature canonical pipeline
│   │
│   ├── indicators/
│   │   └── technical_indicators.py ← RSI, EMA, ATR, momentum
│   │
│   ├── market/
│   │   └── universe.py          ← 100-ticker S&P 500 universe
│   │
│   ├── models/
│   │   └── xgboost.py           ← XGBoost model wrapper
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py    ← KS test + PSI drift scoring
│   │   ├── market_regime_detector.py
│   │   └── retrain_trigger.py
│   │
│   ├── schema/
│   │   └── feature_schema.py    ← 64-feature contract + SHA256 signature
│   │
│   └── time/
│       └── market_time.py       ← Market calendar utilities
│
├── training/                    ← Offline model training
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   ├── regime.py
│   │   └── walk_forward.py      ← Rolling window validation
│   │
│   ├── pipelines/
│   │   └── train_pipeline.py    ← End-to-end training orchestrator
│   │
│   ├── train_xgboost.py         ← XGBoost trainer with promotion gates
│   ├── evaluate.py              ← Model quality metrics
│   └── run_evaluation.py        ← Promotion gate executor
│
├── scripts/
│   ├── generate_api_key.py      ← Cryptographically secure API key generator
│   └── generate_owner_hash.py   ← bcrypt password hash generator
│
├── tests/                       ← 115 passing tests
│   ├── conftest.py
│   ├── test_api_signal_explanation.py
│   ├── test_auth.py
│   ├── test_demo_tracker.py
│   ├── test_drift_detector.py
│   ├── test_feature_engineering.py
│   ├── test_middleware.py
│   ├── test_pipeline_filter.py
│   ├── test_redis_cache.py
│   ├── test_signal_agent.py
│   ├── test_technical_indicators.py
│   ├── test_training_end_to_end.py
│   ├── test_walk_forward.py
│   └── test_xgboost_regressor.py
│
├── artifacts/
│   └── xgboost/
│       ├── model_xgb_*.pkl      ← Versioned model artifacts
│       └── latest.json          ← Model pointer file
│
├── config/
│   └── universe.json            ← 100-ticker S&P 500 universe
│
├── .github/
│   └── workflows/
│       └── ci.yml               ← GitHub Actions CI
│
├── docker-compose.yml           ← Local orchestration
├── Dockerfile                   ← Production inference image
├── .env.example                 ← Environment template
├── .flake8                      ← Linting config
├── pytest.ini                   ← Test config
└── requirements.txt             ← Python dependencies
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- PostgreSQL 14+ (via Docker)
- Redis 7+ (via Docker)

### 1. Clone and Configure

```bash
git clone https://github.com/muhammedshihab1001/MarketSentinel.git
cd MarketSentinel
cp .env.example .env
```

### 2. Generate Credentials

```bash
# Activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Generate API key (no quotes needed in .env)
python scripts/generate_api_key.py

# Generate owner password hash (paste with double quotes in .env)
python scripts/generate_owner_hash.py
```

### 3. Configure .env

```env
# Database
POSTGRES_USER=sentinel
POSTGRES_PASSWORD=sentinel
POSTGRES_DB=marketsentinel
DATABASE_URL=postgresql+psycopg2://sentinel:sentinel@postgres:5432/marketsentinel

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Auth — generate with scripts above
API_KEY=your_generated_hex_key
OWNER_USERNAME=shihab
OWNER_PASSWORD_HASH="$2b$12$your_bcrypt_hash_here"
JWT_SECRET=your_jwt_secret_min_32_chars

# News (for political risk)
NEWSAPI_KEY=your_newsapi_key

# Optional
STORE_PREDICTIONS=1
LLM_ENABLED=false
DEMO_REQUESTS_PER_FEATURE=10
```

### 4. Start Services

```bash
docker-compose up -d
```

### 5. Verify Health

```bash
curl http://localhost:8000/health/ready
# Expected: { "ready": true, "models_loaded": true, "redis_connected": true, "db_connected": true }
```

### 6. First Prediction

```bash
# Login as demo
curl -X POST http://localhost:8000/auth/demo-login \
  -H "Content-Type: application/json" \
  -d "{}" -c cookies.txt

# Get snapshot (wait ~90s for first background compute)
curl -X POST http://localhost:8000/snapshot -b cookies.txt
```

---

## Training a New Model

```bash
# Run training pipeline inside Docker
docker-compose run --rm training python training/train_xgboost.py

# Evaluate and check promotion gates
docker-compose run --rm training python training/run_evaluation.py

# Retrain if drift is detected
docker-compose run --rm training python training/pipelines/train_pipeline.py
```

The trained model is saved to `artifacts/xgboost/` with a timestamp version (e.g. `model_xgb_20260407_215046.pkl`) and registered via `latest.json` pointer.

---

## Running Tests

```bash
# Activate virtual environment
venv\Scripts\activate

# Run all tests
pytest -q

# Expected output:
# 115 passed, 1 skipped in ~60s

# Run with coverage
pytest --cov=app --cov=core --cov-report=term-missing

# Run specific test file
pytest tests/test_signal_agent.py -v
```

---

## Linting

```bash
# Run flake8 (must have 0 violations)
flake8 app/ core/ training/ scripts/ tests/ \
  --max-line-length=100 \
  --extend-ignore=E501,W503,E203,E402,E221,E272,E128

# Expected: no output (0 violations)
```

---

## CI Pipeline

Every push to `feature/*` and pull request to `develop` runs:

```yaml
Jobs:
  1. Install dependencies (pip + requirements.txt)
  2. Start PostgreSQL test database
  3. Run flake8 linting (85% file pass threshold)
  4. Run pytest (115 tests, 1 skipped allowed)
  5. Verify schema signature integrity
```

CI blocks merges when any test fails.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `REDIS_HOST` | Yes | `redis` | Redis hostname |
| `REDIS_PORT` | No | `6379` | Redis port |
| `API_KEY` | Yes | — | External access key (no quotes) |
| `OWNER_USERNAME` | Yes | — | Admin login username |
| `OWNER_PASSWORD_HASH` | Yes | — | bcrypt hash (with double quotes) |
| `JWT_SECRET` | Yes | — | JWT signing secret (32+ chars) |
| `NEWSAPI_KEY` | No | — | NewsAPI key for political risk |
| `STORE_PREDICTIONS` | No | `1` | Store predictions for IC stats |
| `DEMO_REQUESTS_PER_FEATURE` | No | `10` | Demo quota per feature |
| `LLM_ENABLED` | No | `false` | Enable LLM signal explanations |
| `SKIP_DATA_SYNC` | No | `0` | Skip startup data sync |
| `CORS_ORIGINS` | No | `localhost:5173` | Allowed frontend origins |
| `SNAPSHOT_PRECOMPUTE_INTERVAL` | No | `300` | Snapshot refresh interval (seconds) |

---

## Docker Compose Services

```yaml
Services:
  api:        FastAPI inference server (port 8000)
  postgres:   PostgreSQL 14 database (port 5432)
  redis:      Redis 7 cache (port 6379)
  training:   Model training container (run on demand)
```

```bash
# Start all services
docker-compose up -d

# View API logs
docker-compose logs api --tail=50 -f

# Rebuild API after code changes
docker-compose build --no-cache api && docker-compose up -d api

# Check Redis demo keys
docker-compose exec redis redis-cli keys "demo:*"

# Access PostgreSQL
docker-compose exec postgres psql -U sentinel -d marketsentinel
```

---

## Monitoring & Observability

Prometheus metrics are exposed at `GET /metrics`:

```
# Inference
api_request_total{endpoint="/snapshot"}
api_latency_seconds{endpoint="/agent/explain"}
api_error_total{endpoint="/portfolio"}

# System
python_gc_objects_collected_total
process_virtual_memory_bytes
```

Connect Grafana to `http://localhost:8000/metrics` for dashboards.

---

## Security Architecture

```
Layer 1 — Transport
  All production traffic over HTTPS (enforced by Cloud Run / Vercel)

Layer 2 — API Gateway (main.py)
  Public paths: /health/*, /auth/*, /model/info, /agent/agents, /universe
  All other paths: require JWT cookie or X-API-KEY header
  Empty API_KEY: programmatic access disabled, JWT required

Layer 3 — Role Authorization (middleware.py)
  Owner: full access to all endpoints
  Demo: access with per-feature quota enforcement
  Unauthenticated: 401 on all protected paths

Layer 4 — Rate Limiting (Redis)
  Per IP + per endpoint limits
  Login endpoint: 5 req/60s (brute force protection)
  Expensive inference: 10 req/60s
  Fails open on Redis downtime (never blocks on infrastructure fault)

Layer 5 — Demo Quota (DemoTracker)
  Per browser fingerprint (IP + User-Agent hash)
  10 requests per feature group per week
  TTL-based auto-reset via Redis expiry
```

---

## Resolved Issues

All issues from the initial audit have been resolved:

| Issue | Fix |
|---|---|
| `#5` Hash generator quoting | Fixed bcrypt hash env handling |
| `#6` signals=27400 not 100 | Latest-per-ticker filter in pipeline |
| `#7` Concurrent snapshot crash | asyncio.Lock on background loop |
| `#8` REDIS_HOST Docker fix | Environment variable corrected |
| `#9` DB pool too small | Pool size increased 5 → 10 |
| `#10` Demo tracker Redis | Live cache from app.state per request |
| `#11` XGBoost hyperparameters | Retrained with iter=142 |
| `#12` Swagger docs missing | All routes documented |
| `#13` DB indexes not applied | Migration run applied |
| `#14` GDELT always failing | 6-provider fallback chain |
| `#15` Top-5 rationale missing | Full rationale in pipeline v5.9 |
| `#16` Political risk chain | Fallback chain deployed |
| `#17` Logging dev/prod | Environment-split logging |
| `#18` Demo limit was 3 | Changed to 10 |
| `#19` No per-endpoint rate limits | Full rate limit table added |
| `#20` Tests not passing | All 115 tests now passing |
| `#21` equity session_factory | Removed invalid kwarg |
| Security | API key skipped when empty | Always enforced now |
| Auth middleware | cache=None at init | Live Redis per request |
| Model loader | Two separate instances | Singleton via get_model_loader() |
| Equity single endpoint | 30-day window too small | Changed to 90-day |

---

## Performance Benchmarks

Measured on local Docker (Apple M2 / Intel i7):

| Operation | First Run | Cached |
|---|---|---|
| Full snapshot (100 tickers) | ~15–18 seconds | < 100ms |
| Agent explain | ~100–400ms | < 50ms |
| Portfolio | < 50ms | < 20ms |
| Drift | < 100ms | < 50ms |
| Feature engineering | ~12 seconds | N/A |

The first snapshot is slow because it fetches 400 days of price history for 100 tickers and runs the full feature pipeline. Every subsequent call reads from Redis cache.

---

## Author

**Muhammed Shihab P**

Building production ML systems, MLOps platforms, and decision intelligence engines.

---