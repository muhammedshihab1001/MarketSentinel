# MarketSentinel

**Institutional-Grade ML Trading Signal & Decision Intelligence Platform**

[![CI](https://github.com/muhammedshihab1001/MarketSentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammedshihab1001/MarketSentinel/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange.svg)](https://xgboost.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-red.svg)](https://redis.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Table of Contents

1. [What Is MarketSentinel](#what-is-marketsentinel)
2. [System Overview](#system-overview)
3. [End-to-End Data Flow](#end-to-end-data-flow)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Agent Decision System](#agent-decision-system)
7. [Drift Detection System](#drift-detection-system)
8. [Political Risk System](#political-risk-system)
9. [Authentication & Security](#authentication--security)
10. [Demo Quota System](#demo-quota-system)
11. [API Reference](#api-reference)
12. [Database Schema](#database-schema)
13. [Feature Engineering](#feature-engineering)
14. [Model Governance](#model-governance)
15. [Observability](#observability)
16. [Project Structure](#project-structure)
17. [Quick Start](#quick-start)
18. [Environment Variables](#environment-variables)
19. [CI Pipeline](#ci-pipeline)
20. [Performance Benchmarks](#performance-benchmarks)
21. [Author](#author)

---

## What Is MarketSentinel

MarketSentinel is a production-ready machine learning platform that ingests raw equity market data and produces **risk-aware, explainable trading signals** governed by a multi-agent decision system.

It solves the problems that break ML systems in production:

| Problem | MarketSentinel Solution |
|---|---|
| Feature inconsistency between train and inference | Canonical 64-feature schema with SHA256 signature enforcement |
| Silent model decay | Real-time drift detector with 0–15 severity scoring |
| Blind predictions | 4-agent pipeline with per-agent approval/flagging and natural language rationale |
| Uncontrolled access | JWT + API key + per-endpoint Redis rate limiting |
| Unreproducible training | Artifact registry with dataset hash, code hash, schema signature |
| Geopolitical blind spots | GDELT + 5-provider news fallback chain for political risk overlay |
| No IC measurement | Prediction storage + Spearman IC computation over rolling windows |

---

## System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MARKETSENTINEL PLATFORM                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      DATA LAYER                                     │    ║
║  │                                                                     │    ║
║  │  Yahoo Finance  TwelveData  Custom Providers                        │    ║
║  │       │              │            │                                  │    ║
║  │       └──────────────┴────────────┘                                 │    ║
║  │                       │                                             │    ║
║  │              DataSyncService (daily 18:30)                          │    ║
║  │                       │                                             │    ║
║  │              PostgreSQL (OHLCV + Features + Predictions)            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                               │                                              ║
║                               ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    TRAINING PIPELINE                                │    ║
║  │                                                                     │    ║
║  │  FeatureEngineer → XGBoost Training → Walk-Forward Validation       │    ║
║  │  → Artifact Registry → production_pointer.json                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                               │                                              ║
║                               ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                   INFERENCE PIPELINE (live)                         │    ║
║  │                                                                     │    ║
║  │  PostgreSQL OHLCV → FeatureEngineer (64 features)                  │    ║
║  │  → Schema Validation → XGBoost Inference → Per-Ticker Agent Loop   │    ║
║  │  → Drift Overlay → Political Risk Overlay → Hybrid Score           │    ║
║  │  → Portfolio Decision → Top-5 Rationale → Redis Cache              │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                               │                                              ║
║                               ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                   FASTAPI REST API (v3.7)                           │    ║
║  │                                                                     │    ║
║  │  /snapshot  /portfolio  /drift  /performance  /agent/explain        │    ║
║  │  /agent/political-risk  /equity  /model/info  /auth  /health        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                               │                                              ║
║                               ▼                                              ║
║                    React Frontend Dashboard (Vercel)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## End-to-End Data Flow

```
STEP 1 — RAW DATA INGESTION
════════════════════════════
  DataSyncService.sync_universe()
  ┌─────────────────────────────────────────────────────────┐
  │  Input:  100 ticker symbols from config/universe.json   │
  │  Action: Fetch OHLCV data for each ticker               │
  │  Source: Yahoo Finance → TwelveData (fallback)          │
  │  Window: Last 730 days (training) / 400 days (inference)│
  │  Output: ~40,000 rows stored in PostgreSQL              │
  │                                                         │
  │  PostgreSQL table: ohlcv_daily                          │
  │  Constraint: uq_ohlcv_ticker_date (no duplicates)       │
  │  Retention: rows older than 760 days auto-deleted        │
  └─────────────────────────────────────────────────────────┘
  Schedule: Daily at 18:30 (weekdays only)
  Startup:  Runs if data is stale (> 24 hours old)


STEP 2 — FEATURE ENGINEERING
══════════════════════════════
  FeatureEngineer.build_feature_pipeline()
  ┌─────────────────────────────────────────────────────────┐
  │  Input:  Raw OHLCV DataFrame (100 tickers × 400 days)  │
  │  Output: 64 features per row                            │
  │  Feature version: SHA256(feature_names) → used as       │
  │                   cache key in computed_features table  │
  │                                                         │
  │  Inference mode: checks DB cache first (35s → ~2s warm) │
  │  Training mode:  always fresh, never cached             │
  └─────────────────────────────────────────────────────────┘


STEP 3 — XGBOOST INFERENCE
════════════════════════════
  ModelLoader.predict(X)
  ┌─────────────────────────────────────────────────────────┐
  │  Pointer: artifacts/xgboost/production_pointer.json     │
  │           → resolves to model_xgb_YYYYMMDD_HHMMSS.pkl  │
  │  Input:   float32 matrix, shape (100, 64)               │
  │  Output:  raw_scores[], shape (100,)                    │
  │                                                         │
  │  Score filtering: filter_latest_per_ticker()            │
  │  → keeps 1 row per ticker (latest date)                 │
  │  → fixes 27,400-row → 100-row bug (v5.8)               │
  └─────────────────────────────────────────────────────────┘


STEP 4 — HYBRID SCORE CALCULATION
═══════════════════════════════════
  hybrid = (0.50 × raw_model_score
           + 0.30 × signal_agent_score      ← uses "score" key
           + 0.20 × technical_agent_score)

  Political overlay:
    CRITICAL → hybrid = 0.0
    HIGH     → hybrid × 0.5

  Drift overlay:
    weight = hybrid × exposure_scale
    none → 1.0 | soft → 0.6 | hard → 0.25


STEP 5 — CACHE & SERVE
═══════════════════════
  Redis key: ms:background_snapshot:latest
  TTL: 360 seconds (configurable)
  Background loop: every 300 seconds
  asyncio.Lock: prevents concurrent snapshot runs
  Fallback: in-memory dict if Redis is down
```

---

## Training Pipeline

```
TRAINING WORKFLOW
══════════════════

  Run: docker compose --profile training run --rm training

  STEP 1: Data sync (unless SKIP_SYNC=1)
  STEP 2: Feature engineering (training=True, never cached)
  STEP 3: Walk-forward validation (5 rolling windows)
  STEP 4: Final model trained on full dataset
  STEP 5: export_artifacts() saves:
            model_xgb_YYYYMMDD_HHMMSS.pkl
            metadata_YYYYMMDD_HHMMSS.json
            production_pointer.json  ← ModelLoader reads this
  STEP 6: Drift baseline created/promoted

  NOTE: training service uses profiles:[training] so it does NOT
  auto-start with docker compose up -d. Must run explicitly.

  Retrain (data already in DB):
    docker compose --profile training run --rm \
      -e SKIP_SYNC=1 training

  First time (needs full data sync):
    docker compose --profile training run --rm \
      -e CREATE_BASELINE=1 training


MODEL POINTER FILE:
════════════════════
  artifacts/xgboost/production_pointer.json
  {
    "model_version": "xgb_20260407_215046",
    "model_path":    "/app/artifacts/xgboost/model_xgb_20260407_215046.pkl",
    "metadata_path": "/app/artifacts/xgboost/metadata_xgb_20260407_215046.json",
    "updated_at":    "2026-04-07T21:50:46Z"
  }

  ModelLoader v2.9 checks in order:
    1. production_pointer.json  (primary — created by training)
    2. latest.json              (legacy alias — backward compat)
    3. Directory scan           (fallback — warning logged)
```

---

## Inference Pipeline

```
BACKGROUND SNAPSHOT LOOP (every 300 seconds)
═════════════════════════════════════════════

  _background_snapshot_loop() — asyncio task started at boot

  Boot delay: 30 seconds
  Concurrency: asyncio.Lock — one run at a time
  On skip:     logs warning, waits full interval

  Signals: 100 (one per universe ticker)
  Latency: 15–18 seconds first run, < 100ms cached


TOP-5 RATIONALE OBJECT:
════════════════════════
  rank, ticker, signal, hybrid_score, raw_model_score,
  weight, confidence, risk_level, governance_score,
  volatility_regime, technical_bias, drift_context,
  political_context, agent_scores, agents_approved,
  agents_flagged, warnings, selection_reason
```

---

## Agent Decision System

```
4-AGENT PIPELINE
══════════════════

  AGENT 1: SignalAgent                weight: 0.50
  ─────────────────────────────────────────────────
  Input:  raw_model_score, RSI, EMA, momentum, volatility,
          drift_state, political_risk_label
  Output: signal (LONG/SHORT/NEUTRAL), confidence,
          risk_level, governance_score, agent_score,
          score (hybrid formula key), warnings, explanation

  AGENT 2: TechnicalRiskAgent         weight: 0.20
  ─────────────────────────────────────────────────
  Input:  RSI, EMA ratio, momentum_z, regime_feature
  Output: volatility_regime, technical_bias, score,
          agent_score, warnings

  AGENT 3: PoliticalRiskAgent         weight: 0.10
  ─────────────────────────────────────────────────
  Runs ONCE per snapshot (not per ticker)
  6-provider fallback chain:
    GDELT → NewsAPI → GNews → TheNewsAPI → Mediastack → Currentsapi
    If all fail: score=0.0, label=UNAVAILABLE

  Score thresholds:
    0.00–0.25: LOW      (no overlay)
    0.25–0.50: MEDIUM   (no overlay)
    0.50–0.75: HIGH     (hybrid × 0.5)
    0.75–1.00: CRITICAL (hybrid = 0.0)

  AGENT 4: PortfolioDecisionAgent     weight: 0.20
  ─────────────────────────────────────────────────
  Input:  all 100 snapshot_rows + drift dict (full)
  Output: sector-neutral top-K selection, executive summary
  Sector cap: max 2 positions per GICS sector (MAX_PER_SECTOR=2)


HYBRID SCORE FORMULA:
══════════════════════
  hybrid = 0.50 × raw_model_score
         + 0.30 × signal_agent.score   ← "score" key in return dict
         + 0.20 × technical_agent.score
```

---

## Drift Detection System

```
SEVERITY SCORING (0 – 15):
════════════════════════════
  0:      No drift
  1–4:    Minimal — monitor
  5–9:    Soft drift — reduce positions (exposure_scale=0.6)
  10–14:  Hard drift — heavily reduce (exposure_scale=0.25)
  15:     Critical

  NOTE: severity_score is a 0-15 integer, NOT a percentage.
        Display as "X / 15" not "X%".

RETRAIN TRIGGER:
════════════════
  evaluate() accepts: int OR dict (Union type)
  → was a bug where int input caused AttributeError
  Threshold: DRIFT_RETRAIN_THRESHOLD env var (default: 8)
  Cooldown: RETRAIN_COOLDOWN_SECONDS (default: 3600)

API /drift response:
  drift_detected, severity_score (0-15), drift_confidence,
  drift_state, exposure_scale, retrain_required,
  cooldown_active, cooldown_remaining_seconds,
  baseline_exists, baseline_version, universe_size
```

---

## Political Risk System

```
6-PROVIDER FALLBACK CHAIN:
════════════════════════════
  1. GDELT        (free, no key)
  2. NewsAPI      (NEWSAPI_KEY — 100 req/day free)
  3. GNews        (GNEWS_KEY — 100 req/day free)
  4. TheNewsAPI   (THENEWSAPI_KEY — 100 req/day free)
  5. Mediastack   (MEDIASTACK_KEY — 500 req/month free)
  6. CurrentsAPI  (CURRENTSAPI_KEY — 600 req/day free)

  gdelt_status shows which provider was used.
  top_events is string[] — plain headline strings.
```

---

## Authentication & Security

```
TWO ACCESS MODES:
══════════════════
  Mode 1: JWT Cookie (Browser Users)
    POST /auth/owner-login → httpOnly cookie ms_token (30 days)
    POST /auth/demo-login  → httpOnly cookie ms_token (24 hours)

  Mode 2: API Key (External Programmatic Access)
    Header: X-API-KEY: <key from generate_api_key.py>
    Effect: treated as owner role

SECURITY NOTES:
════════════════
  OWNER_USERNAME:     never hardcoded — reads from .env only
  OWNER_PASSWORD_HASH: never hardcoded — reads from .env only
  CORS_ORIGINS:       reads from .env only — never set in docker-compose
  API_KEY:            if not set, programmatic access is fully blocked

5 SECURITY LAYERS:
═══════════════════
  1. Network (HTTPS + CORS_ORIGINS from env)
  2. main.py middleware (PUBLIC_PATHS + API key check)
  3. Redis rate limiting per IP per endpoint (fails open)
  4. AuthMiddleware (role enforcement)
  5. DemoTracker (per-feature quota in Redis)

PER-ENDPOINT RATE LIMITS:
══════════════════════════
  /auth/owner-login      →  5 req / 60s
  /auth/demo-login       → 10 req / 60s
  /snapshot              → 10 req / 60s
  /predict/live-snapshot → 10 req / 60s
  /agent/explain         → 20 req / 60s
  /agent/political-risk  → 20 req / 60s
  /performance           → 20 req / 60s
  /health/live           → 60 req / 60s
  /health/ready          → 60 req / 60s
  All other paths        → 60 req / 60s
```

---

## Demo Quota System

```
PER-FEATURE QUOTA (default: 10 per feature per week):
══════════════════════════════════════════════════════
  Feature key  │ Endpoints
  ─────────────┼─────────────────────────────────────────────
  snapshot     │ /snapshot, /predict/live-snapshot
  portfolio    │ /portfolio
  drift        │ /drift
  performance  │ /performance
  agent        │ /agent/explain, /agent/political-risk
  signals      │ /equity/*, /model/feature-importance

  Each feature has its OWN quota.
  fully_locked = true only when ALL features exhausted.
  TTL: 7 days (auto-reset after 1 week)
  Redis key: demo:usage:{fingerprint}:{feature}
  Fingerprint: SHA256(IP + User-Agent)[:32]
```

---

## API Reference

```
PUBLIC ENDPOINTS (no auth required)
═════════════════════════════════════
  GET  /                    Root info + system status
  GET  /health/live         Docker liveness probe → 200 OK
  GET  /health/ready        Readiness: model + redis + db
  GET  /health/db           Database connectivity + latency
  GET  /health/model        Model artifact integrity
  GET  /universe            100-ticker S&P 500 list
  GET  /model/info          Model version + hashes + feature count
  GET  /agent/agents        4-agent descriptions + weights
  GET  /docs                Swagger UI (interactive API docs)
  GET  /metrics             Prometheus metrics export

AUTH ENDPOINTS (public)
════════════════════════
  POST /auth/owner-login    Admin login → JWT cookie
  POST /auth/demo-login     Demo access → JWT cookie
  GET  /auth/me             Current session + quota
  POST /auth/logout         Clear JWT cookie

PROTECTED ENDPOINTS (owner or demo)
═════════════════════════════════════
  POST /snapshot                          Full inference → 100 signals
  GET  /predict/live-snapshot             Same (GET alias)
  GET  /portfolio                         Portfolio positions + health
  GET  /drift                             Model drift metrics
  GET  /performance?days=N               Strategy performance
  GET  /performance/{ticker}?days=N      Per-ticker performance
  GET  /agent/explain?ticker=X           Per-ticker signal explanation
  GET  /agent/political-risk?ticker=X    Geopolitical risk
  GET  /equity/{ticker}                  Latest OHLCV + returns
  GET  /equity/{ticker}/history?days=N   Historical price data
  GET  /predict/signal-explanation/{ticker}
  GET  /predict/price-history/{ticker}?days=N

OWNER-ONLY ENDPOINTS
═════════════════════
  GET  /model/ic-stats?days=30   IC statistics (Spearman)
  GET  /model/feature-importance Top features by XGBoost gain
  GET  /model/diagnostics        Full model checksums
  POST /admin/sync               Trigger manual data sync


KEY RESPONSE NOTES:
════════════════════
  /drift
    severity_score: 0-15 integer (NOT percentage)
    Display as "X / 15"

  /equity/{ticker}/history
    Response key is "history" (NOT "prices")
    Each item: { date, open, high, low, close, volume }

  /agent/explain (top-5 ticker)
    agents_approved, agents_flagged, selection_reason, agent_scores populated
    rank: 1-5

  /agent/explain (non-top-5 ticker)
    rank: null, in_top_5: false
    agents_approved: [], agents_flagged: [], selection_reason: ""

  /agent/political-risk
    top_events: string[] (plain strings, NOT objects)

  /performance
    annual_return: included in response
    tracking_error, beta, information_ratio: null (no benchmark)
```

---

## Database Schema

```
TABLE: ohlcv_daily
───────────────────
  id       BIGSERIAL PRIMARY KEY
  ticker   VARCHAR(20) NOT NULL
  date     DATE NOT NULL
  open, high, low, close  FLOAT
  volume   FLOAT
  source   VARCHAR(30) DEFAULT 'yfinance'
  UNIQUE: uq_ohlcv_ticker_date
  INDEX:  ix_ohlcv_ticker_date, ix_ohlcv_date
  Rows: ~40,000 (100 tickers × 400 days)
  Retention: rows older than 760 days auto-deleted after sync


TABLE: computed_features
─────────────────────────
  ticker, date, feature_version VARCHAR(64), feature_data JSONB
  UNIQUE: uq_feature_ticker_date_version
  INDEX:  ix_feature_version_ticker_date (composite covering index)
  Purpose: inference feature cache (35s cold → ~2s warm)


TABLE: model_predictions
─────────────────────────
  ticker, date, model_version, schema_signature VARCHAR(64),
  raw_model_score, hybrid_score, weight, signal, drift_state
  UNIQUE: uq_prediction_ticker_date_model
  Purpose: /model/ic-stats Spearman IC computation
  Requires: STORE_PREDICTIONS=1 in .env
```

---

## Feature Engineering

```
64 FEATURES PER TICKER (at inference):
══════════════════════════════════════

  Core features (23):
    return, return_lag1, return_lag5, return_mean_20,
    reversal_5, momentum_20, momentum_60, momentum_composite,
    mom_vol_adj, momentum_regime_interaction, volatility,
    volatility_20, vol_of_vol, return_skew_20, volume_momentum,
    dollar_volume, amihud, rsi, ema_ratio, dist_from_52w_high,
    regime_feature, market_dispersion, breadth,
    regime_multiplier  ← BULL=1.2/SIDEWAYS=1.0/BEAR=0.6/CRISIS=0.3

  Cross-sectional z-scores (20): {feature}_z
  Cross-sectional ranks    (20): {feature}_rank

  Schema signature: SHA256(feature_names + version)
  Must match model artifact signature or warning is logged.

  Inference shortcut:
    filter_latest_per_ticker() keeps 1 row per ticker
    → 100 rows sent to XGBoost (not 27,400)
```

---

## Model Governance

```
ARTIFACT STRUCTURE:
════════════════════
  artifacts/xgboost/
  ├── model_xgb_YYYYMMDD_HHMMSS.pkl     ← model (joblib)
  ├── metadata_xgb_YYYYMMDD_HHMMSS.json ← checksums + metrics
  └── production_pointer.json            ← ModelLoader reads this
                                           (NOT latest.json)

  production_pointer.json:
  {
    "model_version": "xgb_20260407_215046",
    "model_path":    "/app/artifacts/xgboost/model_xgb_...pkl",
    "metadata_path": "/app/artifacts/xgboost/metadata_...json",
    "updated_at":    "2026-04-07T21:50:46Z"
  }


MODEL LOADING ORDER (ModelLoader v2.9):
═════════════════════════════════════════
  1. production_pointer.json  (primary)
  2. latest.json              (legacy alias)
  3. Directory scan           (fallback — warning logged)

  IMPORTANT: directory scan picks alphabetically latest
  model which may NOT be the best model after a bad training
  run. Always use production_pointer.json.


HYPERPARAMETERS (v4.5.3 — tuned for financial time series):
═════════════════════════════════════════════════════════════
  eta:              0.01   (was 0.05 — caused early stopping at iter=4)
  max_depth:        3      (was 4)
  num_boost_rounds: 2000   (was 400)
  early_stopping:   50     (was 30)
  min_boost_rounds: 30     (was 10)
  Expected best_iteration: 80–150


IC STATISTICS:
══════════════
  Spearman correlation: raw_model_score vs next-day forward return
  Rolling 30-day window.
  Requires STORE_PREDICTIONS=1 in .env.

  Interpretation:
    IC > 0.08    → Strong alpha
    IC 0.04–0.08 → Moderate signal
    IC 0.02–0.04 → Weak signal
    IC < 0.02    → Near noise
```

---

## Observability

```
PROMETHEUS METRICS (GET /metrics):
════════════════════════════════════
  api_requests_total{endpoint}
  api_errors_total{endpoint}
  api_latency_seconds{endpoint}
  model_inference_total{model}
  model_inference_latency_seconds{model}
  model_version{model, version}
  signal_distribution_total{signal}
  drift_detected                    ← 0-15 severity score
  cache_hits_total
  cache_misses_total
  db_query_total{operation}
  db_query_latency_seconds{operation}
  db_rows_written_total{table}
  db_sync_total{status}
  db_tickers_synced_total
  missing_feature_ratio
  inference_in_progress
  pipeline_failures_total{stage}

  Prometheus scrapes: api:8000/metrics  (docker service name)
  Grafana datasource: http://prometheus:9090

GRAFANA DASHBOARD PANELS:
══════════════════════════
  - Total API requests (stat)
  - Error rate % (stat)
  - Cache hit rate % (stat)
  - Drift severity score (stat, 0-15)
  - Request rate per endpoint (timeseries)
  - API latency p95 ms (timeseries)
  - Requests by endpoint bar chart
  - Error count per endpoint
  - Active inferences
  - DB query latency p95
  - DB rows written
  - Tickers synced gauge
  - Missing feature ratio
  - Signal distribution LONG vs SHORT

LOG FILES:
═══════════
  logs/marketsentinel.log  ← INFO+ all environments
  logs/issues.log          ← WARNING+ all environments
  logs/debug.log           ← DEBUG dev mode only
  logs/access.log          ← HTTP requests (uvicorn)
  Production: console output DISABLED (files only)
```

---

## Project Structure

```
MarketSentinel/
│
├── app/                           ← FastAPI inference control plane
│   ├── api/routes/
│   │   ├── agent.py               ← /agent/* endpoints
│   │   ├── auth.py                ← /auth/* endpoints
│   │   ├── drift.py               ← /drift
│   │   ├── equity.py              ← /equity/*
│   │   ├── health.py              ← /health/*
│   │   ├── model_info.py          ← /model/* endpoints
│   │   ├── performance.py         ← /performance
│   │   ├── portfolio.py           ← /portfolio
│   │   ├── predict.py             ← /predict/*
│   │   └── universe.py            ← /universe
│   ├── api/schemas.py             ← Pydantic models
│   ├── agent/llm_explainer.py     ← Optional LLM explanations
│   ├── core/auth/
│   │   ├── demo_tracker.py        ← Redis quota tracker
│   │   ├── jwt_handler.py         ← JWT encode/decode
│   │   └── middleware.py          ← AuthMiddleware v2.4
│   ├── inference/
│   │   ├── cache.py               ← Redis + memory fallback
│   │   ├── model_loader.py        ← Singleton loader v2.9
│   │   └── pipeline.py            ← InferencePipeline v5.9.1
│   ├── monitoring/
│   │   ├── metrics.py             ← Prometheus exporters
│   │   ├── prometheus.yml         ← Scrape config (target: api:8000)
│   │   └── grafana_dashboard.json ← Pre-built dashboard
│   └── main.py                    ← FastAPI app v3.7
│
├── core/                          ← Business domain (framework-agnostic)
│   ├── agent/
│   │   ├── base_agent.py
│   │   ├── signal_agent.py
│   │   ├── technical_risk_agent.py
│   │   ├── portfolio_decision_agent.py  ← sector neutralisation
│   │   └── political_risk_agent.py      ← 6-provider fallback
│   ├── analytics/performance_engine.py
│   ├── artifacts/
│   │   ├── metadata_manager.py
│   │   └── model_registry.py
│   ├── config/env_loader.py
│   ├── data/
│   │   ├── data_sync.py           ← Daily 18:30 sync scheduler
│   │   ├── market_data_service.py ← parallel batch reads (8 workers)
│   │   └── providers/market/      ← Yahoo + TwelveData + router
│   ├── db/
│   │   ├── engine.py              ← SQLAlchemy pool (size=10, overflow=5)
│   │   ├── models.py              ← ORM: OHLCV, Features, Predictions
│   │   └── repository.py          ← vectorised bulk ops
│   ├── features/feature_engineering.py  ← 64-feature pipeline
│   ├── indicators/technical_indicators.py
│   ├── market/universe.py         ← 100-ticker universe controller
│   ├── models/xgboost.py          ← SafeXGBRegressor v4.5.3
│   ├── monitoring/
│   │   ├── drift_detector.py      ← KS + PSI drift scoring
│   │   ├── market_regime_detector.py
│   │   └── retrain_trigger.py     ← Union[int, Dict] evaluate()
│   ├── schema/feature_schema.py   ← 64-feature contract + SHA256
│   └── time/market_time.py
│
├── training/
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   └── walk_forward.py        ← rolling validation
│   ├── pipelines/train_pipeline.py ← init_db + sync + train + baseline
│   ├── train_xgboost.py
│   └── evaluate.py
│
├── scripts/
│   ├── generate_api_key.py        ← 32-byte cryptographic hex key
│   └── generate_owner_hash.py     ← bcrypt hash (prompts username)
│
├── tests/                         ← 197 passing, 1 skipped
│   ├── conftest.py
│   └── test_*.py  (17 test files)
│
├── docker/
│   ├── inference.Dockerfile       ← multi-stage, PYTHONPATH=/app
│   ├── training.Dockerfile        ← multi-stage, entrypoint.sh
│   └── entrypoint.sh              ← sync → train → baseline
│
├── config/
│   ├── universe.json              ← 100 S&P 500 tickers
│   └── universe_research.json
│
├── .github/workflows/ci.yml       ← lint + test + docker build
├── docker-compose.yml             ← training uses profiles:[training]
├── .env.example                   ← documented env vars
├── .flake8
├── pytest.ini
└── requirements/
    ├── base.txt
    ├── ci.txt
    ├── inference.txt
    └── training.txt
```

---

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Python 3.10
- 4 GB RAM minimum
- Internet access (for data sync + political risk)

### Step 1 — Clone

```bash
git clone https://github.com/muhammedshihab1001/MarketSentinel.git
cd MarketSentinel
```

### Step 2 — Generate Credentials

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements/base.txt

# Generates API key (no quotes in .env)
python scripts/generate_api_key.py

# Generates bcrypt hash (prompts for username + password)
python scripts/generate_owner_hash.py
```

### Step 3 — Configure .env

Copy `.env.example` to `.env` and fill in:

```env
# Auth (required)
OWNER_USERNAME=your_username
OWNER_PASSWORD_HASH="$2b$12$..."   # double quotes required
JWT_SECRET=your-32-char-secret
API_KEY=your64hexkey               # no quotes

# Database (defaults work with docker-compose)
POSTGRES_HOST=postgres
DATABASE_URL=postgresql+psycopg2://sentinel:sentinel@postgres:5432/marketsentinel

# Redis
REDIS_HOST=redis

# CORS — set to your frontend URL in production
CORS_ORIGINS=http://localhost:5173

# Optional
NEWSAPI_KEY=your_newsapi_key
STORE_PREDICTIONS=1
LLM_ENABLED=false
```

### Step 4 — Start Services

```bash
docker compose up -d
```

This starts: PostgreSQL, Redis, API, Prometheus, Grafana.
Training does NOT start automatically (uses `profiles:[training]`).

### Step 5 — First-Time Training

```bash
# Sync data + train model + create drift baseline
docker compose --profile training run --rm \
  -e CREATE_BASELINE=1 training
```

### Step 6 — Verify

```bash
curl http://localhost:8000/health/ready
# {"ready":true,"models_loaded":true,"db_connected":true,...}

# Wait ~90s for first background snapshot, then:
curl -X POST http://localhost:8000/auth/demo-login \
  -H "Content-Type: application/json" -d "{}" -c cookies.txt

curl -X POST http://localhost:8000/snapshot -b cookies.txt
```

### Retrain (data already in DB)

```bash
docker compose --profile training run --rm \
  -e SKIP_SYNC=1 training
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full PostgreSQL connection string |
| `REDIS_HOST` | Yes | `redis` | Redis hostname (use `redis` in Docker) |
| `REDIS_PORT` | No | `6379` | Redis port |
| `API_KEY` | Yes | — | External API key — no quotes in .env |
| `OWNER_USERNAME` | Yes | — | Admin username — never hardcoded |
| `OWNER_PASSWORD_HASH` | Yes | — | bcrypt hash — with double quotes in .env |
| `JWT_SECRET` | Yes | — | JWT signing secret (minimum 32 characters) |
| `NEWSAPI_KEY` | No | — | NewsAPI fallback for political risk |
| `GNEWS_KEY` | No | — | GNews fallback |
| `THENEWSAPI_KEY` | No | — | TheNewsAPI fallback |
| `MEDIASTACK_KEY` | No | — | Mediastack fallback |
| `CURRENTSAPI_KEY` | No | — | CurrentsAPI fallback |
| `STORE_PREDICTIONS` | No | `1` | Store predictions for IC stats |
| `DEMO_REQUESTS_PER_FEATURE` | No | `10` | Demo quota per feature per week |
| `LLM_ENABLED` | No | `false` | Enable OpenAI explanations |
| `OPENAI_API_KEY` | No | — | Required when LLM_ENABLED=true |
| `SKIP_DATA_SYNC` | No | `0` | Skip data sync on API startup |
| `CORS_ORIGINS` | Yes | — | Comma-separated allowed origins — set in .env only, never in docker-compose |
| `COOKIE_SECURE` | No | `0` | Set to `1` in production (HTTPS) |
| `COOKIE_SAMESITE` | No | `lax` | Set to `none` in production with cross-origin frontend |
| `SNAPSHOT_PRECOMPUTE_INTERVAL` | No | `300` | Background snapshot frequency (seconds) |
| `INFERENCE_LOOKBACK_DAYS` | No | `400` | Price history window for features |
| `TRAINING_LOOKBACK_DAYS` | No | `730` | Price history window for training |
| `GF_SECURITY_ADMIN_USER` | No | `admin` | Grafana admin username |
| `GF_SECURITY_ADMIN_PASSWORD` | No | `admin` | Grafana admin password — change in production |
| `LOG_LEVEL` | No | `INFO` | Log verbosity level |
| `APP_ENV` | No | `production` | `development` or `production` |

---

## CI Pipeline

```
.github/workflows/ci.yml
═════════════════════════

  Triggers:
    push:         feature/*, develop, main
    pull_request: develop, main

  Job 1 — Tests & Lint (runs on all triggers)
  ─────────────────────────────────────────────
  Services: PostgreSQL 16-alpine
  Steps:
    1. Install requirements/ci.txt
    2. flake8 — 85% pass threshold
       max-line-length=100
       ignore: E501, W503, E203, E402, E221, E272, E128
    3. pytest tests/ — 197 must pass

  Job 2 — Docker Build (main/develop only)
  ─────────────────────────────────────────
  Builds both images in parallel:
    docker/inference.Dockerfile → marketsentinel-inference:ci
    docker/training.Dockerfile  → marketsentinel-training:ci

  Current status: all checks passing ✅
  Test count: 197 passed, 1 skipped
```

---

## Performance Benchmarks

| Operation | Cold (first run) | Warm (Redis cached) |
|---|---|---|
| Full snapshot (100 tickers) | 15–18 seconds | < 100ms |
| Feature engineering (100 × 400 days) | ~12 seconds | ~2s (DB cache warm) |
| Agent explain (any ticker) | 100–400ms | < 50ms |
| Portfolio | < 50ms | < 20ms |
| Drift | < 100ms | < 50ms |
| Performance (252 days) | 2–5 seconds | N/A |
| Political risk (GDELT) | 3–8 seconds | < 20ms |
| Political risk (NewsAPI fallback) | 1–3 seconds | < 20ms |

First snapshot is slow because it fetches 400 days × 100 tickers from PostgreSQL, computes 64 features per row, and runs the full 4-agent pipeline. All subsequent calls serve from Redis cache in milliseconds.

The feature cache (`computed_features` table) reduces inference warm start from ~35s to ~2s by storing the computed feature matrix in PostgreSQL between snapshot runs.

---

## Author

**Muhammed Shihab P**

Building production ML systems, MLOps platforms, and decision intelligence engines.

- GitHub: [muhammedshihab1001](https://github.com/muhammedshihab1001)
- Project: [MarketSentinel](https://github.com/muhammedshihab1001/MarketSentinel)

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

Copyright (c) 2026 Muhammed Shihab P