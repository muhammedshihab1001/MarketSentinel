# MarketSentinel

**Institutional-Grade ML Trading Signal & Decision Intelligence Platform**

[![CI](https://github.com/muhammedshihab1001/MarketSentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammedshihab1001/MarketSentinel/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14-blue.svg)](https://postgresql.org/)
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
║  │  → Promotion Gates → Artifact Registry → latest.json pointer        │    ║
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
║                    React Frontend Dashboard                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## End-to-End Data Flow

This is the complete journey from raw market data to a trading signal displayed in the dashboard.

```
STEP 1 — RAW DATA INGESTION
════════════════════════════
  Market opens each trading day
         │
         ▼
  DataSyncService.sync_universe()
  ┌─────────────────────────────────────────────────────────┐
  │  Input:  100 ticker symbols from config/universe.json   │
  │  Action: Fetch OHLCV data for each ticker               │
  │  Source: Yahoo Finance → TwelveData (fallback)          │
  │  Window: Last 400 trading days per ticker               │
  │  Output: 100 × 400 = ~40,000 rows stored in PostgreSQL  │
  │                                                         │
  │  PostgreSQL table: ohlcv_daily                          │
  │  Columns: ticker, date, open, high, low, close, volume  │
  │  Constraint: uq_ohlcv_ticker_date (no duplicates)       │
  └─────────────────────────────────────────────────────────┘
  Schedule: Daily at 18:30 (weekdays only)
  Startup:  Runs if data is stale (> 24 hours old)


STEP 2 — FEATURE ENGINEERING
══════════════════════════════
  PostgreSQL OHLCV data
         │
         ▼
  FeatureEngineer.build_feature_pipeline()
  ┌─────────────────────────────────────────────────────────┐
  │  Input:  Raw OHLCV DataFrame (100 tickers × 400 days)  │
  │                                                         │
  │  Computes 64 features per row:                          │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │ Price & Return      │ Momentum          │ Vol     │   │
  │  │ close_rank          │ momentum_5        │ vol_20  │   │
  │  │ return_1d           │ momentum_20       │ atr_14  │   │
  │  │ return_5d           │ momentum_20_z     │ regime  │   │
  │  │ return_20d          │ reversal_5_rank   │         │   │
  │  │ log_return          │ rsi_14            │         │   │
  │  ├──────────────────────────────────────────────────┤   │
  │  │ Moving Averages     │ EMA Structure     │ Cross   │   │
  │  │ ema_20              │ ema_ratio         │ Sect.   │   │
  │  │ ema_50              │ trend_strength    │ price_  │   │
  │  │ sma_20              │ ema_slope         │ rank    │   │
  │  │ ema_20_slope        │ price_above_ema   │ vol_    │   │
  │  │                     │ golden_cross      │ rank    │   │
  │  └──────────────────────────────────────────────────┘   │
  │                                                         │
  │  Schema validation: compare to feature_schema.py        │
  │  SHA256 signature: must match model artifact signature  │
  │  Mismatch → warning logged, retrain recommended         │
  │                                                         │
  │  Output: DataFrame of shape (100, 64)                   │
  │          One row per ticker (latest date only)          │
  └─────────────────────────────────────────────────────────┘


STEP 3 — XGBOOST INFERENCE
════════════════════════════
  Feature matrix (100 × 64)
         │
         ▼
  ModelLoader.predict(X)
  ┌─────────────────────────────────────────────────────────┐
  │  Model:   artifacts/xgboost/model_xgb_YYYYMMDD_*.pkl   │
  │  Input:   float32 matrix, shape (100, 64)               │
  │  Output:  raw_scores[], shape (100,)                    │
  │                                                         │
  │  Score interpretation:                                  │
  │  > +1.5  Very strong positive signal                    │
  │  > +0.5  Moderate positive signal                       │
  │  ~  0.0  Neutral / no conviction                        │
  │  < -0.5  Moderate negative signal                       │
  │  < -1.5  Very strong negative signal                    │
  │                                                         │
  │  Current model: xgb_20260407_215046                     │
  │  Features used: 62 (of 64 schema features)              │
  │  Best iteration: 142 (early stopping)                   │
  └─────────────────────────────────────────────────────────┘


STEP 4 — PER-TICKER AGENT LOOP
════════════════════════════════
  For each of 100 tickers:
  raw_score + feature_row + drift_state + political_label
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  SignalAgent.analyze(context)                           │
  │  ─────────────────────────────                          │
  │  Input:  raw_score, RSI, EMA, momentum, volatility      │
  │  Logic:                                                 │
  │    confidence = clip(abs(score) / 2.0, 0, 1)           │
  │    if drift: confidence × 0.75                          │
  │    if political == CRITICAL: signal → NEUTRAL           │
  │    governance_score = int(agent_score × 100)           │
  │  Output:                                                │
  │    signal:           LONG / SHORT / NEUTRAL             │
  │    confidence:       0.0 – 1.0                         │
  │    risk_level:       low / moderate / high / elevated   │
  │    governance_score: 0 – 100                            │
  │    agent_score:      0.0 – 1.0                         │
  │    warnings:         list of risk flags                 │
  │    explanation:      "SIGNAL | score | conf | tech"     │
  │                                                         │
  │  TechnicalRiskAgent.analyze(context)                    │
  │  ─────────────────────────────────                      │
  │  Input:  RSI, EMA ratio, momentum_z, regime_feature     │
  │  Logic:                                                 │
  │    if regime_feature > 1.5: high_volatility            │
  │    if ema_ratio > 1.05 and long: bullish alignment      │
  │    technical_score = alignment / 2.0                    │
  │  Output:                                                │
  │    volatility_regime: normal / high / low_volatility    │
  │    technical_bias:    bullish / bearish / neutral        │
  │    agent_score:       0.0 – 1.0                        │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  Hybrid Score Calculation
  ┌─────────────────────────────────────────────────────────┐
  │  hybrid = (0.50 × raw_model_score                       │
  │           + 0.30 × signal_agent_score                   │
  │           + 0.20 × technical_agent_score)               │
  │                                                         │
  │  Political overlay:                                      │
  │    CRITICAL → hybrid = 0.0                             │
  │    HIGH     → hybrid × 0.5                             │
  │    MEDIUM   → hybrid × 0.8 (no change in current impl) │
  │    LOW      → hybrid unchanged                          │
  │                                                         │
  │  Drift overlay:                                         │
  │    weight = hybrid × exposure_scale                     │
  │    exposure_scale:                                      │
  │      none → 1.0  (full position)                       │
  │      soft → 0.6  (60% of position)                     │
  │      hard → 0.2  (20% of position)                     │
  │                                                         │
  │  Signal derivation from weight:                         │
  │    weight > 0.01  → LONG                               │
  │    weight < -0.01 → SHORT                              │
  │    else           → NEUTRAL                            │
  └─────────────────────────────────────────────────────────┘


STEP 5 — PORTFOLIO AGGREGATION
════════════════════════════════
  All 100 hybrid scores + weights
         │
         ▼
  PortfolioDecisionAgent + exposure normalization
  ┌─────────────────────────────────────────────────────────┐
  │  Sort: by raw_model_score descending                    │
  │  Exposure: gross = sum(|weight|), net = sum(weight)     │
  │  Normalize: if gross > 1.0 → scale all weights down    │
  │                                                         │
  │  Top-5 rationale build:                                 │
  │  For each of top-5 tickers:                             │
  │    - agents_approved: which agents said yes             │
  │    - agents_flagged:  which agents raised concerns       │
  │    - selection_reason: natural language paragraph        │
  │    - agent_scores: { signal, technical, raw_model }     │
  │                                                         │
  │  Bias classification:                                   │
  │    long > short × 1.5  → LONG_BIASED                   │
  │    short > long × 1.5  → SHORT_BIASED                  │
  │    else                → BALANCED                       │
  │                                                         │
  │  Output: 100-signal snapshot with full rationale        │
  └─────────────────────────────────────────────────────────┘


STEP 6 — CACHE & SERVE
═══════════════════════
  Complete snapshot result dict
         │
         ▼
  RedisCache.set_background_snapshot(result, ttl=360)
  ┌─────────────────────────────────────────────────────────┐
  │  Redis key: ms:background_snapshot:latest               │
  │  TTL: 360 seconds (6 minutes)                           │
  │  Fallback: in-memory dict if Redis is down              │
  │                                                         │
  │  Background loop: runs every 300 seconds (configurable) │
  │  asyncio.Lock: prevents concurrent snapshot runs        │
  │                                                         │
  │  On API request (/snapshot, /portfolio, /agent/explain):│
  │    cache.get("ms:background_snapshot:latest")           │
  │    → cache hit:  < 100ms response                       │
  │    → cache miss: compute live (~15s first run)          │
  │                                                         │
  │  Prediction storage (for IC stats):                     │
  │    PredictionRepository.store_predictions(records)       │
  │    PostgreSQL table: model_predictions                   │
  │    Used by: /model/ic-stats (Spearman IC computation)   │
  └─────────────────────────────────────────────────────────┘


STEP 7 — API RESPONSE
═══════════════════════
  Browser/Client request → FastAPI route handler
  ┌─────────────────────────────────────────────────────────┐
  │  Auth check (main.py + AuthMiddleware)                  │
  │  Rate limit check (Redis per IP)                        │
  │  Demo quota check (DemoTracker)                         │
  │  Cache lookup (RedisCache)                              │
  │  → JSON response to client                              │
  └─────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

```
TRAINING WORKFLOW
══════════════════

  START: python training/train_xgboost.py
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 1: Data Loading                                           │
  │  ─────────────────────                                          │
  │  MarketDataService.get_price_data_batch(100 tickers, 400 days) │
  │  PostgreSQL → DataFrame (100 tickers × 400 days = 40,000 rows) │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 2: Feature Engineering (Training Mode)                    │
  │  ──────────────────────────────────────────                     │
  │  FeatureEngineer.build_feature_pipeline(df, training=True)     │
  │  → 64 features computed per row                                 │
  │  → Forward returns computed (target variable)                   │
  │  → Schema signature computed: SHA256(feature_names)            │
  │  → Output: feature matrix X, target vector y                   │
  │                                                                 │
  │  Input shape:  (40,000, raw_columns)                           │
  │  Output shape: (40,000, 64) features + (40,000,) returns       │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 3: Walk-Forward Validation                                │
  │  ──────────────────────────────                                 │
  │  WalkForwardValidator.run(X, y, n_splits=5)                    │
  │                                                                 │
  │  Window structure:                                              │
  │  ┌─────────────────────────────────────────────────────┐       │
  │  │ Split 1: Train [0:160]     Val [160:200]            │       │
  │  │ Split 2: Train [0:200]     Val [200:240]            │       │
  │  │ Split 3: Train [0:240]     Val [240:280]            │       │
  │  │ Split 4: Train [0:280]     Val [280:320]            │       │
  │  │ Split 5: Train [0:320]     Val [320:400] ← final   │       │
  │  └─────────────────────────────────────────────────────┘       │
  │                                                                 │
  │  Per split: train XGBoost → compute Sharpe on validation       │
  │  Aggregate: mean Sharpe across all splits                       │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 4: Promotion Gates                                        │
  │  ───────────────────────                                        │
  │  run_evaluation.py checks:                                      │
  │                                                                 │
  │  Gate 1: Sharpe Ratio ≥ 0.25                                   │
  │    PASS → continue                                              │
  │    FAIL → model rejected, not registered                        │
  │                                                                 │
  │  Gate 2: Max Drawdown ≤ 30%                                    │
  │    PASS → continue                                              │
  │    FAIL → model rejected                                        │
  │                                                                 │
  │  Gate 3: Hit Rate ≥ 50%                                        │
  │    PASS → register model                                        │
  │    FAIL → model rejected                                        │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 5: Artifact Registration                                  │
  │  ────────────────────────────                                   │
  │  model.export_artifacts(output_dir="artifacts/xgboost/")       │
  │                                                                 │
  │  Files created:                                                 │
  │    model_xgb_20260407_215046.pkl   ← model artifact            │
  │    metadata_20260407_215046.json   ← training metadata         │
  │    latest.json                     ← pointer to latest model   │
  │                                                                 │
  │  Metadata contains:                                             │
  │    model_version:      xgb_20260407_215046                     │
  │    dataset_hash:       SHA256 of training data                  │
  │    schema_signature:   SHA256 of feature names                  │
  │    training_code_hash: SHA256 of train_xgboost.py              │
  │    feature_checksum:   SHA256 of feature list                   │
  │    best_iteration:     142                                      │
  │    sharpe_ratio:       2.085                                    │
  │    training_window:    400 days                                 │
  └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 6: Model Loading (on API restart)                         │
  │  ──────────────────────────────────────                         │
  │  get_model_loader().load()                                      │
  │                                                                 │
  │  1. Read latest.json → find model path                          │
  │  2. joblib.load(model_path) → artifact                          │
  │  3. Compute artifact_hash = SHA256(model_bytes)                 │
  │  4. Extract version from filename                               │
  │  5. Load companion metadata.json                                │
  │  6. Store as singleton (_loader_instance)                       │
  │                                                                 │
  │  All routes share the SAME loaded instance via get_model_loader │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Inference Pipeline

```
BACKGROUND SNAPSHOT LOOP (every 300 seconds)
═════════════════════════════════════════════

  _background_snapshot_loop() — asyncio task started at boot
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Wait 30 seconds (boot delay)                                   │
  │                                                                 │
  │  Loop:                                                          │
  │    if _snapshot_lock.locked():                                  │
  │      skip this cycle (previous run still in progress)          │
  │    else:                                                        │
  │      async with _snapshot_lock:                                 │
  │        pipeline.run_snapshot()                                  │
  │        cache.set_background_snapshot(result, ttl=360)           │
  │    sleep(300)                                                   │
  │                                                                 │
  │  Signals count:   100 (one per universe ticker)                 │
  │  Latency:         15–18 seconds first run, cached after        │
  │  Lock mechanism:  asyncio.Lock prevents concurrent runs         │
  └─────────────────────────────────────────────────────────────────┘


INFERENCE INPUT / OUTPUT
═════════════════════════

  INPUT to run_snapshot():
  ┌─────────────────────────────────────┐
  │  snapshot_date: "2026-04-07"        │
  │  (defaults to today if not given)   │
  └─────────────────────────────────────┘

  OUTPUT from run_snapshot():
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  meta:                                                       │
  │    model_version:    "xgb_20260407_215046"                  │
  │    drift_state:      "soft"                                  │
  │    long_signals:     66                                      │
  │    short_signals:    34                                      │
  │    avg_hybrid_score: 0.039                                   │
  │    latency_ms:       15894                                   │
  │                                                              │
  │  executive_summary:                                          │
  │    top_5_tickers:   ["MU", "SBUX", "UNH", "QCOM", "AMZN"] │
  │    top_5_rationale: [5 detailed rationale objects]          │
  │    portfolio_bias:  "LONG_BIASED"                           │
  │    gross_exposure:  1.0                                      │
  │    net_exposure:    0.164                                    │
  │                                                              │
  │  snapshot:                                                   │
  │    drift: { severity_score:7, state:"soft", scale:0.6 }    │
  │    signals: [100 objects]                                    │
  │      each: { ticker, date, raw_score, hybrid_score, weight }│
  │                                                              │
  │  _signal_details: { ticker → agent outputs } (internal)     │
  │  _political: { political risk output } (internal)           │
  └──────────────────────────────────────────────────────────────┘


TOP-5 RATIONALE OBJECT (full structure):
══════════════════════════════════════════
  ┌──────────────────────────────────────────────────────────────┐
  │  rank:             1                                         │
  │  ticker:           "MU"                                      │
  │  signal:           "LONG"                                    │
  │  hybrid_score:     0.3971                                    │
  │  raw_model_score:  1.4201                                    │
  │  weight:           0.2382                                    │
  │  confidence:       0.530  (53%)                              │
  │  risk_level:       "low"                                     │
  │  governance_score: 26     (out of 100)                       │
  │  volatility_regime:"normal"                                  │
  │  technical_bias:   "bearish"                                 │
  │  drift_context:    "soft"                                    │
  │  political_context:"HIGH"                                    │
  │  agent_scores:                                               │
  │    signal_agent:   0.0                                       │
  │    technical_agent:0.4204                                    │
  │    raw_model:      1.4201                                    │
  │  agents_approved:  ["TechnicalRiskAgent"]                    │
  │  agents_flagged:   ["SignalAgent (score=0.00, risk=low)",    │
  │                     "PoliticalRiskAgent (label=HIGH)"]       │
  │  warnings:         ["Drift detected: soft",                  │
  │                     "Soft drift — signal quality reduced"]   │
  │  selection_reason: "MU ranked #1 with hybrid consensus       │
  │                     score 0.3971 (raw model: 1.4201).        │
  │                     Signal: NEUTRAL | Confidence: 53.0% |    │
  │                     Risk level: low. Technical bias is       │
  │                     bearish with normal volatility regime.   │
  │                     Drift state is soft — position weight    │
  │                     scaled down by exposure_scale..."        │
  └──────────────────────────────────────────────────────────────┘
```

---

## Agent Decision System

```
4-AGENT DECISION PIPELINE
══════════════════════════

  Each ticker goes through all 4 agents independently.
  Agents cannot block each other — they vote asynchronously.

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  AGENT 1: SignalAgent                weight: 0.50           │
  │  ────────────────────────────────────────────────           │
  │                                                              │
  │  INPUT:                                                      │
  │    raw_model_score  → XGBoost output (-∞ to +∞)            │
  │    rsi_14           → momentum oscillator (0-100)           │
  │    ema_ratio        → price / EMA (1.0 = at EMA)           │
  │    momentum_20_z    → 20-day momentum z-score               │
  │    volatility       → historical volatility                 │
  │    drift_state      → none / soft / hard                    │
  │    political_label  → LOW / MEDIUM / HIGH / CRITICAL        │
  │                                                              │
  │  PROCESSING:                                                 │
  │    confidence = |score| / 2.0  (capped at 1.0)             │
  │    if drift ∈ {soft, hard}: confidence × 0.75              │
  │    if political == CRITICAL: signal forced NEUTRAL           │
  │    if volatility high: confidence adjusted down             │
  │    governance = int(agent_score × 100)  → 0-100 scale      │
  │    trade_approved = signal ≠ NEUTRAL                        │
  │                    AND confidence > threshold               │
  │                    AND drift_flag == False                   │
  │                                                              │
  │  OUTPUT:                                                     │
  │    signal:           LONG / SHORT / NEUTRAL                  │
  │    confidence:       0.0 – 1.0                              │
  │    risk_level:       low / moderate / high / elevated        │
  │    governance_score: 0 – 100                                 │
  │    agent_score:      0.0 – 1.0                              │
  │    warnings:         ["Drift detected: soft", ...]          │
  │    explanation:      "LONG | score=1.42 | conf=0.53 |       │
  │                       tech=0.00 | risk=low"                  │
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  AGENT 2: TechnicalRiskAgent         weight: 0.20           │
  │  ──────────────────────────────────────────────────         │
  │                                                              │
  │  INPUT:                                                      │
  │    rsi_14           → RSI oscillator                        │
  │    ema_ratio        → EMA structure                         │
  │    momentum_20_z    → normalized momentum                   │
  │    regime_feature   → volatility regime indicator           │
  │    signal_direction → from SignalAgent                      │
  │                                                              │
  │  PROCESSING:                                                 │
  │    if regime > 1.5:  high_volatility                        │
  │    if regime < -0.5: low_volatility                         │
  │    else:             normal                                  │
  │                                                              │
  │    alignment = 0                                             │
  │    if LONG and momentum_z > 0: alignment += 1               │
  │    if LONG and ema_ratio > 1.05: alignment += 1             │
  │    technical_score = alignment / 2.0                        │
  │                                                              │
  │    if LONG and RSI > 70: warn "RSI overbought"             │
  │    if SHORT and RSI < 30: warn "RSI oversold"              │
  │                                                              │
  │  OUTPUT:                                                     │
  │    volatility_regime: normal / high_volatility / low_vol    │
  │    technical_bias:    bullish / bearish / neutral            │
  │    agent_score:       0.0 – 1.0                             │
  │    warnings:          ["Momentum contradicts LONG", ...]    │
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  AGENT 3: PoliticalRiskAgent         weight: 0.10           │
  │  ──────────────────────────────────────────────────         │
  │                                                              │
  │  INPUT:                                                      │
  │    ticker:  any ticker (country: US default)                 │
  │    Runs ONCE per snapshot (not per ticker)                   │
  │                                                              │
  │  DATA SOURCES (6-provider fallback chain):                   │
  │    1. GDELT API  (real-time global events)                  │
  │    2. NewsAPI    (financial news headlines)                  │
  │    3. GNews      (Google News aggregator)                   │
  │    4. Guardian   (UK-based global coverage)                 │
  │    5. Mediastack (multi-country headlines)                   │
  │    6. Default    (score=0.0, label=LOW if all fail)         │
  │                                                              │
  │  PROCESSING:                                                 │
  │    Fetch headlines → score sentiment → aggregate risk        │
  │    risk_score: 0.0 (no risk) → 1.0 (extreme risk)          │
  │    label thresholds:                                         │
  │      0.0 – 0.25: LOW                                        │
  │      0.25 – 0.50: MEDIUM                                    │
  │      0.50 – 0.75: HIGH                                      │
  │      0.75 – 1.00: CRITICAL                                  │
  │                                                              │
  │  HYBRID SCORE OVERLAY:                                       │
  │    CRITICAL → hybrid_score = 0.0  (position zeroed)        │
  │    HIGH     → hybrid_score × 0.5 (position halved)         │
  │    MEDIUM   → no change                                     │
  │    LOW      → no change                                     │
  │                                                              │
  │  OUTPUT:                                                     │
  │    political_risk_score:  0.528                              │
  │    political_risk_label:  "HIGH"                             │
  │    top_events:            ["event1", "event2", ...]         │
  │    source:                "newsapi"                          │
  │    gdelt_status:          "gdelt_failed_used_newsapi"        │
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  AGENT 4: PortfolioDecisionAgent     weight: 0.20           │
  │  ──────────────────────────────────────────────────         │
  │                                                              │
  │  INPUT:                                                      │
  │    all 100 snapshot_rows (with hybrid scores)                │
  │    drift_state, gross_exposure, net_exposure                 │
  │                                                              │
  │  PROCESSING:                                                 │
  │    sort by raw_model_score descending                        │
  │    compute gross = sum(|weight|)                             │
  │    compute net   = sum(weight)                               │
  │    if gross > 1.0: normalize all weights                    │
  │    classify: LONG_BIASED / SHORT_BIASED / BALANCED          │
  │    build top-5 rationale objects                            │
  │                                                              │
  │  OUTPUT:                                                     │
  │    100 ordered positions with final weights                  │
  │    top_5_rationale: 5 detailed explanation objects           │
  │    portfolio_bias, gross_exposure, net_exposure              │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘


AGENT APPROVAL LOGIC (in top-5 rationale):
════════════════════════════════════════════
  signal_approved    = signal_agent_score > 0.3
                       AND risk_level != "high"
  technical_approved = technical_agent_score > 0.3
  political_approved = political_label NOT IN ["HIGH", "CRITICAL"]

  agents_approved = [agents where condition is True]
  agents_flagged  = [agents where condition is False + reason]
```

---

## Drift Detection System

```
DRIFT DETECTION — COMPLETE FLOW
═════════════════════════════════

  Runs during every snapshot inference call.

  INPUT:
  ┌─────────────────────────────────────────┐
  │  current_dataset: DataFrame (100 × 64) │
  │  (feature matrix from today's data)     │
  └─────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  DriftDetector._load_verified_baseline()                        │
  │  ─────────────────────────────────────                          │
  │  Loads baseline from: artifacts/drift/baseline.json             │
  │  Contains: feature distributions from training period           │
  │  If no baseline: drift_detected=False, severity=0               │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Statistical Tests per Feature:                                 │
  │  ─────────────────────────────                                  │
  │                                                                 │
  │  Kolmogorov-Smirnov (KS) Test:                                 │
  │    H0: current distribution == baseline distribution            │
  │    if p_value < 0.05: feature has drifted                      │
  │                                                                 │
  │  Population Stability Index (PSI):                              │
  │    PSI = sum((actual% - expected%) × ln(actual%/expected%))    │
  │    PSI < 0.1:  no change (stable)                              │
  │    PSI 0.1-0.2: moderate change (monitor)                      │
  │    PSI > 0.2:  significant change (drift confirmed)            │
  │                                                                 │
  │  Schema Validation:                                             │
  │    SHA256(current_features) == SHA256(model_features)?          │
  │    Mismatch → schema_drift detected                             │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Severity Scoring (0 – 15):                                     │
  │  ─────────────────────────                                      │
  │                                                                 │
  │  n_drifted_features / total_features → drift_fraction          │
  │  severity_score = round(drift_fraction × 15)                   │
  │                                                                 │
  │  0:      No drift detected                                      │
  │  1–4:    Minimal drift — monitor only                           │
  │  5–9:    Soft drift — reduce position sizes                     │
  │  10–14:  Hard drift — heavily reduce positions                  │
  │  15:     Critical — consider blocking inference                 │
  │                                                                 │
  │  Current system: severity=7 → soft drift                        │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Exposure Scale Mapping:                                        │
  │  ───────────────────────                                        │
  │                                                                 │
  │  drift_state  │ exposure_scale │ Effect on positions           │
  │  ─────────────┼────────────────┼───────────────────────────── │
  │  none         │ 1.0            │ Full position size            │
  │  soft         │ 0.6            │ 60% of calculated weight      │
  │  hard         │ 0.2            │ 20% of calculated weight      │
  │                                                                 │
  │  Applied to every weight: weight = hybrid × exposure_scale     │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  API RESPONSE /drift:
  ┌─────────────────────────────────────────────────────────────────┐
  │  drift_detected:          true                                  │
  │  severity_score:          7          (0-15 integer)            │
  │  drift_confidence:        0.467                                 │
  │  drift_state:             "soft"                                │
  │  exposure_scale:          0.6                                   │
  │  retrain_required:        false                                 │
  │  cooldown_active:         false                                 │
  │  cooldown_remaining_seconds: 0                                  │
  │  baseline_exists:         true                                  │
  │  baseline_version:        "26.3"                               │
  │  universe_size:           100                                   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Political Risk System

```
POLITICAL RISK — 6-PROVIDER FALLBACK CHAIN
════════════════════════════════════════════

  Called once per snapshot run (not per ticker)

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Provider 1 │    │   Provider 2 │    │   Provider 3 │
  │   GDELT API  │───▶│   NewsAPI    │───▶│   GNews      │
  │  (real-time  │    │  (financial  │    │  (Google     │
  │   events)    │    │   headlines) │    │   News)      │
  └──────────────┘    └──────────────┘    └──────────────┘
        fail                fail                fail
                                                  │
                                                  ▼
                            ┌──────────────┐    ┌──────────────┐
                            │   Provider 5 │    │   Provider 4 │
                            │  Mediastack  │◀───│   Guardian   │
                            │  (multi-     │    │  (UK global  │
                            │  country)    │    │   coverage)  │
                            └──────────────┘    └──────────────┘
                                  fail
                                    │
                                    ▼
                          ┌──────────────────┐
                          │   Provider 6     │
                          │   Default        │
                          │   score=0.0      │
                          │   label=LOW      │
                          └──────────────────┘

  gdelt_status field shows which provider was used:
    "ok"                        → GDELT succeeded
    "gdelt_failed_used_newsapi" → GDELT failed, NewsAPI used
    "all_failed_default"        → all providers failed


RISK LABEL THRESHOLDS:
═══════════════════════
  Score 0.00 – 0.25 → LOW      (no overlay on signals)
  Score 0.25 – 0.50 → MEDIUM   (no overlay on signals)
  Score 0.50 – 0.75 → HIGH     (hybrid_score × 0.5)
  Score 0.75 – 1.00 → CRITICAL (all signals forced NEUTRAL)
  Unavailable       → UNAVAILABLE (label only, score=0)


/agent/political-risk RESPONSE:
═════════════════════════════════
  {
    "ticker":               "AAPL",
    "political_risk_score": 0.528,
    "political_risk_label": "HIGH",
    "top_events":           ["event string 1", "event string 2", ...],
    "source":               "newsapi",
    "gdelt_status":         "gdelt_failed_used_newsapi",
    "served_from_cache":    true,
    "latency_ms":           15.1
  }

  NOTE: top_events is string[] — plain event title strings,
        NOT objects. No .title, .date, or .url properties.
```

---

## Authentication & Security

```
AUTHENTICATION ARCHITECTURE
═════════════════════════════

  Two access modes:
  ┌──────────────────────────────────────────────────────────────┐
  │  MODE 1: JWT Cookie (Browser Users)                          │
  │  ──────────────────────────────────                          │
  │  POST /auth/owner-login                                      │
  │    → validates username + bcrypt password hash               │
  │    → creates JWT signed with JWT_SECRET                      │
  │    → sets httpOnly cookie: ms_token                          │
  │    → all subsequent requests send cookie automatically       │
  │                                                              │
  │  POST /auth/demo-login                                       │
  │    → no password required                                    │
  │    → creates JWT with role="demo"                            │
  │    → sets httpOnly cookie: ms_token                          │
  │    → demo quota tracking starts immediately                  │
  │                                                              │
  │  MODE 2: API Key (External Programmatic Access)              │
  │  ─────────────────────────────────────────────               │
  │  All requests: pass X-API-KEY header                         │
  │  Value: hex string from scripts/generate_api_key.py          │
  │  Effect: treated as owner role (full access)                 │
  │  Does NOT increment demo quota                               │
  └──────────────────────────────────────────────────────────────┘


SECURITY LAYERS (5 layers):
═════════════════════════════

  Layer 1 — Network
  ─────────────────
  Production: HTTPS only (Cloud Run / Vercel enforced)
  CORS: allowed origins set via CORS_ORIGINS env var

  Layer 2 — main.py (request_context_middleware)
  ───────────────────────────────────────────────
  PUBLIC_PATHS set (no auth needed):
    /  /docs  /openapi.json  /redoc  /metrics
    /health/* /auth/* /universe /model/info /agent/agents

  All other paths:
    has_jwt = request.cookies.get("ms_token") exists
           OR Authorization: Bearer header exists
    if not has_jwt:
      check X-API-KEY header
      if API_KEY not set: programmatic access disabled
      if key doesn't match: 401 "Invalid or missing API key"

  Layer 3 — Rate Limiting (Redis per IP)
  ───────────────────────────────────────
  Key: ms:ratelimit:{client_ip}:{safe_path}
  On limit exceeded: 429 + Retry-After header
  Redis down: fails OPEN (never blocks on downtime)

  Rate limits by endpoint:
  ┌────────────────────────────┬─────────┬────────┐
  │ Endpoint                   │ Limit   │ Window │
  ├────────────────────────────┼─────────┼────────┤
  │ /auth/owner-login          │ 5 req   │ 60s    │
  │ /auth/demo-login           │ 10 req  │ 60s    │
  │ /snapshot                  │ 10 req  │ 60s    │
  │ /predict/live-snapshot     │ 10 req  │ 60s    │
  │ /agent/explain             │ 20 req  │ 60s    │
  │ /agent/political-risk      │ 20 req  │ 60s    │
  │ /performance               │ 20 req  │ 60s    │
  │ /health/live               │ 60 req  │ 60s    │
  │ /health/ready              │ 60 req  │ 60s    │
  │ All other paths            │ 60 req  │ 60s    │
  └────────────────────────────┴─────────┴────────┘

  Layer 4 — AuthMiddleware v2.4 (role enforcement)
  ─────────────────────────────────────────────────
  Check order:
  1. Free paths (FREE_PATHS set) → pass through
  2. Valid X-API-KEY → set role=owner, pass through
  3. Owner-only paths (OWNER_ONLY_PREFIXES):
       /admin/* /model/ic-stats /model/diagnostics
       → demo user: 403
       → no token:  401
  4. JWT role=owner → full access
  5. JWT role=demo  → demo quota check
  6. No token       → 401

  Layer 5 — Demo Quota (DemoTracker)
  ────────────────────────────────────
  Redis key: demo:usage:{fingerprint}:{feature}
  Fingerprint: SHA256(IP + User-Agent)[:32]
  TTL: 7 days (auto-reset after 1 week)

  Feature groups:
  ┌─────────────────┬──────────────────────────────────────────┐
  │ Feature Key     │ Endpoints                                │
  ├─────────────────┼──────────────────────────────────────────┤
  │ snapshot        │ /snapshot, /predict/live-snapshot        │
  │ portfolio       │ /portfolio                               │
  │ drift           │ /drift                                   │
  │ performance     │ /performance                             │
  │ agent           │ /agent/explain, /agent/political-risk    │
  │ signals         │ /equity/*, /model/feature-importance     │
  └─────────────────┴──────────────────────────────────────────┘

  When limit reached (default: 10 per feature):
  HTTP 200 response with body:
  {
    "demo_locked": true,
    "feature": "portfolio",
    "reset_in_seconds": 604800,
    "usage": { ...full usage summary... }
  }
```

---

## Demo Quota System

```
DEMO QUOTA FLOW — COMPLETE
═══════════════════════════

  Browser → POST /auth/demo-login
       │
       ▼
  JWT cookie issued (ms_token, role=demo)
       │
       ▼
  Browser → GET /portfolio (with cookie)
       │
       ▼
  DemoTracker.build_fingerprint(ip, user_agent)
       → SHA256(IP + UA)[:32] → unique fingerprint
       │
       ▼
  DemoTracker.is_locked(fingerprint, "portfolio")
       → Redis: GET demo:usage:{fingerprint}:portfolio
       → value = 9 (under limit of 10)
       → NOT locked → continue
       │
       ▼
  DemoTracker.increment(fingerprint, "portfolio")
       → Redis: INCR demo:usage:{fingerprint}:portfolio
       → Redis: EXPIRE ... 604800  (7 days TTL)
       → value now = 10
       │
       ▼
  Route handler returns portfolio data (200 OK)
       │
       ▼
  Browser → GET /portfolio again
       │
       ▼
  DemoTracker.is_locked(fingerprint, "portfolio")
       → Redis: GET demo:usage:{fingerprint}:portfolio
       → value = 10 (AT limit)
       → LOCKED → return demo_locked response
       │
       ▼
  HTTP 200 with body:
  {
    "demo_locked": true,
    "feature": "portfolio",
    "reset_in_seconds": 604800,
    "message": "Demo limit reached for 'portfolio'. Resets in 604800s.",
    "usage": {
      "features": {
        "snapshot":    {"used":0,  "limit":10, "remaining":10, "locked":false},
        "portfolio":   {"used":10, "limit":10, "remaining":0,  "locked":true},
        "drift":       {"used":0,  "limit":10, "remaining":10, "locked":false},
        "performance": {"used":0,  "limit":10, "remaining":10, "locked":false},
        "agent":       {"used":0,  "limit":10, "remaining":10, "locked":false},
        "signals":     {"used":0,  "limit":10, "remaining":10, "locked":false}
      },
      "fully_locked": false,
      "reset_in_seconds": 604800,
      "limit_per_feature": 10
    }
  }

  NOTE: Each feature has its OWN quota.
        Exhausting portfolio does NOT lock snapshot.
        fully_locked = true only when ALL features exhausted.
```

---

## API Reference

### Complete Endpoint Table

```
PUBLIC ENDPOINTS (no auth required)
═════════════════════════════════════
  GET  /                           Root info + system status
  GET  /health/live                Docker liveness probe → 200 OK
  GET  /health/ready               Readiness: model + redis + db
  GET  /health/db                  Database connectivity
  GET  /health/model               Model artifact integrity
  GET  /universe                   100-ticker S&P 500 list
  GET  /model/info                 Model version + hashes + feature count
  GET  /agent/agents               4-agent descriptions + weights
  GET  /docs                       Swagger UI (interactive API docs)
  GET  /metrics                    Prometheus metrics export

AUTH ENDPOINTS (public — no cookie needed)
═══════════════════════════════════════════
  POST /auth/owner-login           Admin login → JWT cookie
  POST /auth/demo-login            Demo access → JWT cookie
  GET  /auth/me                    Current session + quota
  POST /auth/logout                Clear JWT cookie

PROTECTED ENDPOINTS (owner or demo)
═════════════════════════════════════
  POST /snapshot                   Full inference → 100 signals
  GET  /portfolio                  Portfolio positions + health
  GET  /drift                      Model drift metrics
  GET  /performance?days=N         Strategy performance metrics
  GET  /agent/explain?ticker=X     Per-ticker signal explanation
  GET  /agent/political-risk?ticker=X  Geopolitical risk
  GET  /equity/{ticker}            Latest OHLCV + returns
  GET  /equity/{ticker}/history    Historical price data

OWNER-ONLY ENDPOINTS (JWT role=owner OR X-API-KEY)
════════════════════════════════════════════════════
  GET  /model/ic-stats?days=30     IC statistics (Spearman)
  GET  /model/feature-importance   Top features by XGBoost gain
  GET  /model/diagnostics          Full model checksums
  POST /admin/sync                 Trigger manual data sync
```

### Key Response Schemas

```
GET /health/ready
─────────────────
{
  "ready": true,
  "models_loaded": true,
  "redis_connected": true,
  "db_connected": true,
  "data_synced": true,
  "drift_baseline_loaded": true,
  "model_version": "xgb_20260407_215046",
  "uptime_seconds": 3600
}

GET /agent/explain?ticker=MU (top-5 ticker)
────────────────────────────────────────────
{
  "success": true,
  "data": {
    "ticker": "MU",
    "snapshot_date": "2026-03-27",
    "signal": "LONG",
    "raw_model_score": 1.4201,
    "hybrid_consensus_score": 0.3971,
    "weight": 0.2382,
    "confidence_numeric": null,       ← null for most tickers
    "governance_score": 26,           ← 0-100 scale
    "risk_level": "low",
    "volatility_regime": "normal",
    "technical_bias": "bearish",
    "drift_state": "soft",
    "warnings": ["Drift detected: soft"],
    "explanation": "NEUTRAL | score=1.42 | conf=0.53 | tech=0.00 | risk=low",
    "llm": null,
    "rank": 1,                        ← null if not in top-5
    "in_top_5": true,
    "agents_approved": ["TechnicalRiskAgent"],
    "agents_flagged": ["SignalAgent (score=0.00, risk=low)", "PoliticalRiskAgent (label=HIGH)"],
    "selection_reason": "MU ranked #1 with hybrid consensus score 0.3971...",
    "agent_scores": {
      "signal_agent": 0.0,
      "technical_agent": 0.4204,
      "raw_model": 1.4201
    }
  }
}

GET /agent/explain?ticker=AAPL (non-top-5 ticker)
──────────────────────────────────────────────────
{
  "data": {
    "ticker": "AAPL",
    "signal": "LONG",
    "raw_model_score": 0.6326,
    "rank": null,
    "in_top_5": false,
    "agents_approved": [],            ← empty for non-top-5
    "agents_flagged": [],             ← empty for non-top-5
    "selection_reason": "",           ← empty for non-top-5
    "agent_scores": {}                ← empty for non-top-5
  }
}

GET /performance?days=252
──────────────────────────
{
  "metrics": {
    "sharpe_ratio":       2.085,      ← annualized Sharpe
    "sortino_ratio":      3.110,      ← downside-adjusted Sharpe
    "calmar_ratio":       3.544,      ← return / max drawdown
    "cumulative_return":  0.177,      ← 17.7% total return
    "max_drawdown":      -0.061,      ← -6.1% peak-to-trough
    "max_drawdown_duration": 23,      ← days underwater
    "hit_rate":           0.564,      ← 56.4% positive return days
    "annual_return":      0.215,      ← 21.5% annualized
    "annual_volatility":  0.096,      ← 9.6% annualized vol
    "turnover":           0.0,        ← portfolio turnover
    "skewness":          -0.245,      ← return distribution skew
    "downside_deviation": 0.064,      ← downside vol only
    "tracking_error":     null,       ← no benchmark set
    "beta":               null,       ← no benchmark set
    "information_ratio":  null        ← no benchmark set
  }
}

GET /drift
───────────
{
  "drift_detected": true,
  "severity_score": 7,               ← 0-15 integer (NOT percentage)
  "drift_confidence": 0.467,
  "drift_state": "soft",
  "exposure_scale": 0.6,
  "retrain_required": false,
  "cooldown_active": false,
  "cooldown_remaining_seconds": 0,
  "baseline_exists": true,
  "baseline_version": "26.3",
  "baseline_model_version": "xgb_20260329_032805",
  "universe_size": 100
}

GET /equity/AAPL/history?days=90
──────────────────────────────────
{
  "ticker": "AAPL",
  "days_requested": 90,
  "rows_returned": 80,               ← ~80 trading days in 90 calendar days
  "data_source": "postgresql",
  "history": [                       ← key is "history" NOT "prices"
    {
      "date": "2025-12-08",
      "open": 278.13,
      "high": 279.67,
      "low": 276.15,
      "close": 277.63,
      "volume": 38211800
    }
  ]
}
```

---

## Database Schema

```
POSTGRESQL TABLES
══════════════════

  Table: ohlcv_daily
  ───────────────────
  id          SERIAL PRIMARY KEY
  ticker      VARCHAR(10)   NOT NULL
  date        DATE          NOT NULL
  open        FLOAT
  high        FLOAT
  low         FLOAT
  close       FLOAT
  volume      BIGINT
  source      VARCHAR(20)   DEFAULT 'yfinance'
  UNIQUE(ticker, date)  ← constraint: uq_ohlcv_ticker_date
  INDEX: (ticker, date) ← for fast per-ticker queries

  Rows: ~40,000 (100 tickers × 400 days)


  Table: computed_features
  ─────────────────────────
  id               SERIAL PRIMARY KEY
  ticker           VARCHAR(10)
  date             DATE
  feature_version  VARCHAR(64)   ← SHA256 of schema signature
  feature_data     JSONB         ← 64 feature values as JSON
  UNIQUE(ticker, date, feature_version)
  INDEX: (ticker, date, feature_version)

  Used for: caching computed features (optional path)


  Table: model_predictions
  ─────────────────────────
  id               SERIAL PRIMARY KEY
  ticker           VARCHAR(10)
  date             DATE
  model_version    VARCHAR(50)
  schema_signature VARCHAR(64)
  raw_model_score  FLOAT
  hybrid_score     FLOAT
  weight           FLOAT
  signal           VARCHAR(10)   ← LONG / SHORT / NEUTRAL
  drift_state      VARCHAR(20)
  predicted_at     TIMESTAMP     DEFAULT NOW()
  UNIQUE(ticker, date, model_version)

  Used for: /model/ic-stats — Spearman IC computation
  Stored by: pipeline.run_snapshot() when STORE_PREDICTIONS=1
```

---

## Feature Engineering

```
FEATURE PIPELINE — INPUT TO OUTPUT
════════════════════════════════════

  INPUT:
  ┌─────────────────────────────────────────────────────────┐
  │  Raw OHLCV DataFrame                                    │
  │  Columns: ticker, date, open, high, low, close, volume  │
  │  Rows:    100 tickers × 400 days = ~40,000 rows         │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  FeatureEngineer.build_feature_pipeline(df, training=False)
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  COMPUTED FEATURES (64 total)                           │
  │                                                         │
  │  Returns (price momentum):                              │
  │    return_1d      = close.pct_change(1)                 │
  │    return_5d      = close.pct_change(5)                 │
  │    return_20d     = close.pct_change(20)                │
  │    log_return     = log(close / close.shift(1))         │
  │                                                         │
  │  Momentum (rate of change):                             │
  │    momentum_5     = close - close.shift(5)              │
  │    momentum_20    = close - close.shift(20)             │
  │    momentum_20_z  = (mom20 - mean) / std                │
  │    reversal_5_rank= percentile rank of 5-day reversal   │
  │                                                         │
  │  RSI (Relative Strength Index):                         │
  │    rsi_14         = 14-period RSI (0-100)               │
  │    > 70 = overbought, < 30 = oversold                   │
  │                                                         │
  │  Moving averages:                                       │
  │    ema_20         = 20-day exponential MA               │
  │    ema_50         = 50-day exponential MA               │
  │    sma_20         = 20-day simple MA                    │
  │    ema_20_slope   = rate of change of ema_20            │
  │                                                         │
  │  EMA structure:                                         │
  │    ema_ratio      = close / ema_20                      │
  │    trend_strength = ema_20 / ema_50                     │
  │    price_above_ema= 1 if close > ema_20 else 0         │
  │    golden_cross   = 1 if ema_20 > ema_50 else 0        │
  │    ema_slope      = ema_20 direction                    │
  │                                                         │
  │  Volatility:                                            │
  │    volatility_20  = std(return_1d, 20 days)            │
  │    atr_14         = 14-period Average True Range        │
  │    regime_feature = volatility rank vs history          │
  │    vol_ratio      = current_vol / long_term_vol         │
  │    high_low_range = (high - low) / close                │
  │                                                         │
  │  Cross-sectional (ranked across universe):              │
  │    close_rank     = percentile rank of close vs peers   │
  │    price_rank     = cross-sectional price rank          │
  │    volume_rank    = cross-sectional volume rank         │
  │    return_rank    = cross-sectional return rank         │
  │    momentum_rank  = cross-sectional momentum rank       │
  │    relative_strength = ticker return / universe mean    │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  Schema Validation:
    feature_schema.py defines MODEL_FEATURES (64 names)
    Any missing feature → filled with 0.0
    SHA256(feature_names) → schema_signature
    If signature != model artifact signature → warning

  OUTPUT:
  ┌─────────────────────────────────────────────────────────┐
  │  DataFrame shape: (100, 64) for inference               │
  │  (one row per ticker, latest date only)                 │
  │  dtype: float32 (matches model training dtype)          │
  └─────────────────────────────────────────────────────────┘
```

---

## Model Governance

```
MODEL ARTIFACT STRUCTURE
══════════════════════════

  artifacts/xgboost/
  ├── model_xgb_20260407_215046.pkl     ← model artifact (joblib)
  ├── metadata_20260407_215046.json     ← companion metadata
  └── latest.json                       ← pointer file

  latest.json:
  { "path": "model_xgb_20260407_215046.pkl" }

  metadata JSON:
  {
    "model_version":      "xgb_20260407_215046",
    "dataset_hash":       "SHA256 of training X + y",
    "schema_signature":   "SHA256 of feature name list",
    "training_code_hash": "SHA256 of train_xgboost.py",
    "feature_checksum":   "SHA256 of feature columns",
    "best_iteration":     142,
    "sharpe_ratio":       2.085,
    "training_window":    400,
    "trained_at":         "2026-04-07T21:50:46Z"
  }


PROMOTION GATES:
════════════════
  Model trained → Walk-forward validation → Gates:
  ┌─────────────────────────────────────────────────┐
  │  Gate 1: Sharpe ≥ 0.25                         │
  │  Gate 2: Max Drawdown ≤ 30%                    │
  │  Gate 3: Hit Rate ≥ 50%                        │
  │                                                 │
  │  ALL gates must pass → model registered        │
  │  ANY gate fails → model rejected, not saved    │
  └─────────────────────────────────────────────────┘


IC STATISTICS (Information Coefficient):
════════════════════════════════════════
  Measures predictive quality of model signals.
  Formula: Spearman correlation between:
    - raw_model_score (prediction)
    - next_day_forward_return (actual)

  Computed over rolling 30-day window.
  Requires STORE_PREDICTIONS=1 in .env.

  Interpretation:
  ┌──────────────────────────────────────────────────┐
  │  IC > 0.08  → Strong   — meaningful alpha        │
  │  IC 0.04–0.08 → Moderate — usable signal         │
  │  IC 0.02–0.04 → Weak   — marginal                │
  │  IC < 0.02  → Noise   — near random              │
  └──────────────────────────────────────────────────┘

  API: GET /model/ic-stats?days=30  (owner only)
  Response: { ic_mean, ic_std, ic_t_stat, signal_quality, daily_ic[] }
```

---

## Observability

```
PROMETHEUS METRICS (GET /metrics)
═══════════════════════════════════

  Inference counters:
    api_request_total{endpoint}     ← total requests per endpoint
    api_error_total{endpoint}       ← errors per endpoint
    api_latency_seconds{endpoint}   ← request duration histogram

  System metrics (auto-exported by prometheus_client):
    python_gc_objects_collected_total
    process_virtual_memory_bytes
    process_resident_memory_bytes
    process_cpu_seconds_total

  Connect Grafana datasource to: http://localhost:8000/metrics


STRUCTURED LOGGING:
════════════════════
  All logs use JSON format in production (LOG_ENV=production)
  Human-readable format in development (LOG_ENV=development)

  Key log events:
    Startup complete | time=Xs | db=True | redis=True | model=True
    Background snapshot cached | signals=100 | model=xgb_* | took=Xs
    Demo locked | fingerprint=* | feature=portfolio | reset_in=604800s
    Rate limit exceeded | ip=* | path=* | count=* | limit=*
    Drift detection | severity=7 | state=soft | exposure=0.6
    Daily sync complete | synced=100 | skipped=0 | errors=0
    API key configured | programmatic access enabled
    Predictions stored | date=* | count=100


HEALTH PROBES:
═══════════════
  GET /health/live   → Always 200 (liveness — is process up?)
  GET /health/ready  → 200 if models loaded + db connected
                       (readiness — can it serve traffic?)
  GET /health/db     → DB connectivity + latency ms
  GET /health/model  → model loaded + version + artifact hash
```

---

## Project Structure

```
MarketSentinel/
│
├── app/                           ← FastAPI inference control plane
│   ├── api/routes/
│   │   ├── agent.py               ← /agent/* (explain, political-risk, agents)
│   │   ├── auth.py                ← /auth/* (login, logout, me)
│   │   ├── drift.py               ← /drift
│   │   ├── equity.py              ← /equity/*
│   │   ├── health.py              ← /health/*
│   │   ├── model_info.py          ← /model/* (info, features, ic-stats)
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
│   │   ├── model_loader.py        ← Singleton model loader v2.8
│   │   └── pipeline.py            ← InferencePipeline v5.9
│   ├── monitoring/metrics.py      ← Prometheus exporters
│   └── main.py                    ← FastAPI app v3.7
│
├── core/                          ← Business domain (framework-agnostic)
│   ├── agent/
│   │   ├── base_agent.py          ← Abstract agent interface
│   │   ├── signal_agent.py        ← XGBoost signal interpreter
│   │   ├── technical_risk_agent.py
│   │   ├── portfolio_decision_agent.py
│   │   └── political_risk_agent.py ← GDELT 6-provider chain
│   ├── analytics/performance_engine.py ← Sharpe/Sortino/Calmar/IC
│   ├── artifacts/
│   │   ├── metadata_manager.py
│   │   └── model_registry.py
│   ├── data/
│   │   ├── data_fetcher.py
│   │   ├── data_sync.py           ← Daily 18:30 sync scheduler
│   │   ├── market_data_service.py
│   │   └── providers/market/      ← Yahoo + TwelveData + fallbacks
│   ├── db/
│   │   ├── engine.py              ← SQLAlchemy pool (size=10)
│   │   ├── models.py              ← ORM: OHLCV, Features, Predictions
│   │   └── repository.py          ← Data access layer v2.1
│   ├── features/feature_engineering.py ← 64-feature pipeline
│   ├── indicators/technical_indicators.py
│   ├── market/universe.py         ← 100-ticker universe
│   ├── models/xgboost.py          ← XGBoost wrapper
│   ├── monitoring/
│   │   ├── drift_detector.py      ← KS + PSI drift scoring
│   │   ├── market_regime_detector.py
│   │   └── retrain_trigger.py
│   ├── schema/feature_schema.py   ← 64-feature contract + SHA256
│   └── time/market_time.py
│
├── training/
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   ├── regime.py
│   │   └── walk_forward.py        ← 5-fold rolling validation
│   ├── pipelines/train_pipeline.py
│   ├── train_xgboost.py           ← Main training script
│   ├── evaluate.py
│   └── run_evaluation.py          ← Promotion gate executor
│
├── scripts/
│   ├── generate_api_key.py        ← 32-byte cryptographic hex key
│   └── generate_owner_hash.py     ← bcrypt hash generator
│
├── tests/                         ← 115 passing tests
│   ├── conftest.py
│   ├── test_api_signal_explanation.py
│   ├── test_auth.py
│   ├── test_demo_tracker.py
│   ├── test_drift_detector.py
│   ├── test_evaluate.py
│   ├── test_feature_engineering.py
│   ├── test_metadata_integrity.py
│   ├── test_middleware.py
│   ├── test_pipeline_filter.py
│   ├── test_redis_cache.py
│   ├── test_rsi_extreme_cases.py
│   ├── test_schema_signature.py
│   ├── test_signal_agent.py
│   ├── test_technical_indicators.py
│   ├── test_training_end_to_end.py
│   ├── test_walk_forward.py
│   └── test_xgboost_regressor.py
│
├── artifacts/xgboost/             ← Model artifacts (gitignored)
├── config/universe.json           ← 100 S&P 500 tickers
├── .github/workflows/ci.yml       ← GitHub Actions CI
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .flake8
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Python 3.10
- 4 GB RAM minimum (model + data)
- Internet access (for data sync + political risk)

### Step 1 — Clone

```bash
git clone https://github.com/muhammedshihab1001/MarketSentinel.git
cd MarketSentinel
```

### Step 2 — Generate Credentials

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements/base.txt
pip install -r requirements/inference.txt
pip install -r requirements/ci.txt
pip install -r requirements/training.txt

# Generate API key (paste directly in .env — no quotes)
python scripts/generate_api_key.py

# Generate password hash (paste with double quotes in .env)
python scripts/generate_owner_hash.py
```

### Step 3 — Configure .env

```env
# Database
POSTGRES_USER=sentinel
POSTGRES_PASSWORD=sentinel
POSTGRES_DB=marketsentinel
DATABASE_URL=postgresql+psycopg2://sentinel:sentinel@postgres:5432/marketsentinel

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Security
API_KEY=abc123your64hexcharkey           ← no quotes
OWNER_USERNAME=shihab
OWNER_PASSWORD_HASH="$2b$12$hashhere"    ← with double quotes
JWT_SECRET=your-32-char-secret-here

# News
NEWSAPI_KEY=your_newsapi_key

# Optional
STORE_PREDICTIONS=1
LLM_ENABLED=false
CORS_ORIGINS=http://localhost:5173
```

### Step 4 — Start

```bash
docker-compose up -d
```

### Step 5 — Verify

```bash
curl http://localhost:8000/health/ready
# {"ready":true,"models_loaded":true,"redis_connected":true,"db_connected":true}

# Wait ~90s for first background snapshot, then:
curl -X POST http://localhost:8000/auth/demo-login \
  -H "Content-Type: application/json" -d "{}" -c cookies.txt

curl -X POST http://localhost:8000/snapshot -b cookies.txt
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full PostgreSQL connection string |
| `REDIS_HOST` | Yes | `redis` | Redis hostname (use `redis` in Docker) |
| `REDIS_PORT` | No | `6379` | Redis port |
| `API_KEY` | Yes | — | External API key — no quotes in .env |
| `OWNER_USERNAME` | Yes | — | Admin username |
| `OWNER_PASSWORD_HASH` | Yes | — | bcrypt hash — with double quotes in .env |
| `JWT_SECRET` | Yes | — | JWT signing secret (minimum 32 characters) |
| `NEWSAPI_KEY` | No | — | NewsAPI key for political risk fallback |
| `STORE_PREDICTIONS` | No | `1` | Store predictions to DB for IC stats |
| `DEMO_REQUESTS_PER_FEATURE` | No | `10` | Demo quota per feature per week |
| `LLM_ENABLED` | No | `false` | Enable LLM-powered signal explanations |
| `SKIP_DATA_SYNC` | No | `0` | Skip data sync on startup |
| `CORS_ORIGINS` | No | `localhost:5173` | Comma-separated allowed origins |
| `SNAPSHOT_PRECOMPUTE_INTERVAL` | No | `300` | Background snapshot frequency (seconds) |
| `INFERENCE_LOOKBACK_DAYS` | No | `400` | Price history window for features |
| `DATA_STALENESS_HOURS` | No | `24` | Hours before data considered stale |
| `LOG_ENV` | No | `development` | `development` or `production` |
| `APP_VERSION` | No | `5.0.0` | API version string |

---

## CI Pipeline

```yaml
# .github/workflows/ci.yml
# Triggers: push to feature/*, pull_request to develop or main

Jobs:
  1. Checkout code
  2. Set up Python 3.10
  3. Install dependencies (pip install -r requirements.txt)
  4. Start PostgreSQL 14 (GitHub Actions service container)
     POSTGRES_USER: msuser
     POSTGRES_PASSWORD: mspass123
     POSTGRES_DB: mstest
  5. Run flake8 linting
     - max-line-length: 100
     - ignored: E501, W503, E203, E402, E221, E272, E128
     - threshold: 85% of files must be clean
  6. Run pytest
     - 115 tests must pass
     - 1 skip allowed (Redis-dependent test in CI)
     - timeout: 60 seconds per test
  7. Verify schema signature integrity

Status: All checks pass on every push to develop/main
```

---

## Performance Benchmarks

| Operation | Cold (first run) | Warm (Redis cached) |
|---|---|---|
| Full snapshot (100 tickers) | 15–18 seconds | < 100ms |
| Feature engineering (100 × 400 days) | ~12 seconds | N/A |
| Agent explain (any ticker) | 100–400ms | < 50ms |
| Portfolio | < 50ms | < 20ms |
| Drift | < 100ms | < 50ms |
| Performance (252 days) | 2–5 seconds | N/A |
| Political risk (GDELT) | 3–8 seconds | < 20ms |
| Political risk (NewsAPI fallback) | 1–3 seconds | < 20ms |

First snapshot is slow because it fetches 400 days × 100 tickers of price data from PostgreSQL, runs 64 feature computations per row, and executes the full agent pipeline. All subsequent calls read from Redis in milliseconds.

---

## Author

**Muhammed Shihab P**

Building production ML systems, MLOps platforms, and decision intelligence engines.

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

Copyright (c) 2026 Muhammed Shihab P