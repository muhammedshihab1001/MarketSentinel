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
2. [Live Dashboard](#live-dashboard)
3. [System Architecture](#system-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Feature Engineering](#feature-engineering)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Hybrid Score Formula](#hybrid-score-formula)
9. [Agent Decision System](#agent-decision-system)
10. [Drift Detection](#drift-detection)
11. [Political Risk System](#political-risk-system)
12. [Security Architecture](#security-architecture)
13. [Demo Quota System](#demo-quota-system)
14. [Model Governance](#model-governance)
15. [Observability](#observability)
16. [API Reference](#api-reference)
17. [Database Schema](#database-schema)
18. [Project Structure](#project-structure)
19. [Quick Start](#quick-start)
20. [Environment Variables](#environment-variables)
21. [CI Pipeline](#ci-pipeline)
22. [Performance Benchmarks](#performance-benchmarks)
23. [Author](#author)

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

## Live Dashboard

The executive dashboard shows live system state: signal distribution, equity curve, drift alerts, and the top-5 ranked tickers with their scores and portfolio weights.

![MarketSentinel Dashboard](docs/images/dashboard_ui.png)

> **66 LONG / 34 SHORT signals** across 100 S&P 500 tickers. Sharpe ratio 2.09. Hit rate 56.4%.
> Soft drift detected — positions automatically scaled to 60% of calculated weights.

---

## System Architecture

MarketSentinel is structured in four layers: Data Layer (ingestion into PostgreSQL), ML Pipeline (feature engineering + XGBoost), Inference Engine (background snapshot loop + 4-agent pipeline + Redis cache), and API + Frontend (FastAPI + React/Vercel).

![Platform Architecture](docs/images/arch_overview.png)

---

## Data Pipeline

Market data is ingested daily via a 2-provider fallback chain. Yahoo Finance (primary) → TwelveData (fallback). ThreadPoolExecutor with 4 parallel workers reduces 100-ticker sync time from ~3,000ms to ~400ms.

![Data Sync Pipeline](docs/images/data_sync.png)

- Delta sync: fetches only missing rows per ticker with 3-day overlap for weekends/holidays
- Rows older than 760 days auto-deleted after each sync to bound database size
- Schedule: 18:30 weekdays; on startup if data is stale (> 24 hours)

---

## Feature Engineering

The FeatureEngineer pipeline computes 64 features per ticker per date in three categories.

![Feature Engineering Pipeline](docs/images/features.png)

| Category | Count | Description |
|---|---|---|
| Core features | 24 | Returns at multiple lags, momentum composites, RSI, EMA ratios, volatility, Amihud illiquidity, regime_multiplier (BULL=1.2 / SIDEWAYS=1.0 / BEAR=0.6 / CRISIS=0.3) |
| Cross-sectional z-scores | 20 | Each core feature normalised relative to the 100-ticker universe: `(feature - universe_mean) / universe_std` |
| Cross-sectional ranks | 20 | Percentile rank within universe — scale-invariant across market regimes |

Schema enforcement: SHA256 signature of the feature name list. Mismatch triggers warning and retraining recommendation.

**Feature caching:** Computed features cached in PostgreSQL `computed_features` table. Cold: ~35s → Warm: ~2s.

---

## Training Pipeline

Training runs in a separate Docker container (`profiles:[training]`) — does not start with `docker compose up -d`.

![ML Training Pipeline](docs/images/training_pipeline.png)

**Walk-forward validation:** 5 expanding windows with embargo gaps (= prediction horizon, 5 days) to prevent data leakage.

**Promotion gates — all must pass:**

| Gate | Threshold |
|---|---|
| Sharpe Ratio | ≥ 0.10 on walk-forward aggregate |
| Maximum Drawdown | ≤ 60% on worst validation fold |
| Directional Hit Rate | ≥ 50% across folds |

**Artifact registry** stores 4 integrity hashes per model: `dataset_hash`, `schema_signature`, `training_code_hash`, `artifact_hash`.

```bash
# First time (sync + train + create drift baseline)
docker compose --profile training run --rm -e CREATE_BASELINE=1 training

# Retrain with existing DB data
docker compose --profile training run --rm -e SKIP_SYNC=1 training
```

**Model pointer file** `artifacts/xgboost/production_pointer.json`:
```json
{
  "model_version": "xgb_20260407_215046",
  "model_path": "/app/artifacts/xgboost/model_xgb_20260407_215046.pkl",
  "metadata_path": "/app/artifacts/xgboost/metadata_xgb_20260407_215046.json",
  "updated_at": "2026-04-07T21:50:46Z"
}
```

ModelLoader v2.9 load order: `production_pointer.json` → `latest.json` (alias) → directory scan (warning logged).

---

## Inference Pipeline

The background snapshot loop runs every 300 seconds, computes the full 100-ticker inference, and caches the result in Redis. All API endpoints serve from cache — delivering sub-100ms responses.

![Background Inference & Redis Caching](docs/images/caching.png)

- `asyncio.Lock` prevents concurrent snapshot runs — one computation at a time
- Redis TTL: 360 seconds. Falls back to in-memory dict if Redis is unavailable
- `filter_latest_per_ticker()` reduces the 27,400-row frame to 100 rows before inference (v5.8 fix)
- Boot delay: 30 seconds before first snapshot run

---

## Hybrid Score Formula

For each ticker the final trading signal weight is a 3-component weighted formula modified by two overlays.

![Hybrid Score Formula](docs/images/hybrid_score.png)

```
hybrid = 0.50 × raw_model_score       ← XGBoost output
       + 0.30 × signal_agent.score    ← "score" key in return dict
       + 0.20 × technical_agent.score

Political overlay:  HIGH → hybrid × 0.5  |  CRITICAL → hybrid = 0.0
Drift overlay:      weight = hybrid × exposure_scale
                    none=1.0  |  soft=0.6  |  hard=0.25

Signal direction:   weight > 0.01  → LONG
                    weight < -0.01 → SHORT
                    else           → NEUTRAL
```

---

## Agent Decision System

The 4-agent pipeline processes each ticker independently after XGBoost inference. Agents cannot block each other — failures return a safe empty result and the pipeline continues.

![4-Agent Pipeline](docs/images/agent_pipeline.png)

| Agent | Weight | Function |
|---|---|---|
| **SignalAgent** | 50% | Confidence, risk level, drift penalty (×0.75 on soft/hard), political override (NEUTRAL if CRITICAL) |
| **TechnicalRiskAgent** | 20% | RSI extremes, EMA alignment, momentum z-score, volatility regime |
| **PoliticalRiskAgent** | 10% | 6-provider news chain. Runs ONCE per snapshot. Applies score overlay to all tickers |
| **PortfolioDecisionAgent** | 20% | Sector-neutral top-K (max 2 per GICS sector). Builds top-5 rationale objects |

---

## Drift Detection

The DriftDetector compares current inference distributions against the training baseline in `artifacts/drift/baseline.json`.

![Real-Time Drift Detection](docs/images/drift_detection.png)

**Statistical tests per feature:**
- **KS test:** p-value < 0.05 = drifted
- **PSI:** > 0.2 = significant drift confirmed

**Severity scale:**

| Severity | State | Exposure Scale | Action |
|---|---|---|---|
| 0–4 | None / Low | 1.0 | Monitor only |
| 5–7 | Soft Drift | 0.6 | Consider retrain |
| 8–9 | Soft + Retrain Flag | 0.6 | `retrain_required: true` |
| 10–14 | Hard Drift | 0.25 | Immediate retrain |
| 15 | Critical | 0.25 | Consider halting |

> `severity_score` is a **0–15 integer**, not a percentage. Display as `"X / 15"`.

---

## Political Risk System

The PoliticalRiskAgent fetches geopolitical headlines via a 6-provider priority-ordered fallback chain.

![Political Risk 6-Provider Chain](docs/images/political_risk.png)

| Priority | Provider | Key | Free Tier |
|---|---|---|---|
| 1 | GDELT API | None | Free, real-time |
| 2 | NewsAPI | `NEWSAPI_KEY` | 100 req/day |
| 3 | GNews | `GNEWS_KEY` | 100 req/day |
| 4 | TheNewsAPI | `THENEWSAPI_KEY` | 100 req/day |
| 5 | Mediastack | `MEDIASTACK_KEY` | 500 req/month |
| 6 | CurrentsAPI | `CURRENTSAPI_KEY` | 600 req/day |

If all fail: `score=0.0`, `label=UNAVAILABLE` — positions not penalised.

**Risk label thresholds:**

| Score | Label | Effect on Hybrid Score |
|---|---|---|
| 0.00–0.25 | LOW | No change |
| 0.25–0.50 | MEDIUM | No change |
| 0.50–0.75 | HIGH | `hybrid × 0.5` |
| 0.75–1.00 | CRITICAL | `hybrid = 0.0` (all positions zeroed) |

Results cached in Redis for 1 hour with 24-hour stale backup.

---

## Security Architecture

5 independent security layers — failure in one does not compromise others.

![5-Layer Security Architecture](docs/images/security.png)

| Layer | Mechanism | Key Detail |
|---|---|---|
| 1 — Network | HTTPS + `CORS_ORIGINS` env var | Origins from `.env` only — never in `docker-compose.yml` |
| 2 — Request Gate | PUBLIC_PATHS whitelist + JWT/API key | 8 public paths; all others require JWT cookie or `X-API-KEY` |
| 3 — Rate Limiting | Redis counter per (IP, endpoint) | 5 req/min on owner login; fails open when Redis is down |
| 4 — Role Enforcement | AuthMiddleware v2.4 | Owner-only paths return 403 for demo; API key = owner access |
| 5 — Demo Quota | DemoTracker per (fingerprint, feature) | 10 req per feature per 7-day window; independent per group |

**Per-endpoint rate limits:**

| Endpoint | Limit |
|---|---|
| `POST /auth/owner-login` | 5 req / 60s |
| `POST /auth/demo-login` | 10 req / 60s |
| `POST /snapshot` | 10 req / 60s |
| `GET /agent/explain` | 20 req / 60s |
| `GET /health/*` | 60 req / 60s |

---

## Demo Quota System

Demo users receive 10 requests per feature group per 7-day window. When a limit is reached, the API returns HTTP 200 with `demo_locked: true` — not an HTTP error — allowing the frontend to show an informative UI component.

![Demo Quota System](docs/images/demo_quota.png)

| Feature Key | Endpoints |
|---|---|
| `snapshot` | `/snapshot`, `/predict/live-snapshot` |
| `portfolio` | `/portfolio` |
| `drift` | `/drift` |
| `performance` | `/performance` |
| `agent` | `/agent/explain`, `/agent/political-risk` |
| `signals` | `/equity/*`, `/model/feature-importance` |

- Fingerprint: `SHA256(IP + User-Agent)[:32]` — Redis key: `demo:usage:{fingerprint}:{feature}`
- TTL: 7 days (auto-reset). `fully_locked: true` only when ALL 6 features exhausted

---

## Model Governance

```
artifacts/xgboost/
├── model_xgb_YYYYMMDD_HHMMSS.pkl       ← model artifact (joblib)
├── metadata_xgb_YYYYMMDD_HHMMSS.json   ← 4 integrity hashes + metrics
└── production_pointer.json              ← ModelLoader reads THIS (not latest.json)
```

**XGBoost hyperparameters (v4.5.3):**

| Parameter | Value | Old | Reason |
|---|---|---|---|
| `eta` | 0.01 | 0.05 | Old value caused best_iteration=4 on noisy financial targets |
| `num_boost_rounds` | 2000 | 400 | More convergence room at lower learning rate |
| `early_stopping_rounds` | 50 | 30 | More patience for noisy validation signal |
| `max_depth` | 3 | 4 | Better generalisation on ~360 rows/ticker |
| `min_boost_rounds` | 30 | 10 | Rejects degenerate models |

**IC Statistics** (owner-only, `GET /model/ic-stats?days=30`, requires `STORE_PREDICTIONS=1`):

| IC Range | Quality |
|---|---|
| > 0.08 | Strong alpha |
| 0.04–0.08 | Moderate |
| 0.02–0.04 | Weak |
| < 0.02 | Near noise — retrain |

---

## Observability

**Prometheus metrics (GET `/metrics`)** scraped every 15s from `api:8000`:

```
api_requests_total{endpoint}          api_errors_total{endpoint}
api_latency_seconds{endpoint}         model_inference_total{model}
model_inference_latency_seconds       signal_distribution_total{signal}
cache_hits_total                      cache_misses_total
db_query_total{operation}             db_query_latency_seconds{operation}
db_rows_written_total{table}          db_tickers_synced_total
drift_detected (0-15 severity)        missing_feature_ratio
inference_in_progress                 pipeline_failures_total{stage}
```

Grafana datasource: `http://prometheus:9090`. Pre-built 17-panel dashboard: `app/monitoring/grafana_dashboard.json`.

**Log files:**

| File | Level |
|---|---|
| `logs/marketsentinel.log` | INFO+ (all environments) |
| `logs/issues.log` | WARNING+ (all environments) |
| `logs/debug.log` | DEBUG (development only) |
| `logs/access.log` | HTTP requests |

Production: console output disabled (files only).

---

## API Reference

**Public (no auth):**
```
GET  /health/live       Docker liveness probe
GET  /health/ready      Readiness: model + redis + db
GET  /universe          100-ticker S&P 500 list
GET  /model/info        Model version + hashes
GET  /agent/agents      4-agent descriptions + weights
GET  /metrics           Prometheus metrics
GET  /docs              Swagger UI
```

**Auth:**
```
POST /auth/owner-login  Admin login → JWT httpOnly cookie
POST /auth/demo-login   Demo access → JWT httpOnly cookie
GET  /auth/me           Current session + quota
POST /auth/logout       Clear JWT cookie
```

**Protected (owner or demo):**
```
POST /snapshot                          100 signals
GET  /portfolio                         Portfolio + health
GET  /drift                             Drift metrics
GET  /performance?days=N               Strategy performance
GET  /agent/explain?ticker=X           Per-ticker explanation
GET  /agent/political-risk?ticker=X    Geopolitical risk
GET  /equity/{ticker}/history?days=N   Price history
```

**Owner-only:**
```
GET  /model/ic-stats?days=30   IC statistics (Spearman)
GET  /model/feature-importance Feature importance by XGBoost gain
GET  /model/diagnostics        Full model checksums
POST /admin/sync               Manual data sync
```

**Key response notes:**
- `/drift` — `severity_score` is **0–15 integer** not a percentage. Display as `"X / 15"`
- `/equity/{ticker}/history` — response key is `"history"` not `"prices"`
- `/agent/explain` non-top-5 — `rank: null`, `agents_approved: []`, `selection_reason: ""`

---

## Database Schema

```sql
-- Raw market data (~40,000 rows; rows > 760 days auto-deleted)
CREATE TABLE ohlcv_daily (
  id      BIGSERIAL PRIMARY KEY,
  ticker  VARCHAR(20) NOT NULL,
  date    DATE NOT NULL,
  open, high, low, close FLOAT,
  volume  FLOAT,
  UNIQUE (ticker, date)
);

-- Feature cache (35s cold → 2s warm hit)
CREATE TABLE computed_features (
  ticker, date, feature_version VARCHAR(64), feature_data JSONB,
  UNIQUE (ticker, date, feature_version)
);

-- Prediction audit trail (for IC stats, requires STORE_PREDICTIONS=1)
CREATE TABLE model_predictions (
  ticker, date, model_version, schema_signature VARCHAR(64),
  raw_model_score, hybrid_score, weight FLOAT,
  signal VARCHAR(10), drift_state VARCHAR(20),
  UNIQUE (ticker, date, model_version)
);
```

---

## Project Structure

```
MarketSentinel/
├── app/                     ← FastAPI inference control plane
│   ├── api/routes/          ← 10 route modules
│   ├── core/auth/           ← JWT, middleware, demo tracker
│   ├── inference/           ← cache, model_loader, pipeline
│   ├── monitoring/          ← Prometheus metrics, Grafana dashboard
│   └── main.py              ← FastAPI app v3.7
│
├── core/                    ← Business domain (framework-agnostic)
│   ├── agent/               ← 4 agents + base class
│   ├── data/                ← sync, market service, providers
│   ├── db/                  ← engine, ORM models, repository
│   ├── features/            ← 64-feature pipeline
│   ├── models/              ← SafeXGBRegressor v4.5.3
│   ├── monitoring/          ← drift detector, regime, retrain trigger
│   └── schema/              ← 64-feature contract + SHA256
│
├── training/                ← Training pipeline (separate container)
│   ├── backtesting/         ← walk-forward, backtest engine
│   └── train_xgboost.py
│
├── tests/                   ← 197 passing, 1 skipped (17 test files)
│
├── docs/images/             ← Diagrams and screenshots
│   ├── dashboard_ui.png
│   ├── arch_overview.png
│   ├── data_sync.png
│   ├── features.png
│   ├── training_pipeline.png
│   ├── caching.png
│   ├── hybrid_score.png
│   ├── agent_pipeline.png
│   ├── drift_detection.png
│   ├── political_risk.png
│   ├── security.png
│   └── demo_quota.png
│
├── docker/                  ← inference + training Dockerfiles
├── scripts/                 ← generate_api_key, generate_owner_hash
├── config/universe.json     ← 100 S&P 500 tickers
├── .github/workflows/ci.yml ← lint + test + docker build
└── docker-compose.yml       ← training uses profiles:[training]
```

---

## Quick Start

### Prerequisites
Docker Desktop · Python 3.10 · 4 GB RAM · Internet access

### 1 — Clone
```bash
git clone https://github.com/muhammedshihab1001/MarketSentinel.git
cd MarketSentinel
```

### 2 — Generate Credentials
```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements/base.txt
python scripts/generate_api_key.py       # paste output as API_KEY in .env
python scripts/generate_owner_hash.py    # paste output as OWNER_PASSWORD_HASH in .env
```

### 3 — Configure .env
```env
OWNER_USERNAME=your_username
OWNER_PASSWORD_HASH="$2b$12$..."    # double quotes required
JWT_SECRET=your-32-char-secret
API_KEY=your64hexkey                # no quotes

POSTGRES_HOST=postgres
DATABASE_URL=postgresql+psycopg2://sentinel:sentinel@postgres:5432/marketsentinel
REDIS_HOST=redis
CORS_ORIGINS=http://localhost:5173  # set to your frontend URL in production
NEWSAPI_KEY=your_newsapi_key        # optional
STORE_PREDICTIONS=1
LLM_ENABLED=false
```

### 4 — Start Services
```bash
docker compose up -d
# Starts: PostgreSQL, Redis, API, Prometheus, Grafana
# Training does NOT start automatically
```

### 5 — First-Time Training
```bash
docker compose --profile training run --rm -e CREATE_BASELINE=1 training
```

### 6 — Verify
```bash
curl http://localhost:8000/health/ready
# → {"ready":true,"models_loaded":true,"db_connected":true,...}
```

### Retrain
```bash
docker compose --profile training run --rm -e SKIP_SYNC=1 training
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full PostgreSQL connection string |
| `REDIS_HOST` | Yes | `redis` | Redis hostname |
| `API_KEY` | Yes | — | External API key — no quotes |
| `OWNER_USERNAME` | Yes | — | Admin username — never hardcoded |
| `OWNER_PASSWORD_HASH` | Yes | — | bcrypt hash — with double quotes |
| `JWT_SECRET` | Yes | — | JWT signing secret (min 32 chars) |
| `NEWSAPI_KEY` | No | — | NewsAPI fallback |
| `GNEWS_KEY` | No | — | GNews fallback |
| `THENEWSAPI_KEY` | No | — | TheNewsAPI fallback |
| `MEDIASTACK_KEY` | No | — | Mediastack fallback |
| `CURRENTSAPI_KEY` | No | — | CurrentsAPI fallback |
| `STORE_PREDICTIONS` | No | `1` | Store predictions for IC stats |
| `DEMO_REQUESTS_PER_FEATURE` | No | `10` | Demo quota per feature per week |
| `LLM_ENABLED` | No | `false` | Enable OpenAI explanations |
| `CORS_ORIGINS` | Yes | — | Allowed origins — `.env` only, never in `docker-compose.yml` |
| `COOKIE_SECURE` | No | `0` | Set `1` in production |
| `COOKIE_SAMESITE` | No | `lax` | Set `none` for cross-origin frontend |
| `SNAPSHOT_PRECOMPUTE_INTERVAL` | No | `300` | Background snapshot frequency |
| `INFERENCE_LOOKBACK_DAYS` | No | `400` | Price history window for inference |
| `TRAINING_LOOKBACK_DAYS` | No | `730` | Price history window for training |
| `GF_SECURITY_ADMIN_PASSWORD` | No | `admin` | Grafana admin password |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |

---

## CI Pipeline

```
Triggers: push to feature/*, develop, main | PR to develop, main

Job 1 — Tests & Lint:
  PostgreSQL 16 service container
  flake8: 85% file clean threshold, max-line-length=100
  pytest: 197 must pass (1 skip allowed)

Job 2 — Docker Build (main/develop only):
  inference.Dockerfile → marketsentinel-inference:ci
  training.Dockerfile  → marketsentinel-training:ci

Status: ✅ All checks passing | 197 passed, 1 skipped
```

---

## Performance Benchmarks

| Operation | Cold | Warm (Cached) |
|---|---|---|
| Full snapshot (100 tickers) | 15–18s | **< 100ms** |
| Feature engineering (100×400 days) | ~12s | ~2s (DB cache) |
| XGBoost predict() (100×64 matrix) | < 200ms | < 200ms |
| Agent explain (any ticker) | 100–400ms | < 50ms |
| Portfolio | < 50ms | < 20ms |
| Drift | < 100ms | < 50ms |
| Political risk (GDELT) | 3–8s | < 20ms |

Two-tier caching: Redis (snapshot TTL=360s) + PostgreSQL (feature cache by schema version). Users refreshing every 30 seconds never trigger recomputation.

---

## Author

**Muhammed Shihab P** 

Building production ML systems, MLOps platforms, and decision intelligence engines.

---

## License

MIT License — see [LICENSE](./LICENSE) for details.  
Copyright (c) 2026 Muhammed Shihab P