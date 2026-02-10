# 📈 MarketSentinel – Production-Grade Financial Forecasting & Decision Intelligence System

MarketSentinel is a **production-structured Machine Learning platform** designed to generate **probabilistic forecasts and actionable trading signals** by combining:

- Market price data  
- Financial news sentiment (FinBERT)  
- Multi-model forecasting  
- Feature store architecture  
- Walk-forward validated strategies  
- Decision intelligence  
- Drift monitoring  
- Observability  

⚠️ This is NOT a notebook-based ML project.

It is engineered like a **real-world ML system** with:

- domain separation  
- artifact governance  
- model registry  
- canonical feature pipelines  
- CI validation  
- containerized infrastructure  

This repository demonstrates how modern ML systems are built for **reliability, traceability, and deployment readiness.**

---

# 🚨 Problem Statement

Financial markets are unpredictable and heavily influenced by both data and narrative.

Most forecasting projects fail because they rely on:

- single-model predictions  
- inconsistent features  
- manual sentiment interpretation  
- no validation strategy  
- no artifact lifecycle  
- zero production monitoring  

Additionally, many ML repositories stop after training and ignore critical system concerns such as:

- inference orchestration  
- feature drift  
- model versioning  
- risk-aware decisions  
- deployment safety  

---

# ✅ MarketSentinel Solution

MarketSentinel is built as a **decision-support system**, not just a forecasting tool.

It solves real-world ML challenges through:

## Multi-Model Intelligence
Each model answers a different market question.

| Model | Responsibility |
|--------|----------------|
| **XGBoost** | Predict price direction probability |
| **LSTM** | Forecast short-term price trajectory |
| **Prophet** | Detect macro market trend |
| **FinBERT** | Quantify financial sentiment |

Signals are produced only when models align with market context.

---

## Canonical Feature Contract
Training, inference, and backtesting all use the **same feature pipeline**, preventing one of the biggest causes of production ML failure — feature drift.

---

## Artifact Governance
Models are versioned using a lightweight filesystem registry with metadata including:

- dataset fingerprint  
- metrics  
- training window  
- feature schema  

This enables:

✅ reproducibility  
✅ rollback capability  
✅ audit readiness  

---

## Walk-Forward Strategy Validation
Instead of relying only on test splits, MarketSentinel validates strategies using:

- rolling training windows  
- forward simulation  
- equity curve tracking  
- drawdown measurement  

Promotion gates ensure only robust models are deployed.

---

## Feature Store Layer
The feature store provides:

- canonical datasets  
- offline feature reuse  
- rebuild detection  
- dataset persistence  

This moves the system closer to enterprise ML patterns such as **Feast-style architectures.**

---

## Drift Monitoring
MarketSentinel includes statistical drift visibility through:

```core/monitoring/drift_detector.py```

Feature distributions are monitored to detect:

- data instability  
- regime shifts  
- upstream data issues  

Early detection reduces model risk.

---

## Observability First
Metrics are exported via Prometheus and visualized with Grafana.

Tracked signals include:

- inference latency  
- prediction distribution  
- feature statistics  
- pipeline failures  

Observability is treated as a **core system component**, not an afterthought.

---

# 🧠 System Architecture

MarketSentinel follows a **domain-driven architecture**, separating business logic from infrastructure.

```
MarketSentinel/
│
├── app/ ← FastAPI inference control plane
│ ├── api/routes/ ← Prediction & health endpoints
│ ├── inference/ ← Model loader, cache, pipeline
│ └── monitoring/ ← Metrics + Prometheus config
│
├── core/ ← Domain layer (system backbone)
│ ├── artifacts/ ← Metadata + model registry
│ ├── data/ ← Market & news ingestion
│ ├── sentiment/ ← FinBERT analyzer
│ ├── features/ ← Feature engineering + store
│ ├── forecasting/ ← Probabilistic outputs
│ ├── signals/ ← Decision engine
│ ├── scenario/ ← Scenario generator
│ ├── explainability/ ← Human-readable reasoning
│ ├── monitoring/ ← Drift detection
│ └── schema/ ← Feature contracts
│
├── training/ ← Offline model pipelines
│ ├── pipelines/ ← Training orchestrator
│ ├── backtesting/ ← Strategy validation
│ ├── indicators/ ← Technical indicators
│ └── sentiment/ ← Training sentiment tools
│
├── models/ ← Model definitions
├── docker/ ← Training & inference containers
├── tests/ ← System validation
└── requirements/ ← Environment segmentation
```

This structure mirrors production ML platforms.

---

# 🧭 Decision Intelligence Layer

Located in:

```core/signals/signal_engine.py```

The decision engine evaluates:

- probability  
- sentiment  
- volatility  
- RSI  
- expected return  
- macro trend  

Signals are issued only when conviction passes risk thresholds.

Otherwise → **HOLD**

This prevents overtrading — a critical institutional behavior.

---

# 🔮 Scenario Engine

MarketSentinel generates alternative futures instead of relying only on point forecasts.

Examples:

- bullish expansion  
- bearish contraction  
- volatility shock  
- sentiment collapse  

This supports **stress-tested decision making.**

---

# 🧾 Explainability

Every signal can be translated into human-readable reasoning via:

```core/explainability/decision_explainer.py```

Explainability is essential in financial ML systems where blind predictions are rarely trusted.

---

# 🚀 Inference Control Plane

The FastAPI service orchestrates:

- dataset retrieval  
- feature generation  
- model loading  
- drift metrics  
- decision logic  
- scenario analysis  

### Primary Endpoint

```POST /predict```

Returns:

- signal  
- confidence  
- probability  
- forecast timeline  
- scenarios  
- explanation  

Designed for integration with dashboards, research tools, or trading interfaces.

---

# ❤️ Reliability & Health

### Liveness Probe
```GET /health/```

Confirms the service is alive.

### Readiness Probe
```GET /health/ready```

Ensures:

- models loaded  
- registry accessible  
- inference ready  

This separation follows production deployment standards.

---

# 🧪 Testing & Validation

Tests cover:

- feature engineering  
- sentiment pipeline  
- decision logic  
- backtesting  
- evaluation  

CI blocks merges if failures occur.

---

## Model Quality Gates

Executed via:

```training/run_evaluation.py```

| Model | Requirement |
|--------|--------------|
| XGBoost | Accuracy ≥ 0.55 |
| LSTM | RMSE ≤ 5 |
| Prophet | MAE ≤ 5 |

Only models that pass thresholds are promoted.

---

# 🔁 CI Pipeline

GitHub Actions workflow:

1️⃣ Install dependencies  
2️⃣ Run tests  
3️⃣ Execute evaluation checks  
4️⃣ Validate training pipeline  
5️⃣ Build Docker images  

This guarantees deployment readiness.

---

# 🐳 Container Strategy

MarketSentinel uses separate containers for:

## Training
Allows heavy ML dependencies.

## Inference
Slim, fast, production-oriented.

Artifacts are mounted rather than baked into images, enabling rapid redeployment.

---

# 🎯 What This Project Demonstrates

MarketSentinel reflects real-world ML engineering principles:

- domain-driven design  
- model registry  
- dataset fingerprinting  
- walk-forward validation  
- promotion gates  
- feature store architecture  
- drift monitoring  
- multi-model decision fusion  
- observability-first mindset  
- CI-backed validation  
- containerized deployment  

This project represents the shift from:

👉 **ML prototype → production-ready system**

---

# 🔮 Upcoming Enhancements

Planned upgrades include:

### Model Risk Kill Switch
Automatically disable signals during severe feature drift.

### Position Sizing Engine
Move from directional signals to portfolio-aware allocation.

### Automated Retraining
Trigger training pipelines when drift persists.

### Advanced Statistical Drift Detection
Expand monitoring to include distribution shift tests.

These improvements will push MarketSentinel further toward institutional-grade autonomy.

---

# 👨‍💻 Author

**Muhammed Shihab P**

Focused on:

- Machine Learning Systems  
- MLOps  
- Production AI Infrastructure  
- Decision Intelligence Architectures  
