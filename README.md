# 📈 MarketSentinel – Institutional-Style Financial Forecasting & Decision Intelligence System

MarketSentinel is a **production-structured Machine Learning system** designed to generate **probabilistic forecasts and actionable trading signals** by combining:

- Market price data  
- Financial news sentiment (FinBERT)  
- Multi-model forecasting  
- Scenario analysis  
- Decision intelligence  
- Observability  

This is **NOT a notebook-based ML project.**  
It is engineered as a **real-world ML system** with clear domain boundaries, artifact governance, testing, and infrastructure readiness.

---

# 🚨 Problem Statement

Financial markets are complex, noisy, and heavily influenced by narrative.

Most retail forecasting systems fail because they rely on:

- Single-model predictions  
- Static indicators  
- Manual sentiment interpretation  
- No risk-aware decision logic  
- No production monitoring  

Additionally, many ML repositories stop at training — they do not address:

- inference architecture  
- artifact lifecycle  
- model validation  
- system reliability  

---

# ✅ MarketSentinel Solution

MarketSentinel is built as a **decision-support engine**, not just a predictor.

It solves the above challenges through:

### Multi-Signal Intelligence
- Price-based technical features
- Transformer-based sentiment analysis
- Ensemble forecasting

### Risk-Aware Decision Layer
Signals are generated only when model conviction and market context align.

### Canonical Feature Pipeline
Training and inference use the **same feature generation contract**, preventing feature drift.

### Artifact Governance
Models are versioned with metadata describing:

- training window  
- metrics  
- feature schema  

### Observability Built-In
System metrics are exported via Prometheus and visualized in Grafana.

### CI-Enforced Model Quality
Model performance thresholds are validated automatically.

---

# 🧠 System Architecture

MarketSentinel follows a **domain-driven ML architecture**.

```
MarketSentinel/
│
├── core/                 ← Domain layer (system spine)
│   ├── data/             ← Market + news ingestion
│   ├── sentiment/        ← FinBERT analysis
│   ├── features/         ← Canonical feature pipeline
│   ├── forecasting/      ← Probabilistic modeling
│   ├── scenario/         ← Stress testing forecasts
│   ├── signals/          ← Decision engine
│   ├── explainability/   ← Human-readable reasoning
│   └── artifacts/        ← Metadata management
│
├── training/             ← Offline model training
├── app/                  ← Inference control plane (FastAPI)
├── docker/               ← Training & inference containers
├── tests/                ← System-level validation
├── requirements/         ← Environment segmentation
└── docker-compose.yml
```

---

# 🤖 Model Stack

MarketSentinel uses a **multi-model ensemble**, where each model answers a different market question.

| Model | Role |
|--------|--------|
| **XGBoost** | Predict directional probability |
| **LSTM** | Forecast short-term price trajectory |
| **Prophet** | Identify macro trend |
| **FinBERT** | Quantify narrative sentiment |

No single model drives decisions.

Signals emerge from **model agreement + risk constraints.**

---

# ⚙️ Canonical Feature Pipeline

Implemented in:

```
core/features/feature_engineering.py
```

This pipeline is the **only approved path** for generating model inputs.

Features include:

- Returns  
- Volatility  
- RSI  
- MACD  
- Sentiment aggregates  
- Lagged features  

### Why This Matters
Feature inconsistency is one of the biggest causes of production ML failure.

MarketSentinel prevents this by enforcing a shared pipeline across:

- training  
- inference  
- backtesting  

---

# 🧭 Decision Intelligence Layer

Located in:

```
core/signals/
```

The decision engine:

- evaluates model probability  
- incorporates sentiment  
- respects volatility caps  
- validates expected return  
- confirms macro trend  

Signals are only issued when conviction meets risk thresholds.

Otherwise → **HOLD**

This avoids overtrading — a critical real-world behavior.

---

# 🔮 Scenario Engine

MarketSentinel does not rely solely on point forecasts.

It generates alternative futures:

- Bull case  
- Bear case  
- Volatility shock  
- Sentiment crash  

This enables **stress-tested decision making.**

---

# 🧾 Explainability

Every signal can be translated into human-readable reasoning via:

```
core/explainability/decision_explainer.py
```

Example outputs include:

- Model shows strong upward probability  
- Market sentiment is positive  
- Asset is overbought  

Explainability is essential for financial systems.

Black-box signals are rarely trusted.

---

# 🚀 Inference Control Plane

The FastAPI service orchestrates:

- data ingestion  
- feature generation  
- model loading  
- decision logic  
- scenario analysis  

### Primary Endpoint

```
POST /predict
```

Returns:

- signal  
- confidence  
- probability  
- forecast path  
- scenarios  
- explanation  

Designed for direct integration with dashboards or trading interfaces.

---

# ❤️ Health & Reliability

MarketSentinel exposes two critical probes:

### Liveness
```
GET /health/
```
Confirms the container is alive.

### Readiness
```
GET /health/ready
```
Verifies:

- models loaded  
- artifacts accessible  
- inference ready  

This separation is standard in production ML deployments.

---

# 📊 Observability

### Prometheus Metrics

Tracked signals include:

- model inference latency  
- prediction volume  
- signal distribution  
- confidence scores  
- feature missing ratio  

### Grafana Dashboards

Used to visualize system behavior and detect anomalies early.

Observability is treated as a **first-class system component**, not an afterthought.

---

# 🧪 Testing & Validation

Tests live in:

```
tests/
```

Coverage includes:

- feature engineering  
- sentiment pipeline  
- decision logic  
- backtesting  
- evaluation  

---

## Model Quality Gates

Executed via:

```
training/run_evaluation.py
```

Minimum thresholds:

| Model | Requirement |
|--------|--------------|
| XGBoost | Accuracy ≥ 0.55 |
| LSTM | RMSE ≤ 5 |
| Prophet | MAE ≤ 5 |

CI fails if models regress.

---

# 🔁 CI Pipeline

Implemented using GitHub Actions.

Pipeline:

1️⃣ Install dependencies  
2️⃣ Run unit tests  
3️⃣ Execute evaluation checks  
4️⃣ Block merges on failure  

Prevents silent model degradation.

---

# 🐳 Container Strategy

MarketSentinel uses **separate containers** for:

### Training Environment
Heavy ML dependencies allowed.

### Inference Environment
Slim, fast, production-oriented.

Artifacts are mounted — **not baked into images** — enabling rapid redeployment without rebuilding containers.

---

# 🎯 What This Project Demonstrates

MarketSentinel reflects real-world ML system design principles:

- Domain-driven architecture  
- Artifact governance  
- Feature contract enforcement  
- Multi-model decision fusion  
- Scenario-based forecasting  
- Observability-first mindset  
- CI-backed model validation  
- Containerized infrastructure  

This repository represents a transition from **ML prototype → production-shaped system.**

---

# 👨‍💻 Author

**Muhammed Shihab P**

Focused on:

- Machine Learning Systems  
- MLOps  
- Production AI Infrastructure  
- Decision Intelligence Architectures  

---
