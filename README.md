# 📈 MarketSentinel – Intelligent Stock Forecasting & Trading Signal System

MarketSentinel is an **end-to-end production-grade Machine Learning system** that generates **BUY / SELL / HOLD** trading signals by combining **market data, news sentiment, multiple ML models, and monitoring**.

This project focuses on **real-world ML engineering**, not just model training.

---

## 🚨 Problem Statement

Retail traders and analysts face several challenges:

- Market price data is **highly noisy**
- News sentiment is often **ignored or manually interpreted**
- Single-model predictions are **unreliable**
- Most ML projects lack **monitoring, testing, and CI/CD**

---

## ✅ MarketSentinel Solution

MarketSentinel solves these problems by:

- Combining **price action + sentiment analysis**
- Using **multiple ML models** instead of one
- Applying **risk-aware signal logic**
- Providing **real-time monitoring (Prometheus + Grafana)**
- Enforcing **model quality checks in CI/CD**

---

## 📂 Project Structure
```
MarketSentinel/
│
├── app/
│ ├── api/ # FastAPI routes
│ ├── services/ # Business logic
│ ├── models/ # Model loading & inference
│ ├── monitoring/ # Prometheus metrics
│ └── main.py # FastAPI entry point
│
├── training/ # Model training & evaluation
├── tests/ # Pytest unit tests
├── monitoring/ # Prometheus config
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ # CI/CD pipelines
└── README.md
└── requirements-ci.txt
└── requirements.docker.txt
└── requirements.txt
```

---

## 🔌 API Endpoints

| Endpoint | Description |
|--------|------------|
| `/` | Root service status |
| `/health` | Health check |
| `/full-prediction` | Complete trading signal |
| `/metrics` | Prometheus metrics |
| `/docs` | Swagger UI |

---

## 📊 Feature Engineering

Implemented in `app/services/feature_engineering.py`

Features include:
- Daily returns
- Volatility
- RSI
- MACD
- Lagged returns
- Lagged sentiment

These features are used consistently across all models.

---

## 🤖 Machine Learning Models

| Model | Purpose |
|-----|--------|
| **XGBoost** | Predict next-day direction |
| **LSTM** | Forecast short-term price movement |
| **Prophet** | Detect long-term trend |

Models are:
- Trained **offline**
- Loaded in **production**
- Evaluated automatically

---

## 🧠 Signal Engine

Implemented in `app/services/signal_engine.py`

Decision logic:
- Combines outputs from all models
- Applies risk constraints
- Avoids trading during high volatility

Example logic:
- BUY only if all models agree
- HOLD if volatility is too high
- SELL only with bearish confirmation

---

## 📈 Monitoring & Observability

### Prometheus
**File:** `app/monitoring/metrics.py`

Tracked metrics:
- API request count
- Prediction count
- Error count
- Latency
- Average confidence


---

### Grafana
Used to visualize:
- API health
- Prediction volume
- Error trends
- Confidence distributions

---

## 🧪 Testing Strategy

### Unit Tests
Located in `tests/`

Covers:
- Feature engineering
- Signal logic
- API responses
- Metrics validation

### Model Evaluation
Located in `training/run_evaluation.py`

Quality thresholds:
- XGBoost accuracy ≥ **0.55**
- LSTM MAE ≤ **5**
- Prophet MAE ≤ **6**

CI fails if thresholds are not met.

---

## 🔁 CI/CD Pipeline

**Tool:** GitHub Actions  
**File:** `.github/workflows/ci.yml`

Pipeline steps:
1. Install dependencies
2. Run unit tests
3. Run model evaluation
4. Block merge on failure

---

## 🐳 Docker & Deployment

- Slim Python base image
- Optimized requirements
- Multi-container setup:
  - API
  - Prometheus
  - Grafana

  ---

## 🎯 Why This Project Matters

MarketSentinel demonstrates:

- Production ML system design
- Multi-model decision fusion
- Monitoring & observability
- CI/CD with model quality gates
- Clean, scalable architecture


---

## 👨‍💻 Author

**Muhammed Shihab P**  
Focused on **Machine Learning Engineering, MLOps & Production AI Systems**