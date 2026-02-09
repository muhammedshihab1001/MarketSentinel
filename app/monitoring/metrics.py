from prometheus_client import Counter, Histogram, Gauge

# -----------------------------------------
# REQUEST LAYER
# -----------------------------------------

API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint"]
)

API_ERROR_COUNT = Counter(
    "api_errors_total",
    "Total API errors",
    ["endpoint"]
)

API_LATENCY = Histogram(
    "api_latency_seconds",
    "API request latency",
    ["endpoint"]
)

# -----------------------------------------
# INFERENCE LAYER
# -----------------------------------------

MODEL_INFERENCE_COUNT = Counter(
    "model_inference_total",
    "Number of model inferences",
    ["model"]
)

MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Latency per model",
    ["model"]
)

SIGNAL_DISTRIBUTION = Counter(
    "signal_distribution_total",
    "Distribution of BUY/SELL/HOLD",
    ["signal"]
)

FORECAST_HORIZON = Gauge(
    "forecast_horizon_days",
    "Forecast horizon length"
)

CONFIDENCE_SCORE = Gauge(
    "prediction_confidence",
    "Model confidence score"
)

# -----------------------------------------
# DATA QUALITY
# -----------------------------------------

MISSING_FEATURE_RATIO = Gauge(
    "missing_feature_ratio",
    "Ratio of missing features during inference"
)
