from prometheus_client import Counter, Histogram, Gauge

# -----------------------------
# API METRICS
# -----------------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint"]
)

# -----------------------------
# MODEL METRICS
# -----------------------------

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["signal"]
)

AVG_CONFIDENCE = Gauge(
    "model_avg_confidence",
    "Average prediction confidence"
)

# -----------------------------
# DATA QUALITY METRICS
# -----------------------------

MISSING_VALUE_RATIO = Gauge(
    "data_missing_ratio",
    "Ratio of missing values in input features"
)
