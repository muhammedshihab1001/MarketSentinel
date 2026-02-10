from prometheus_client import Counter, Histogram, Gauge

# =====================================================
# REQUEST LAYER
# =====================================================

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

# =====================================================
# MODEL INFERENCE
# =====================================================

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

# 🔥 NEW — live model version visibility
MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently loaded model version",
    ["model", "version"]
)

SIGNAL_DISTRIBUTION = Counter(
    "signal_distribution_total",
    "Distribution of BUY/SELL/HOLD",
    ["signal"]
)

# 🔥 NEW — prediction class skew detection
PREDICTION_CLASS_PROBABILITY = Histogram(
    "prediction_probability",
    "Prediction probability distribution",
    ["model"]
)

FORECAST_HORIZON = Gauge(
    "forecast_horizon_days",
    "Forecast horizon length"
)

CONFIDENCE_SCORE = Gauge(
    "prediction_confidence",
    "Model confidence score"
)

# =====================================================
# DATA QUALITY
# =====================================================

MISSING_FEATURE_RATIO = Gauge(
    "missing_feature_ratio",
    "Ratio of missing features during inference"
)

# 🔥 NEW — feature drift sentinel
FEATURE_MEAN = Gauge(
    "feature_mean",
    "Live feature mean",
    ["feature"]
)

FEATURE_STD = Gauge(
    "feature_std",
    "Live feature standard deviation",
    ["feature"]
)

# 🔥 NEW — extreme value detection
FEATURE_MAX = Gauge(
    "feature_max",
    "Live feature max",
    ["feature"]
)

FEATURE_MIN = Gauge(
    "feature_min",
    "Live feature min",
    ["feature"]
)

# =====================================================
# PIPELINE HEALTH
# =====================================================

PIPELINE_FAILURES = Counter(
    "pipeline_failures_total",
    "Total pipeline failures",
    ["stage"]
)
