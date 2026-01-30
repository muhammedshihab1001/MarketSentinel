from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["ticker"]
)

AVG_CONFIDENCE = Gauge(
    "prediction_confidence",
    "Average prediction confidence",
    ["ticker"]
)


# -----------------------------
# DATA QUALITY METRICS
# -----------------------------

MISSING_VALUE_RATIO = Gauge(
    "data_missing_ratio",
    "Ratio of missing values in input features"
)
