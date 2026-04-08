from prometheus_client import Counter, Histogram, Gauge


# =====================================================
# LATENCY BUCKETS (ML-AWARE)
# =====================================================

LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)


# =====================================================
# REQUEST LAYER
# =====================================================

API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint"],
)

API_ERROR_COUNT = Counter(
    "api_errors_total",
    "Total API errors",
    ["endpoint"],
)

API_LATENCY = Histogram(
    "api_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)


# =====================================================
# MODEL INFERENCE
# =====================================================

MODEL_INFERENCE_COUNT = Counter(
    "model_inference_total",
    "Number of model inferences",
    ["model"],
)

MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Latency per model",
    ["model"],
    buckets=LATENCY_BUCKETS,
)

MODEL_VERSION = Gauge(
    "model_version",
    "Currently loaded model version",
    ["model", "version"],
)

SIGNAL_DISTRIBUTION = Counter(
    "signal_distribution_total",
    "Distribution of LONG/SHORT signals",
    ["signal"],
)

PREDICTION_CLASS_PROBABILITY = Histogram(
    "prediction_probability",
    "Prediction probability distribution",
    ["model"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)


# =====================================================
# DATA QUALITY
# =====================================================

MISSING_FEATURE_RATIO = Gauge(
    "missing_feature_ratio",
    "Ratio of missing features during inference",
)


# =====================================================
# FEATURE MONITORING
# =====================================================

FEATURE_MEAN = Gauge(
    "feature_mean",
    "Live feature mean",
    ["feature"],
)

FEATURE_STD = Gauge(
    "feature_std",
    "Live feature standard deviation",
    ["feature"],
)

FEATURE_MAX = Gauge(
    "feature_max",
    "Live feature max",
    ["feature"],
)

FEATURE_MIN = Gauge(
    "feature_min",
    "Live feature min",
    ["feature"],
)

DRIFT_DETECTED = Gauge(
    "drift_detected",
    "Global drift detection flag",
)


# =====================================================
# CACHE OBSERVABILITY
# =====================================================

CACHE_HITS = Counter(
    "cache_hits_total",
    "Number of Redis cache hits",
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Number of Redis cache misses",
)


# =====================================================
# DATABASE OBSERVABILITY  (NEW)
# =====================================================

DB_QUERY_COUNT = Counter(
    "db_query_total",
    "Total PostgreSQL queries",
    ["operation"],  # labels: read, write, upsert
)

DB_QUERY_LATENCY = Histogram(
    "db_query_latency_seconds",
    "PostgreSQL query latency",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

DB_ROWS_WRITTEN = Counter(
    "db_rows_written_total",
    "Total rows written to PostgreSQL",
    ["table"],  # labels: ohlcv_daily, computed_features, model_predictions
)

DB_SYNC_COUNT = Counter(
    "db_sync_total",
    "Number of data sync runs",
    ["status"],  # labels: success, error, skipped
)

DB_SYNC_LATENCY = Histogram(
    "db_sync_latency_seconds",
    "Data sync duration per ticker",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

DB_TICKERS_SYNCED = Gauge(
    "db_tickers_synced_total",
    "Number of tickers successfully synced in last run",
)


# =====================================================
# PIPELINE HEALTH
# =====================================================

PIPELINE_FAILURES = Counter(
    "pipeline_failures_total",
    "Total pipeline failures",
    ["stage"],
)

INFERENCE_IN_PROGRESS = Gauge(
    "inference_in_progress",
    "Number of active inferences",
)
