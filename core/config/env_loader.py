import os
import logging
import threading
import hashlib
import sys
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv


logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False
_ENV_LOCK = threading.Lock()


# ============================================================
# LOGGING CONFIG (CI + DOCKER SAFE)
# ============================================================

def _configure_logging():

    if logging.getLogger().handlers:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console logging (always enabled)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Optional file logging (disabled in CI/Docker unless explicitly enabled)
    if os.getenv("ENABLE_FILE_LOGGING", "0") == "1":

        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "marketsentinel.log"),
            maxBytes=10_000_000,
            backupCount=3,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info("Logging initialized | level=%s", log_level)


# ============================================================
# HELPERS
# ============================================================

def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_path(key: str, default: str):
    path = os.getenv(key, default)
    os.environ.setdefault(key, path)

    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create/access path for {key}: {e}"
        )


# ============================================================
# PROVIDER VALIDATION
# ============================================================

def _validate_market_providers():

    providers = {
        "twelvedata": "TWELVEDATA_API_KEY",
        "alphavantage": "ALPHAVANTAGE_API_KEY",
    }

    available = []

    for provider, key in providers.items():
        if os.getenv(key):
            available.append(provider)

    # Yahoo fallback always available
    available.append("yahoo")

    os.environ["MARKET_PROVIDERS_ACTIVE"] = ",".join(available)

    if len(available) == 1:
        logger.warning("Only Yahoo provider available.")


def _validate_news():

    if not _as_bool(os.getenv("ENABLE_SENTIMENT", "0")):
        logger.info("Sentiment disabled.")
        return

    primary = os.getenv("NEWS_PROVIDER_PRIMARY", "auto").lower()

    required_keys = {
        "finnhub": "FINNHUB_API_KEY",
        "marketaux": "MARKETAUX_API_KEY",
        "gnews": "GNEWS_API_KEY",
    }

    if primary == "auto":
        logger.info("News provider auto-selection enabled.")
        return

    if primary not in required_keys:
        logger.warning("Invalid news provider. Disabling sentiment.")
        os.environ["ENABLE_SENTIMENT"] = "0"
        return

    if not os.getenv(required_keys[primary]):
        logger.warning("Missing API key for news provider. Disabling sentiment.")
        os.environ["ENABLE_SENTIMENT"] = "0"


# ============================================================
# ENVIRONMENT FINGERPRINT
# ============================================================

def _environment_fingerprint():

    payload = (
        sys.version +
        os.getenv("ENABLE_SENTIMENT", "") +
        os.getenv("MODEL_REGISTRY_PATH", "") +
        os.getenv("FEATURE_STORE_PATH", "")
    )

    fp = hashlib.sha256(payload.encode()).hexdigest()[:12]

    os.environ["ENV_FINGERPRINT"] = fp
    logger.info("Environment fingerprint -> %s", fp)


# ============================================================
# INIT ENV
# ============================================================

def init_env():

    global _ENV_INITIALIZED

    if _ENV_INITIALIZED:
        return

    with _ENV_LOCK:

        if _ENV_INITIALIZED:
            return

        _configure_logging()

        # Load .env only if present (safe for CI)
        if os.getenv("DOTENV_ENABLED", "1") == "1":
            load_dotenv(override=False)

        # Safe defaults
        defaults = {
            "ENABLE_SENTIMENT": "0",
            "LOG_LEVEL": "INFO",
            "MODEL_REGISTRY_PATH": "artifacts/registry",
            "FEATURE_STORE_PATH": "artifacts/feature_store",
            "XGB_REGISTRY_DIR": "artifacts/xgboost",
        }

        for key, value in defaults.items():
            os.environ.setdefault(key, value)

        # Ensure required directories exist
        _ensure_path("MODEL_REGISTRY_PATH", "artifacts/registry")
        _ensure_path("FEATURE_STORE_PATH", "artifacts/feature_store")
        _ensure_path("XGB_REGISTRY_DIR", "artifacts/xgboost")

        # Validate providers
        _validate_market_providers()
        _validate_news()

        _environment_fingerprint()

        logger.info("Environment initialized successfully.")

        _ENV_INITIALIZED = True


# ============================================================
# ACCESSORS
# ============================================================

def get_env(key: str, default=None, required: bool = False):

    val = os.getenv(key, default)

    if required and val is None:
        raise RuntimeError(f"Missing required env: {key}")

    return val


def get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default


def get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


def get_bool(key: str, default: bool = False) -> bool:
    return _as_bool(os.getenv(key), default)