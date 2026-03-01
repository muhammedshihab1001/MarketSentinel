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

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

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
# LLM VALIDATION
# ============================================================

def _validate_llm():

    llm_enabled = _as_bool(os.getenv("LLM_ENABLED", "0"))

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if llm_enabled and not api_key:
        logger.error("LLM_ENABLED=1 but OPENAI_API_KEY missing.")
        raise RuntimeError("LLM misconfiguration: missing OPENAI_API_KEY")

    if llm_enabled:
        logger.info(
            "LLM enabled | model=%s | rate_limit=%s/min",
            model,
            os.getenv("LLM_RATE_LIMIT_PER_MIN", "30")
        )
    else:
        logger.info("LLM disabled.")


# ============================================================
# ENVIRONMENT FINGERPRINT
# ============================================================

def _environment_fingerprint():

    payload = (
        sys.version +
        os.getenv("ENABLE_SENTIMENT", "") +
        os.getenv("MODEL_REGISTRY_PATH", "") +
        os.getenv("FEATURE_STORE_PATH", "") +
        os.getenv("XGB_REGISTRY_DIR", "") +
        os.getenv("LLM_ENABLED", "") +
        os.getenv("LOG_LEVEL", "") +
        os.getenv("MARKET_PROVIDERS_ACTIVE", "")
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

        # Load dotenv FIRST so LOG_LEVEL etc. apply correctly
        if os.getenv("DOTENV_ENABLED", "1") == "1":
            load_dotenv(override=False)

        _configure_logging()

        defaults = {
            "ENABLE_SENTIMENT": "0",
            "LOG_LEVEL": "INFO",
            "MODEL_REGISTRY_PATH": "artifacts/registry",
            "FEATURE_STORE_PATH": "artifacts/feature_store",
            "XGB_REGISTRY_DIR": "artifacts/xgboost",

            "LLM_ENABLED": "0",
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_TIMEOUT": "12",
            "LLM_RATE_LIMIT_PER_MIN": "30",
            "LLM_CACHE_ENABLED": "1",
            "LLM_CACHE_TTL_SEC": "120",
            "LLM_AUDIT_ENABLED": "1",
        }

        for key, value in defaults.items():
            os.environ.setdefault(key, value)

        _ensure_path("MODEL_REGISTRY_PATH", "artifacts/registry")
        _ensure_path("FEATURE_STORE_PATH", "artifacts/feature_store")
        _ensure_path("XGB_REGISTRY_DIR", "artifacts/xgboost")

        _validate_market_providers()
        _validate_news()
        _validate_llm()

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
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise RuntimeError(f"Invalid integer value for env {key}: {val}")


def get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise RuntimeError(f"Invalid float value for env {key}: {val}")


def get_bool(key: str, default: bool = False) -> bool:
    return _as_bool(os.getenv(key), default)