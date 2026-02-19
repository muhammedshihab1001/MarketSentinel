import os
import logging
import threading
import hashlib
import sys
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv, dotenv_values


logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False
_ENV_LOCK = threading.Lock()


########################################################
# GLOBAL LOGGING CONFIG (WINDOWS SAFE)
########################################################

def _configure_logging():

    if logging.getLogger().handlers:
        return  # Prevent double configuration

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    ####################################################
    # Console Handler (UTF-8 SAFE)
    ####################################################

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # Not supported on some environments

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    ####################################################
    # Rotating File Handler
    ####################################################

    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "marketsentinel.log"),
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging configured | level=%s", log_level)


########################################################
# HELPERS
########################################################

def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_not_empty(key: str) -> bool:
    val = os.getenv(key)
    return bool(val and val.strip())


########################################################
# DOTENV SAFETY
########################################################

def _fail_fast_dotenv():

    if not os.path.exists(".env"):
        logger.warning(".env not found - relying on system environment.")
        return

    try:
        dotenv_values(".env")
    except Exception as e:
        raise RuntimeError(
            f"Malformed .env file detected. Refusing to boot.\nError: {str(e)}"
        )


########################################################
# PATH VALIDATION
########################################################

def _validate_paths():

    required_paths = [
        "MODEL_REGISTRY_PATH",
        "FEATURE_STORE_PATH",
        "DATA_LAKE_PATH"
    ]

    missing = [p for p in required_paths if not _validate_not_empty(p)]

    if missing:
        raise RuntimeError(f"Missing critical path env vars: {missing}")

    for key in required_paths:

        path = os.getenv(key)

        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to access/create path '{path}' from {key}. Error: {e}"
            )


########################################################
# MARKET PROVIDER VALIDATION
########################################################

def _validate_market_providers():

    providers = {
        "twelvedata": "TWELVEDATA_API_KEY",
        "alphavantage": "ALPHAVANTAGE_API_KEY",
    }

    available = []

    for provider, key in providers.items():

        if _validate_not_empty(key) and len(os.getenv(key)) >= 8:
            available.append(provider)

    # Yahoo fallback always available
    available.append("yahoo")

    if len(available) == 1:
        logger.warning("Only Yahoo provider available - reliability reduced.")

    logger.info("Market providers detected -> %s", available)

    os.environ["MARKET_PROVIDERS_ACTIVE"] = ",".join(available)


########################################################
# NEWS PROVIDER VALIDATION
########################################################

def _validate_news_provider_secrets():

    if not _as_bool(os.getenv("ENABLE_SENTIMENT"), False):
        logger.info("Sentiment disabled via ENABLE_SENTIMENT=0")
        return

    primary = os.getenv("NEWS_PROVIDER_PRIMARY", "auto").lower()

    providers = {
        "finnhub": "FINNHUB_API_KEY",
        "marketaux": "MARKETAUX_API_KEY",
        "gnews": "GNEWS_API_KEY",
    }

    if primary == "auto":

        for provider, key in providers.items():

            if _validate_not_empty(key) and len(os.getenv(key)) >= 10:

                logger.info("Auto-selected news provider -> %s", provider)
                os.environ["NEWS_PROVIDER_PRIMARY"] = provider
                return

        logger.warning("No news provider keys detected - disabling sentiment.")
        os.environ["ENABLE_SENTIMENT"] = "0"
        return

    if primary not in providers:
        raise RuntimeError(
            f"Invalid NEWS_PROVIDER_PRIMARY='{primary}'."
        )

    if not _validate_not_empty(providers[primary]):
        logger.warning(
            f"Primary news provider '{primary}' missing - disabling sentiment."
        )
        os.environ["ENABLE_SENTIMENT"] = "0"


########################################################
# ENVIRONMENT FINGERPRINT
########################################################

def _environment_fingerprint():

    payload = (
        sys.version +
        os.getenv("TWELVEDATA_API_KEY", "") +
        os.getenv("ALPHAVANTAGE_API_KEY", "") +
        os.getenv("ENABLE_SENTIMENT", "") +
        os.getenv("MODEL_REGISTRY_PATH", "")
    )

    fp = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    os.environ["ENV_FINGERPRINT"] = fp

    logger.info("Environment fingerprint -> %s", fp)


########################################################
# INIT ENV
########################################################

def init_env() -> None:
    global _ENV_INITIALIZED

    if _ENV_INITIALIZED:
        return

    with _ENV_LOCK:

        if _ENV_INITIALIZED:
            return

        _configure_logging()

        if _as_bool(os.getenv("DOTENV_ENABLED", "1")):
            _fail_fast_dotenv()
            load_dotenv(override=False)

        defaults = {
            "ENABLE_SENTIMENT": "0",
            "NEWS_PROVIDER_FAILOVER": "1",
            "NEWS_PROVIDER_PRIMARY": "auto",
            "LOG_LEVEL": "INFO",
            "LOG_DIR": "logs",
            "PYTHONHASHSEED": "42",
        }

        for k, v in defaults.items():
            os.environ.setdefault(k, v)

        _validate_paths()
        _validate_market_providers()
        _validate_news_provider_secrets()
        _environment_fingerprint()

        logger.info("Environment bootstrapped safely.")

        _ENV_INITIALIZED = True


########################################################
# ACCESSORS
########################################################

def get_env(key: str, default=None, required: bool = False):
    val = os.getenv(key, default)

    if required and val is None:
        raise RuntimeError(f"Required env missing: {key}")

    return val


def get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default


def get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


def get_bool(key: str, default: bool = False) -> bool:
    return _as_bool(os.getenv(key), default)
