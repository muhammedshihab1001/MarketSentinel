import os
import logging
import threading
import hashlib
import sys
from dotenv import load_dotenv, dotenv_values


logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False
_ENV_LOCK = threading.Lock()


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
# DOTENV HARD SAFETY
########################################################

def _fail_fast_dotenv():

    if not os.path.exists(".env"):
        logger.warning(".env not found — relying on system environment.")
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
                f"Failed to create/access path '{path}' from {key}. Error: {e}"
            )


########################################################
# 🔥 MARKET PROVIDER VALIDATION (NEW — VERY IMPORTANT)
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

    # Yahoo always exists (no key required)
    available.append("yahoo")

    if len(available) == 1 and available[0] == "yahoo":

        logger.warning(
            "⚠ ONLY Yahoo provider available. "
            "Production reliability reduced."
        )

    logger.info(
        "Market providers detected → %s",
        available
    )

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

    ####################################################
    # AUTO MODE
    ####################################################

    if primary == "auto":

        for provider, key in providers.items():

            if _validate_not_empty(key) and len(os.getenv(key)) >= 10:

                logger.info(
                    "Auto-selected news provider → %s",
                    provider
                )

                os.environ["NEWS_PROVIDER_PRIMARY"] = provider
                return

        logger.warning(
            "No news provider keys detected — sentiment auto-disabled."
        )

        os.environ["ENABLE_SENTIMENT"] = "0"
        return

    ####################################################
    # EXPLICIT MODE
    ####################################################

    if primary not in providers:
        raise RuntimeError(
            f"Invalid NEWS_PROVIDER_PRIMARY='{primary}'. "
            f"Valid options: {list(providers.keys()) + ['auto']}"
        )

    primary_key = providers[primary]

    if not _validate_not_empty(primary_key):

        logger.warning(
            f"Primary news provider '{primary}' disabled — missing {primary_key}."
        )

        os.environ["ENABLE_SENTIMENT"] = "0"
        return

    if len(os.getenv(primary_key)) < 10:
        raise RuntimeError(f"{primary_key} appears invalid.")


########################################################
# ENVIRONMENT FINGERPRINT (VERY HIGH VALUE)
########################################################

def _environment_fingerprint():

    payload = (
        sys.version +
        os.getenv("TWELVEDATA_API_KEY", "") +
        os.getenv("ALPHAVANTAGE_API_KEY", "") +
        os.getenv("ENABLE_SENTIMENT", "")
    )

    fp = hashlib.sha256(payload.encode()).hexdigest()[:16]

    os.environ["ENV_FINGERPRINT"] = fp

    logger.info("Environment fingerprint → %s", fp)


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

        ####################################################
        # DOTENV
        ####################################################

        if _as_bool(os.getenv("DOTENV_ENABLED", "1")):
            _fail_fast_dotenv()
            load_dotenv(override=False)

        ####################################################
        # SAFE DEFAULTS
        ####################################################

        defaults = {

            "ENABLE_SENTIMENT": "0",
            "NEWS_PROVIDER_FAILOVER": "1",
            "NEWS_PROVIDER_PRIMARY": "auto",

            "DOTENV_ENABLED": "1",

            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "TOKENIZERS_PARALLELISM": "false",

            "PYTHONHASHSEED": "42",
        }

        for k, v in defaults.items():
            os.environ.setdefault(k, v)

        ####################################################
        # VALIDATIONS
        ####################################################

        _validate_paths()
        _validate_market_providers()
        _validate_news_provider_secrets()
        _environment_fingerprint()

        ####################################################
        # OFFLINE MODE SAFETY
        ####################################################

        if _as_bool(os.getenv("ALLOW_OFFLINE_MODE"), False):

            if not (
                _as_bool(os.getenv("CI"))
                or "PYTEST_CURRENT_TEST" in os.environ
            ):
                raise RuntimeError(
                    "ALLOW_OFFLINE_MODE forbidden outside CI/tests."
                )

            logger.warning("Running in OFFLINE MODE.")

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

    if val is None:
        return default

    try:
        return int(val)
    except Exception:
        raise RuntimeError(f"Invalid integer for env '{key}'")


def get_float(key: str, default: float) -> float:
    val = os.getenv(key)

    if val is None:
        return default

    try:
        return float(val)
    except Exception:
        raise RuntimeError(f"Invalid float for env '{key}'")


def get_bool(key: str, default: bool = False) -> bool:
    return _as_bool(os.getenv(key), default)
