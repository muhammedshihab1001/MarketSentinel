import os
import logging
import threading
from dotenv import load_dotenv, dotenv_values


logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False
_ENV_LOCK = threading.Lock()


########################################################
# INTERNAL HELPERS
########################################################

def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_not_empty(key: str) -> bool:
    val = os.getenv(key)

    if val is None:
        return False

    return bool(val.strip())


def _fail_fast_dotenv():
    """
    Detect malformed .env BEFORE loading it.
    python-dotenv logs parse errors but does not raise.
    We enforce crash-on-malformed-config.
    """

    if not os.path.exists(".env"):
        return

    try:
        dotenv_values(".env")
    except Exception as e:
        raise RuntimeError(
            f"Malformed .env file detected. Refusing to boot.\nError: {str(e)}"
        )


def _validate_paths():
    """
    Prevent silent registry / feature store corruption.
    """

    required_paths = [
        "MODEL_REGISTRY_PATH",
        "FEATURE_STORE_PATH",
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


def _validate_sentiment_secrets():
    """
    Secrets enforced ONLY if sentiment is enabled.
    """

    if not _as_bool(os.getenv("ENABLE_SENTIMENT"), False):
        logger.info("Sentiment disabled via ENABLE_SENTIMENT=0")
        return

    required = [
        "MARKETAUX_API_KEY",
        "GNEWS_API_KEY",
    ]

    missing = [k for k in required if not _validate_not_empty(k)]

    if missing:
        raise RuntimeError(
            f"ENABLE_SENTIMENT=1 but secrets missing: {missing}"
        )

    if len(os.getenv("MARKETAUX_API_KEY")) < 10:
        raise RuntimeError("MARKETAUX_API_KEY appears invalid.")

    if len(os.getenv("GNEWS_API_KEY")) < 10:
        raise RuntimeError("GNEWS_API_KEY appears invalid.")


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
        # DOTENV SAFETY
        ####################################################

        if _as_bool(os.getenv("DOTENV_ENABLED", "1")):
            _fail_fast_dotenv()
            load_dotenv(override=False)

        ####################################################
        # SAFE DEFAULTS
        ####################################################

        defaults = {

            # provider flags
            "ENABLE_SENTIMENT": "0",
            "NEWS_PROVIDER_FAILOVER": "1",

            # dotenv behavior
            "DOTENV_ENABLED": "1",

            # tokenizer / hf stability
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "TOKENIZERS_PARALLELISM": "false",

            # deterministic hashing
            "PYTHONHASHSEED": "42",
        }

        for k, v in defaults.items():
            os.environ.setdefault(k, v)

        ####################################################
        # PATH SAFETY
        ####################################################

        _validate_paths()

        ####################################################
        # SENTIMENT SAFETY
        ####################################################

        _validate_sentiment_secrets()

        ####################################################
        # OFFLINE MODE CONTROL
        ####################################################

        if _as_bool(os.getenv("ALLOW_OFFLINE_MODE"), False):

            if not (
                _as_bool(os.getenv("CI"))
                or "PYTEST_CURRENT_TEST" in os.environ
            ):
                raise RuntimeError(
                    "ALLOW_OFFLINE_MODE is forbidden outside CI/tests."
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
