import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False


########################################################
# INTERNAL HELPERS
########################################################

def _as_bool(value: str | None, default: bool = False) -> bool:

    if value is None:
        return default

    value = value.strip().lower()

    return value in {"1", "true", "yes", "on"}


def _validate_not_empty(key: str):

    val = os.getenv(key)

    if val is None:
        return False

    if not val.strip():
        return False

    return True


########################################################
# INIT ENV
########################################################

def init_env() -> None:

    global _ENV_INITIALIZED

    if _ENV_INITIALIZED:
        return

    ####################################################
    # LOAD .env (only outside docker)
    ####################################################

    if _as_bool(os.getenv("DOTENV_ENABLED", "1")):
        load_dotenv(override=False)

    ####################################################
    # SAFE DEFAULTS
    ####################################################

    defaults = {

        # news system
        "NEWS_MAX_ITEMS": "250",
        "NEWS_LOOKBACK_DAYS": "3",

        # provider controls
        "NEWS_PROVIDER_FAILOVER": "1",
        "ALLOW_OFFLINE_MODE": "0",

        # dotenv behavior
        "DOTENV_ENABLED": "1",

        # huggingface stability
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",

        # training stability
        "PYTHONHASHSEED": "42",
    }

    for k, v in defaults.items():
        os.environ.setdefault(k, v)

    ####################################################
    # REQUIRED SECRETS
    ####################################################

    required = [
        "MARKETAUX_API_KEY",
        "GNEWS_API_KEY",
    ]

    missing = [
        k for k in required
        if not _validate_not_empty(k)
    ]

    ####################################################
    # CI / TEST BYPASS
    ####################################################

    if missing:

        if _as_bool(os.getenv("CI")):
            logger.warning(
                "Missing secrets in CI — external providers disabled."
            )
            os.environ["NEWS_PROVIDER_FAILOVER"] = "0"
            os.environ["ALLOW_OFFLINE_MODE"] = "1"
            _ENV_INITIALIZED = True
            return

        if "PYTEST_CURRENT_TEST" in os.environ:
            logger.warning(
                "Missing secrets in tests — APIs disabled."
            )
            os.environ["NEWS_PROVIDER_FAILOVER"] = "0"
            os.environ["ALLOW_OFFLINE_MODE"] = "1"
            _ENV_INITIALIZED = True
            return

        if _as_bool(os.getenv("ALLOW_OFFLINE_MODE")):
            logger.warning(
                "Running in OFFLINE_MODE — news providers disabled."
            )
            os.environ["NEWS_PROVIDER_FAILOVER"] = "0"
            _ENV_INITIALIZED = True
            return

        raise RuntimeError(
            f"Missing environment variables: {missing}"
        )

    ####################################################
    # HARD SECRET VALIDATION
    ####################################################

    if len(os.getenv("MARKETAUX_API_KEY")) < 10:
        raise RuntimeError("MARKETAUX_API_KEY appears invalid.")

    if len(os.getenv("GNEWS_API_KEY")) < 10:
        raise RuntimeError("GNEWS_API_KEY appears invalid.")

    ####################################################
    # LOG FINAL MODE
    ####################################################

    logger.info(
        "Environment initialized | failover=%s | offline=%s",
        os.getenv("NEWS_PROVIDER_FAILOVER"),
        os.getenv("ALLOW_OFFLINE_MODE"),
    )

    _ENV_INITIALIZED = True


########################################################
# ACCESSORS
########################################################

def get_env(
    key: str,
    default=None,
    required: bool = False
):

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
        raise RuntimeError(
            f"Invalid integer for env '{key}'"
        )


def get_float(key: str, default: float) -> float:

    val = os.getenv(key)

    if val is None:
        return default

    try:
        return float(val)
    except Exception:
        raise RuntimeError(
            f"Invalid float for env '{key}'"
        )


def get_bool(key: str, default: bool = False) -> bool:

    return _as_bool(os.getenv(key), default)
