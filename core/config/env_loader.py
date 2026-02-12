import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False


########################################################
# SAFE BOOL PARSER
########################################################

def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "on"}


########################################################
# INIT ENV (FAIL FAST)
########################################################

def init_env() -> None:
    """
    Institutional environment bootstrap.

    Guarantees:
    - single execution
    - fail-fast secrets validation
    - docker-safe
    - CI-safe
    - test-safe
    """

    global _ENV_INITIALIZED

    if _ENV_INITIALIZED:
        return

    # Only load .env if NOT inside docker
    # Docker should inject env vars directly.
    if os.getenv("DOTENV_ENABLED", "1") == "1":
        load_dotenv(override=False)

    required = [
        "MARKETAUX_API_KEY",
        "GNEWS_API_KEY",
    ]

    optional_defaults = {
        "NEWS_MAX_ITEMS": "250",
        "NEWS_LOOKBACK_DAYS": "3",
        "DOTENV_ENABLED": "1",
        "NEWS_PROVIDER_FAILOVER": "1",
    }

    # Apply defaults safely
    for key, value in optional_defaults.items():
        os.environ.setdefault(key, value)

    missing = [k for k in required if not os.getenv(k)]

    ####################################################
    # CI / TEST BYPASS
    ####################################################

    if missing:

        if _as_bool(os.getenv("CI")):
            logger.warning(
                "Missing secrets in CI — bypassing external news providers."
            )
            os.environ["NEWS_PROVIDER_FAILOVER"] = "0"
            _ENV_INITIALIZED = True
            return

        if "PYTEST_CURRENT_TEST" in os.environ:
            logger.warning(
                "Missing secrets in tests — external APIs disabled."
            )
            os.environ["NEWS_PROVIDER_FAILOVER"] = "0"
            _ENV_INITIALIZED = True
            return

        raise RuntimeError(
            f"Missing environment variables: {missing}"
        )

    ####################################################
    # HARD VALIDATION
    ####################################################

    if len(os.getenv("MARKETAUX_API_KEY", "")) < 10:
        raise RuntimeError("MARKETAUX_API_KEY appears invalid.")

    if len(os.getenv("GNEWS_API_KEY", "")) < 10:
        raise RuntimeError("GNEWS_API_KEY appears invalid.")

    logger.info("Environment initialized successfully.")

    _ENV_INITIALIZED = True


########################################################
# SAFE ACCESSORS (VERY IMPORTANT)
########################################################

def get_env(key: str, default=None, required: bool = False):

    value = os.getenv(key, default)

    if required and value is None:
        raise RuntimeError(f"Required env missing: {key}")

    return value


def get_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def get_bool(key: str, default: bool = False) -> bool:
    return _as_bool(os.getenv(key), default)
