import hashlib
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from dotenv import load_dotenv

logger = logging.getLogger("marketsentinel.env")

_ENV_INITIALIZED = False
_ENV_LOCK = threading.Lock()

# Valid OpenAI model strings accepted by this project
_KNOWN_OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
}


# ============================================================
# LOGGING CONFIG  (CI + Docker safe)
# ============================================================

def _configure_logging() -> None:
    """Set up root logger. Idempotent — safe to call multiple times."""

    if logging.getLogger().handlers:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file logging
    if os.getenv("ENABLE_FILE_LOGGING", "0") == "1":
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)

        fh = RotatingFileHandler(
            os.path.join(log_dir, "marketsentinel.log"),
            maxBytes=10_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)

    logger.info(
        "Logging initialised | level=%s | python=%s | pid=%s",
        log_level,
        sys.version.split()[0],
        os.getpid(),
    )


# ============================================================
# HELPERS
# ============================================================

def _as_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse a string env value to bool."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_path(key: str, default: str) -> None:
    """
    Guarantee the directory referenced by env[key] exists.
    """
    path = os.getenv(key, default)
    os.environ.setdefault(key, path)

    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create/access path for {key}='{path}': {exc}"
        ) from exc


# ============================================================
# MARKET PROVIDER VALIDATION
# ============================================================

def _validate_market_providers() -> None:
    """
    Validate available market data providers.

    Yahoo Finance is always available.
    TwelveData is optional.
    """

    active: List[str] = ["yahoo"]

    if os.getenv("TWELVEDATA_API_KEY"):
        active.append("twelvedata")
    else:
        logger.warning(
            "TwelveData provider unavailable | missing env: TWELVEDATA_API_KEY"
        )

    os.environ["MARKET_PROVIDERS_ACTIVE"] = ",".join(active)

    logger.info(
        "Market providers active | priority=%s",
        " → ".join(active),
    )


# ============================================================
# LLM / OPENAI VALIDATION
# ============================================================

def _validate_llm() -> None:
    """
    Validate OpenAI config when LLM_ENABLED=1.
    """
    llm_enabled = _as_bool(os.getenv("LLM_ENABLED"), default=False)

    if not llm_enabled:
        logger.info("LLM disabled (LLM_ENABLED=0). AI explanations unavailable.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_ENABLED=1 but OPENAI_API_KEY is not set. "
            "Either set the key or disable LLM with LLM_ENABLED=0."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if model not in _KNOWN_OPENAI_MODELS:
        logger.warning(
            "OPENAI_MODEL='%s' is not a recognised model. Known models: %s",
            model,
            sorted(_KNOWN_OPENAI_MODELS),
        )

    logger.info(
        "LLM enabled | model=%s | timeout=%ss | rate_limit=%s/min",
        model,
        os.getenv("OPENAI_TIMEOUT", "12"),
        os.getenv("LLM_RATE_LIMIT_PER_MIN", "30"),
    )


# ============================================================
# ENVIRONMENT FINGERPRINT
# ============================================================

def _environment_fingerprint() -> None:
    """
    Generate a short SHA256 fingerprint of active configuration.
    """

    payload = "".join([
        sys.version,
        os.getenv("MODEL_REGISTRY_PATH", ""),
        os.getenv("FEATURE_STORE_PATH", ""),
        os.getenv("XGB_REGISTRY_DIR", ""),
        os.getenv("LLM_ENABLED", ""),
        os.getenv("OPENAI_MODEL", ""),
        os.getenv("LOG_LEVEL", ""),
        os.getenv("MARKET_PROVIDERS_ACTIVE", ""),
        os.getenv("YAHOO_SOFT_FAIL", ""),
        os.getenv("YFINANCE_SOFT_MODE", ""),
    ])

    fp = hashlib.sha256(payload.encode()).hexdigest()[:12]

    os.environ["ENV_FINGERPRINT"] = fp

    logger.info("Environment fingerprint → %s", fp)


# ============================================================
# MAIN INIT
# ============================================================

def init_env() -> None:
    """
    Initialise the full application environment.
    Thread-safe and idempotent.
    """

    global _ENV_INITIALIZED

    if _ENV_INITIALIZED:
        return

    with _ENV_LOCK:

        if _ENV_INITIALIZED:
            return

        if os.getenv("DOTENV_ENABLED", "1") == "1":
            load_dotenv(override=False)

        _configure_logging()

        defaults = {

            # Artifact paths
            "MODEL_REGISTRY_PATH": "artifacts/registry",
            "FEATURE_STORE_PATH": "artifacts/feature_store",
            "XGB_REGISTRY_DIR": "artifacts/xgboost",

            # LLM
            "LLM_ENABLED": "0",
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_TIMEOUT": "12",
            "LLM_RATE_LIMIT_PER_MIN": "30",
            "LLM_CACHE_ENABLED": "1",
            "LLM_CACHE_TTL_SEC": "120",
            "LLM_AUDIT_ENABLED": "1",

            # Data layer
            "YAHOO_SOFT_FAIL": "1",
            "YFINANCE_SOFT_MODE": "1",
            "YAHOO_MAX_CONCURRENT": "1",

            # Logging
            "LOG_LEVEL": "INFO",
            "ENABLE_FILE_LOGGING": "0",
        }

        for key, value in defaults.items():
            os.environ.setdefault(key, value)

        _ensure_path("MODEL_REGISTRY_PATH", "artifacts/registry")
        _ensure_path("FEATURE_STORE_PATH", "artifacts/feature_store")
        _ensure_path("XGB_REGISTRY_DIR", "artifacts/xgboost")

        _validate_market_providers()
        _validate_llm()

        _environment_fingerprint()

        logger.info("Environment initialised successfully.")

        _ENV_INITIALIZED = True


# ============================================================
# PUBLIC ACCESSORS
# ============================================================

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Read a string env value."""

    val = os.getenv(key, default)

    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required environment variable: '{key}'")

    return val


def get_int(key: str, default: int) -> int:
    """Read integer env value."""

    val = os.getenv(key)

    if val is None:
        return default

    try:
        return int(val)
    except ValueError:
        raise RuntimeError(
            f"Environment variable '{key}' must be an integer. Got: '{val}'"
        )


def get_float(key: str, default: float) -> float:
    """Read float env value."""

    val = os.getenv(key)

    if val is None:
        return default

    try:
        return float(val)
    except ValueError:
        raise RuntimeError(
            f"Environment variable '{key}' must be a float. Got: '{val}'"
        )


def get_bool(key: str, default: bool = False) -> bool:
    """Read boolean env value."""
    return _as_bool(os.getenv(key), default)


def get_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    """Read comma-separated env value as list."""

    val = os.getenv(key)

    if not val:
        return default if default is not None else []

    return [item.strip() for item in val.split(",") if item.strip()]