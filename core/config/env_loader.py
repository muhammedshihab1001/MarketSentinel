
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
_ENV_LOCK        = threading.Lock()

# Provider priority — must match MarketProviderRouter registration order
_PROVIDER_PRIORITY = ["yahoo", "alphavantage", "twelvedata"]

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
    level     = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file logging — off by default, useful in production
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

    logger.info("Logging initialised | level=%s", log_level)


# ============================================================
# HELPERS
# ============================================================

def _as_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse a string env value to bool. Returns default for None."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_path(key: str, default: str) -> None:
    """
    Guarantee the directory referenced by env[key] exists.
    Creates it if missing, raises RuntimeError if it cannot be created.
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
    Check which fallback providers have API keys set.
    Yahoo is always available (no key needed).

    Priority order written to MARKET_PROVIDERS_ACTIVE:
        yahoo → alphavantage → twelvedata
    """
    optional_providers = {
        "alphavantage": "ALPHAVANTAGE_API_KEY",
        "twelvedata":   "TWELVEDATA_API_KEY",
    }

    # Build in priority order — yahoo always first
    active: List[str] = ["yahoo"]

    for provider in ["alphavantage", "twelvedata"]:
        key = optional_providers[provider]
        if os.getenv(key):
            active.append(provider)
        else:
            logger.warning(
                "Market provider unavailable → %s | missing env: %s",
                provider, key,
            )

    os.environ["MARKET_PROVIDERS_ACTIVE"] = ",".join(active)

    logger.info(
        "Market providers active | priority=%s",
        " → ".join(active),
    )

    if len(active) == 1:
        logger.warning(
            "Only Yahoo Finance is available. "
            "Set ALPHAVANTAGE_API_KEY or TWELVEDATA_API_KEY for fallback coverage."
        )


# ============================================================
# LLM / OPENAI VALIDATION
# ============================================================

def _validate_llm() -> None:
    """
    Validate OpenAI config when LLM_ENABLED=1.
    Raises immediately on startup so misconfiguration is caught early
    rather than at the first inference request.
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
            "OPENAI_MODEL='%s' is not a recognised model. "
            "Known models: %s. Proceeding anyway.",
            model, sorted(_KNOWN_OPENAI_MODELS),
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
    Generate a short SHA-256 fingerprint of the active configuration.
    Written to ENV_FINGERPRINT — useful for comparing deployments in logs.
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

    Thread-safe and idempotent — safe to call from multiple
    entry points (API server, training scripts, tests).
    """
    global _ENV_INITIALIZED

    if _ENV_INITIALIZED:
        return

    with _ENV_LOCK:
        if _ENV_INITIALIZED:
            return

        # ── 1. Load .env file first so all subsequent getenv calls see it ────
        if os.getenv("DOTENV_ENABLED", "1") == "1":
            load_dotenv(override=False)

        # ── 2. Configure logging before anything else logs ───────────────────
        _configure_logging()

        # ── 3. Apply defaults (never overrides values already in environment) ─
        defaults = {
            # Artifact paths
            "MODEL_REGISTRY_PATH":    "artifacts/registry",
            "FEATURE_STORE_PATH":     "artifacts/feature_store",
            "XGB_REGISTRY_DIR":       "artifacts/xgboost",

            # LLM / OpenAI
            "LLM_ENABLED":            "0",
            "OPENAI_MODEL":           "gpt-4o-mini",
            "OPENAI_TIMEOUT":         "12",
            "LLM_RATE_LIMIT_PER_MIN": "30",
            "LLM_CACHE_ENABLED":      "1",
            "LLM_CACHE_TTL_SEC":      "120",
            "LLM_AUDIT_ENABLED":      "1",

            # Data layer provider flags
            "YAHOO_SOFT_FAIL":        "1",
            "YFINANCE_SOFT_MODE":     "1",
            "YAHOO_MAX_CONCURRENT":   "1",

            # Logging
            "LOG_LEVEL":              "INFO",
            "ENABLE_FILE_LOGGING":    "0",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)

        # ── 4. Ensure artifact directories exist ─────────────────────────────
        _ensure_path("MODEL_REGISTRY_PATH", "artifacts/registry")
        _ensure_path("FEATURE_STORE_PATH",  "artifacts/feature_store")
        _ensure_path("XGB_REGISTRY_DIR",    "artifacts/xgboost")

        # ── 5. Validation passes ──────────────────────────────────────────────
        _validate_market_providers()
        _validate_llm()

        # ── 6. Fingerprint ────────────────────────────────────────────────────
        _environment_fingerprint()

        logger.info("Environment initialised successfully.")
        _ENV_INITIALIZED = True


# ============================================================
# PUBLIC ACCESSORS  (use these instead of os.getenv directly)
# ============================================================

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Read a string env value.
    Raises RuntimeError if required=True and the key is missing.
    """
    val = os.getenv(key, default)
    if required and val is None:
        raise RuntimeError(f"Missing required environment variable: '{key}'")
    return val


def get_int(key: str, default: int) -> int:
    """Read an integer env value. Raises on invalid format."""
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
    """Read a float env value. Raises on invalid format."""
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
    """Read a boolean env value (1/true/yes/on → True)."""
    return _as_bool(os.getenv(key), default)


def get_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    """
    Read a comma-separated env value as a Python list.
    Example: MARKET_PROVIDERS_ACTIVE=yahoo,alphavantage → ['yahoo', 'alphavantage']
    """
    val = os.getenv(key)
    if not val:
        return default if default is not None else []
    return [item.strip() for item in val.split(",") if item.strip()]