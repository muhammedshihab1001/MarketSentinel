"""
MarketSentinel — Professional Logging System v2.0

Changes from v1.0:
  NEW: Dev vs Production log separation
       Dev  (APP_ENV=development): DEBUG level, console + all 4 files
       Prod (APP_ENV=production):  INFO level, files only (no console),
                                   no stack traces in HTTP responses

  NEW: Full file PATH in every log entry (was module name only)
       Format: core/data/data_fetcher.py:fetch:142
       Previously: data_fetcher:fetch:142
       
  NEW: logs/debug.log — DEBUG only, dev mode only (never in prod)
  NEW: logs/access.log — HTTP request log (Uvicorn access events)
  KEPT: logs/marketsentinel.log — INFO+ all environments
  KEPT: logs/issues.log — WARNING+ all environments
  KEPT: RotatingFileHandler 10MB / 5 backups
  KEPT: Suppression of noisy third-party loggers

Usage:
    from core.logging.logger import get_logger

    logger = get_logger(__name__)

    logger.info(
        "Price data fetched | ticker=%s rows=%d",
        "AAPL", 250,
        extra={"component": "data_fetcher", "function": "fetch"},
    )

Log format (all environments):
    2026-03-24 10:08:35.362 | INFO     | core/data/market_data_service.py:get_price_data:210 | [market_data_service] Market data served

Log format fields:
    timestamp     — YYYY-MM-DD HH:MM:SS.mmm
    level         — DEBUG / INFO / WARNING / ERROR / CRITICAL
    filepath:func:line — full relative path from project root
    [component]   — from extra={"component": "..."} or module name
    message       — log message with | separated key=value pairs
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ─── Environment Detection ────────────────────────────────────

APP_ENV = os.getenv("APP_ENV", "development").lower()
IS_PRODUCTION = APP_ENV == "production"
IS_DEVELOPMENT = not IS_PRODUCTION

LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

# In production always enforce minimum INFO — never DEBUG in prod
if IS_PRODUCTION:
    LOG_LEVEL = max(LOG_LEVEL, logging.INFO)

# ─── Log Directory ────────────────────────────────────────────

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAIN_LOG_FILE   = LOG_DIR / "marketsentinel.log"   # INFO+  — all envs
ISSUE_LOG_FILE  = LOG_DIR / "issues.log"           # WARNING+ — all envs
DEBUG_LOG_FILE  = LOG_DIR / "debug.log"            # DEBUG — dev only
ACCESS_LOG_FILE = LOG_DIR / "access.log"           # HTTP requests — all envs

MAX_LOG_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT  = 5                   # 5 rotated backups kept


# ─── Formatters ──────────────────────────────────────────────

class DetailedFormatter(logging.Formatter):
    """
    Format: TIMESTAMP | LEVEL | FILEPATH:FUNCTION:LINE | [COMPONENT] MESSAGE

    NEW v2.0: Uses %(pathname)s instead of %(module)s so every log
    entry shows the full file path, not just the module name.
    This makes it trivial to find which file caused an issue.

    Example output:
        2026-03-24 10:08:35.362 | INFO | core/data/market_data_service.py:get_price_data:210 | [market_data_service] Market data served

    The component field comes from extra={"component": "..."}.
    If not provided, falls back to the module name.
    """

    # NEW v2.0: %(pathname)s replaced %(module)s — full file path
    FORMAT = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
        "%(short_pathname)s:%(funcName)s:%(lineno)d | "
        "[%(component)s] %(message)s"
    )

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Project root for making paths relative
    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

    def format(self, record: logging.LogRecord) -> str:

        # ── Component field ───────────────────────────────────
        if not hasattr(record, "component"):
            record.component = record.module

        # ── Short relative file path ──────────────────────────
        # Convert absolute path to relative from project root.
        # Fallback to pathname if conversion fails.
        try:
            full_path = str(Path(record.pathname).resolve())
            if full_path.startswith(self._PROJECT_ROOT):
                rel = full_path[len(self._PROJECT_ROOT):].lstrip("/\\")
            else:
                rel = record.pathname
        except Exception:
            rel = record.pathname

        record.short_pathname = rel

        return super().format(record)


class ColorFormatter(DetailedFormatter):
    """
    Adds ANSI color codes to console output (dev mode only).
    Production: plain text, no color codes (avoids polluting log aggregators).
    """

    COLORS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[41m",    # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{formatted}{self.RESET}" if color else formatted


class ProductionFormatter(DetailedFormatter):
    """
    Production formatter — no color codes, no stack traces in output.
    Stack traces still appear in issues.log but are suppressed from
    the main log to keep it clean for log aggregators (GCP Logging etc).
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        # In production: suppress stack traces from main log
        # They still appear in issues.log via the WARNING+ handler
        if record.exc_info and record.levelno < logging.WARNING:
            record.exc_info = None
            record.exc_text = None
        return formatted


# ─── Setup Flag ───────────────────────────────────────────────

_initialized = False


def setup_logging() -> None:
    """
    Initialize the logging system. Thread-safe, idempotent.

    Development mode (APP_ENV=development):
      - Console: DEBUG+ with colors
      - marketsentinel.log: DEBUG+
      - issues.log: WARNING+
      - debug.log: DEBUG only (verbose, dev-only)
      - access.log: INFO (HTTP requests)

    Production mode (APP_ENV=production):
      - Console: DISABLED (files only)
      - marketsentinel.log: INFO+
      - issues.log: WARNING+
      - debug.log: NOT CREATED
      - access.log: INFO (HTTP requests)
    """

    global _initialized

    if _initialized:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Root always DEBUG — handlers filter
    root_logger.handlers.clear()

    plain_formatter = DetailedFormatter(
        fmt=DetailedFormatter.FORMAT,
        datefmt=DetailedFormatter.DATE_FORMAT,
    )

    prod_formatter = ProductionFormatter(
        fmt=DetailedFormatter.FORMAT,
        datefmt=DetailedFormatter.DATE_FORMAT,
    )

    color_formatter = ColorFormatter(
        fmt=DetailedFormatter.FORMAT,
        datefmt=DetailedFormatter.DATE_FORMAT,
    )

    # ── Handler 1: Console ────────────────────────────────────
    # Dev: DEBUG+ with colors
    # Prod: DISABLED — no console output in production
    if IS_DEVELOPMENT:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(color_formatter)
        root_logger.addHandler(console_handler)

    # ── Handler 2: Main log file (INFO+, all environments) ────
    # Primary log — what you tail in production to monitor the system
    main_formatter = plain_formatter if IS_DEVELOPMENT else prod_formatter

    main_file_handler = RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(main_formatter)
    root_logger.addHandler(main_file_handler)

    # ── Handler 3: Issues log (WARNING+, all environments) ────
    # Dedicated file for warnings + errors — used for alerting in prod
    issue_file_handler = RotatingFileHandler(
        ISSUE_LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    issue_file_handler.setLevel(logging.WARNING)
    issue_file_handler.setFormatter(plain_formatter)
    root_logger.addHandler(issue_file_handler)

    # ── Handler 4: Debug log (DEBUG only, dev mode only) ──────
    # Full verbose output — never created in production
    # Useful for tracing inference pipeline, feature engineering etc
    if IS_DEVELOPMENT:
        debug_file_handler = RotatingFileHandler(
            DEBUG_LOG_FILE,
            maxBytes=MAX_LOG_BYTES,
            backupCount=2,   # Only 2 backups — debug logs are large
            encoding="utf-8",
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(plain_formatter)

        # Only capture DEBUG — don't duplicate INFO+ (already in main log)
        debug_file_handler.addFilter(
            lambda r: r.levelno == logging.DEBUG
        )
        root_logger.addHandler(debug_file_handler)

    # ── Handler 5: Access log (HTTP requests, all environments) ─
    # Separate file for HTTP request logs — avoids polluting main log
    access_file_handler = RotatingFileHandler(
        ACCESS_LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    access_file_handler.setLevel(logging.INFO)
    access_file_handler.setFormatter(plain_formatter)

    # Only capture access logs from Uvicorn access logger
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addHandler(access_file_handler)
    access_logger.propagate = False   # Don't duplicate in root logger

    # ── Suppress noisy third-party loggers ────────────────────
    noisy = [
        "urllib3", "yfinance", "httpx", "httpcore",
        "asyncio", "multipart", "passlib",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    # In production also suppress sqlalchemy INFO (very verbose)
    if IS_PRODUCTION:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _initialized = True

    # First log entry — confirms environment and log files
    root_logger.info(
        "Logging initialized | level=%s env=%s "
        "main_log=%s issue_log=%s debug_log=%s access_log=%s",
        LOG_LEVEL_STR,
        APP_ENV,
        MAIN_LOG_FILE,
        ISSUE_LOG_FILE,
        DEBUG_LOG_FILE if IS_DEVELOPMENT else "disabled",
        ACCESS_LOG_FILE,
    )


# ─── Logger Factory ──────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. Auto-initializes on first call.

    Usage:
        logger = get_logger(__name__)

        # Basic log
        logger.info("Snapshot complete | tickers=%d", 100)

        # With component tag (appears as [component] in log line)
        logger.warning(
            "Slow query | duration=%.2fs", 1.23,
            extra={"component": "db.engine"},
        )

        # Error with exception
        try:
            risky_call()
        except Exception:
            logger.exception("Inference failed | ticker=%s", "AAPL")
    """

    setup_logging()
    return logging.getLogger(name)


# ─── Environment Info (for startup diagnostics) ───────────────

def log_environment_summary() -> None:
    """
    Log a summary of the current logging configuration.
    Call once during app startup after setup_logging().
    """
    logger = get_logger("marketsentinel.logger")

    logger.info(
        "Log configuration | env=%s | "
        "console=%s | main=%s | issues=%s | debug=%s | access=%s",
        APP_ENV,
        "enabled" if IS_DEVELOPMENT else "disabled",
        MAIN_LOG_FILE,
        ISSUE_LOG_FILE,
        DEBUG_LOG_FILE if IS_DEVELOPMENT else "disabled",
        ACCESS_LOG_FILE,
    )

    if IS_PRODUCTION:
        logger.info(
            "Production logging: console disabled, "
            "stack traces suppressed from INFO log, "
            "sqlalchemy engine logging suppressed"
        )