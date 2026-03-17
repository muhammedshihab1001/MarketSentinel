"""
MarketSentinel — Professional Logging System

Features:
  - Dedicated log file: logs/marketsentinel.log (all levels)
  - Issue tracker file: logs/issues.log (WARNING+ only)
  - Console output with color coding
  - Every log entry includes: timestamp, level, file, line,
    function, component, and message
  - JSON-structured metadata via 'extra' dict
  - Automatic log rotation (10MB max, 5 backups)

Usage:
    from core.logging.logger import get_logger

    logger = get_logger(__name__)

    logger.info(
        "Price data fetched | ticker=%s rows=%d",
        "AAPL", 250,
        extra={"component": "data_fetcher", "function": "fetch"},
    )

    logger.warning(
        "Slow query | duration=%.2fs",
        1.23,
        extra={"component": "db.engine"},
    )

Log output format:
    2025-01-15 14:23:45.123 | INFO     | core.data.data_fetcher:fetch:142 | [data_fetcher] Price data fetched | ticker=AAPL rows=250
    2025-01-15 14:23:46.456 | WARNING  | core.db.engine:_after_execute:89 | [db.engine] Slow query | duration=1.23s
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAIN_LOG_FILE = LOG_DIR / "marketsentinel.log"
ISSUE_LOG_FILE = LOG_DIR / "issues.log"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


# ─── Custom Formatter ────────────────────────────────────────

class DetailedFormatter(logging.Formatter):
    """
    Format: TIMESTAMP | LEVEL | MODULE:FUNCTION:LINE | [COMPONENT] MESSAGE

    The component field comes from extra={"component": "..."}.
    If not provided, uses the module name.
    """

    FORMAT = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
        "%(module)s:%(funcName)s:%(lineno)d | "
        "[%(component)s] %(message)s"
    )

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def format(self, record):

        if not hasattr(record, "component"):
            record.component = record.module

        return super().format(record)


# ─── Console Formatter (with color) ──────────────────────────

class ColorFormatter(DetailedFormatter):
    """Adds ANSI color codes to console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red background
    }

    RESET = "\033[0m"

    def format(self, record):

        formatted = super().format(record)

        color = self.COLORS.get(record.levelname, "")

        if color:
            return f"{color}{formatted}{self.RESET}"

        return formatted


# ─── Setup Flag ───────────────────────────────────────────────

_initialized = False


def setup_logging():
    """
    Initialize the logging system.

    Creates three handlers:
      1. Console (stdout) — all levels, colored
      2. Main log file — all levels, detailed
      3. Issue log file — WARNING+ only, for tracking problems

    Safe to call multiple times — only runs once.
    """

    global _initialized

    if _initialized:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    formatter = DetailedFormatter(
        fmt=DetailedFormatter.FORMAT,
        datefmt=DetailedFormatter.DATE_FORMAT,
    )

    color_formatter = ColorFormatter(
        fmt=DetailedFormatter.FORMAT,
        datefmt=DetailedFormatter.DATE_FORMAT,
    )

    # ── Handler 1: Console ──

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(color_formatter)
    root_logger.addHandler(console_handler)

    # ── Handler 2: Main log file (all levels) ──

    main_file_handler = RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(formatter)
    root_logger.addHandler(main_file_handler)

    # ── Handler 3: Issue log file (WARNING+ only) ──

    issue_file_handler = RotatingFileHandler(
        ISSUE_LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    issue_file_handler.setLevel(logging.WARNING)
    issue_file_handler.setFormatter(formatter)
    root_logger.addHandler(issue_file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "yfinance", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True

    root_logger.info(
        "Logging initialized | level=%s main_log=%s issue_log=%s",
        LOG_LEVEL,
        MAIN_LOG_FILE,
        ISSUE_LOG_FILE,
    )


# ─── Logger Factory ──────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with auto-initialization.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"component": "my_module"})
    """

    setup_logging()

    return logging.getLogger(name)