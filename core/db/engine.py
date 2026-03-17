"""
MarketSentinel — Database Engine & Session Factory

Provides SQLAlchemy async-compatible engine with connection pooling,
health checks, and graceful fallback for local development.

Usage:
    from core.db.engine import get_session, get_engine, init_db

    # Initialize tables on startup
    init_db()

    # Use in request handlers
    with get_session() as session:
        session.query(OHLCVDaily).filter_by(ticker="AAPL").all()
"""

import os
import logging
import time
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError

from core.logging.logger import get_logger

logger = get_logger(__name__)


# ─── Base Model ───────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ─── Configuration ────────────────────────────────────────────

def _build_database_url() -> str:
    """
    Build database URL from environment variables.

    Supports:
      - DATABASE_URL (full connection string)
      - Individual POSTGRES_* variables (for docker-compose)
    """

    full_url = os.getenv("DATABASE_URL")

    if full_url:
        return full_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "sentinel")
    password = os.getenv("POSTGRES_PASSWORD", "sentinel")
    db_name = os.getenv("POSTGRES_DB", "marketsentinel")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    return url


# ─── Engine Singleton ─────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_engine():
    """
    Return the shared SQLAlchemy engine (singleton).

    Pool settings tuned for inference workload:
      - pool_size=5   : enough for concurrent API requests
      - max_overflow=3 : burst capacity during batch fetches
      - pool_timeout=10: fail fast if DB is overloaded
      - pool_recycle=1800: refresh connections every 30min
    """

    global _engine

    if _engine is not None:
        return _engine

    url = _build_database_url()

    _engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=3,
        pool_timeout=10,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "0") == "1",
    )

    # Log slow queries (> 500ms)
    @event.listens_for(_engine, "before_cursor_execute")
    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("query_start", []).append(time.time())

    @event.listens_for(_engine, "after_cursor_execute")
    def _after_execute(conn, cursor, statement, parameters, context, executemany):
        total = time.time() - conn.info["query_start"].pop()
        if total > 0.5:
            logger.warning(
                "Slow query detected | duration=%.3fs | statement=%s",
                total,
                statement[:200],
                extra={
                    "component": "db.engine",
                    "duration_sec": round(total, 3),
                },
            )

    logger.info(
        "Database engine initialized | pool_size=5 | host=%s",
        url.split("@")[-1].split("/")[0] if "@" in url else "localhost",
        extra={"component": "db.engine"},
    )

    return _engine


def get_session_factory() -> sessionmaker:
    """Return the shared session factory."""

    global _SessionFactory

    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    return _SessionFactory


# ─── Session Context Manager ─────────────────────────────────

@contextmanager
def get_session():
    """
    Provide a transactional session scope.

    Usage:
        with get_session() as session:
            session.add(record)
            # auto-commits on success, auto-rollbacks on exception
    """

    factory = get_session_factory()
    session: Session = factory()

    try:
        yield session
        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


# ─── Initialization ──────────────────────────────────────────

def init_db():
    """
    Create all tables that don't exist yet.

    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    Call this once during application startup.
    """

    engine = get_engine()

    Base.metadata.create_all(engine)

    logger.info(
        "Database tables initialized",
        extra={"component": "db.engine"},
    )


def check_db_health() -> dict:
    """
    Quick health check for the database connection.

    Returns:
        {"status": "healthy", "latency_ms": float}
        {"status": "unhealthy", "error": str}
    """

    try:
        start = time.time()

        with get_session() as session:
            session.execute(text("SELECT 1"))

        latency_ms = round((time.time() - start) * 1000, 2)

        return {"status": "healthy", "latency_ms": latency_ms}

    except Exception as exc:

        logger.error(
            "Database health check failed | error=%s",
            exc,
            extra={"component": "db.engine"},
        )

        return {"status": "unhealthy", "error": str(exc)}


def dispose_engine():
    """Dispose the engine and connection pool. For clean shutdown."""

    global _engine, _SessionFactory

    if _engine:
        _engine.dispose()
        _engine = None
        _SessionFactory = None

        logger.info(
            "Database engine disposed",
            extra={"component": "db.engine"},
        )