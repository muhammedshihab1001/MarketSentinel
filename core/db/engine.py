"""
MarketSentinel — Database Engine & Session Factory

FIX v2 (item 17): Pool settings now read from environment variables.

FIX v3: Increased pool defaults — previous defaults (pool_size=5,
        max_overflow=3 = 8 max connections) were exhausted when
        MARKET_MAX_WORKERS=8 ran concurrent DB queries during
        snapshot computation. AMD ticker failed with QueuePool
        timeout. New defaults: pool_size=10, max_overflow=5 = 15
        max connections.

Set these in .env to override:
  DB_POOL_SIZE=10
  DB_MAX_OVERFLOW=5
  DB_POOL_TIMEOUT=10
  DB_POOL_RECYCLE=1800
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


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _build_database_url() -> str:
    full_url = os.getenv("DATABASE_URL")
    if full_url:
        return full_url
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "sentinel")
    password = os.getenv("POSTGRES_PASSWORD", "sentinel")
    db_name = os.getenv("POSTGRES_DB", "marketsentinel")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"


_engine = None
_SessionFactory = None


def get_engine():
    """
    Return the shared SQLAlchemy engine (singleton).

    FIX v3: Increased pool defaults to prevent QueuePool exhaustion
    when MARKET_MAX_WORKERS=8 runs concurrent DB queries.

    Rule of thumb: pool_size + max_overflow >= MARKET_MAX_WORKERS + 2 buffer
    Default: 10 + 5 = 15 >= 8 + 2 buffer = 10 ✓

    Env vars:
      DB_POOL_SIZE      — persistent connections (default: 10)
      DB_MAX_OVERFLOW   — burst connections above pool_size (default: 5)
      DB_POOL_TIMEOUT   — seconds to wait for a connection (default: 15)
      DB_POOL_RECYCLE   — seconds before recycling a connection (default: 1800)
    """

    global _engine

    if _engine is not None:
        return _engine

    url = _build_database_url()

    # FIX v3: Raised defaults — was 5/3/10 which caused QueuePool timeouts
    pool_size    = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "5"))
    pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "15"))
    pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))

    _engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "0") == "1",
    )

    @event.listens_for(_engine, "before_cursor_execute")
    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("query_start", []).append(time.time())

    @event.listens_for(_engine, "after_cursor_execute")
    def _after_execute(conn, cursor, statement, parameters, context, executemany):
        starts = conn.info.get("query_start", [])
        if not starts:
            return
        total = time.time() - starts.pop()
        if total > 0.5:
            logger.warning(
                "Slow query detected | duration=%.3fs | statement=%s",
                total,
                statement[:200],
                extra={"component": "db.engine", "duration_sec": round(total, 3)},
            )

    logger.info(
        "Database engine initialized | pool_size=%d | max_overflow=%d | host=%s",
        pool_size,
        max_overflow,
        url.split("@")[-1].split("/")[0] if "@" in url else "localhost",
        extra={"component": "db.engine"},
    )

    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionFactory


@contextmanager
def get_session():
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


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)

    # Run inline migrations for existing databases
    _run_migrations(engine)

    logger.info("Database tables initialized", extra={"component": "db.engine"})


def _run_migrations(engine):
    """
    Safe inline migrations for schema changes.
    Each migration is idempotent — safe to run multiple times.
    """
    migrations = [
        # FIX: schema_signature was String(32) — sha256 hex is 64 chars
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='model_predictions'
                AND column_name='schema_signature'
                AND character_maximum_length = 32
            ) THEN
                ALTER TABLE model_predictions
                ALTER COLUMN schema_signature TYPE varchar(64);
            END IF;
        END $$;
        """,
        # FIX: Add composite index on computed_features for feature version queries
        """
        CREATE INDEX IF NOT EXISTS ix_feature_version_ticker_date
            ON computed_features(feature_version, ticker, date);
        """,
    ]

    with engine.connect() as conn:
        for migration in migrations:
            try:
                conn.execute(text(migration))
                conn.commit()
            except Exception as e:
                logger.debug("Migration note: %s", e)
                conn.rollback()


def check_db_health() -> dict:
    try:
        start = time.time()
        with get_session() as session:
            session.execute(text("SELECT 1"))
        latency_ms = round((time.time() - start) * 1000, 2)
        return {"status": "healthy", "latency_ms": latency_ms}
    except Exception as exc:
        logger.error("Database health check failed | error=%s", exc,
                     extra={"component": "db.engine"})
        return {"status": "unhealthy", "error": str(exc)}


def dispose_engine():
    global _engine, _SessionFactory
    if _engine:
        _engine.dispose()
        _engine = None
        _SessionFactory = None
        logger.info("Database engine disposed", extra={"component": "db.engine"})