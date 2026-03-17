# =====================================================
# MARKET SENTINEL APPLICATION ENTRYPOINT v3.1
# Hybrid Multi-Agent | DB-Backed | CV-Optimized
# =====================================================

import asyncio
import time
import gc
import hashlib
import os
import psutil
import uuid
import datetime
from contextlib import asynccontextmanager
from datetime import timezone
from datetime import datetime as dt
from collections import defaultdict, deque

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from core.config.env_loader import init_env, get_int, get_env, get_bool
from core.schema.feature_schema import get_schema_signature
from core.logging.logger import get_logger

from app.api.routes import (
    drift,
    model_info,
    portfolio,
    universe,
    health,
    predict,
    performance,
    equity,
    agent,
)

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache
from core.monitoring.drift_detector import DriftDetector
from core.db.engine import init_db, check_db_health, dispose_engine
from core.data.data_sync import DataSyncService


# =====================================================
# ENV INITIALIZATION
# =====================================================

init_env()

logger = get_logger("marketsentinel")


# =====================================================
# SYSTEM METADATA
# =====================================================

BOOT_ID = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]

STARTUP_TIMEOUT_SEC = get_int("STARTUP_TIMEOUT_SEC", 180)

APP_VERSION = get_env("APP_VERSION", "5.0.0")

CORS_ORIGINS = get_env("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",")]

API_KEY = os.getenv("API_KEY")

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MIN", "180"))
WINDOW_SECONDS = 180

SKIP_DATA_SYNC = os.getenv("SKIP_DATA_SYNC", "0") == "1"

# Background snapshot pre-computation interval (seconds)
SNAPSHOT_PRECOMPUTE_INTERVAL = int(os.getenv("SNAPSHOT_PRECOMPUTE_INTERVAL", "120"))

PUBLIC_PATHS = {
    "/",
    "/docs",
    "/openapi.json",
    "/metrics",
    "/health/live",
    "/health/ready",
}

request_store = defaultdict(lambda: deque())
MAX_TRACKED_IPS = 5000


# =====================================================
# READINESS STATE
# =====================================================

class ReadinessState:

    def __init__(self):

        self.models_loaded = False
        self.redis_connected = False
        self.db_connected = False
        self.data_synced = False
        self.drift_baseline_loaded = False

        self.schema_signature = None
        self.model_version = None
        self.artifact_hash = None
        self.dataset_hash = None
        self.training_code_hash = None

        self.llm_enabled = False
        self.llm_model = None
        self.llm_rate_limit = None
        self.llm_cache_enabled = None

        self.boot_id = BOOT_ID
        self.start_time = int(time.time())

        self.boot_memory_mb = None
        self.config_fingerprint = None

        self.sync_report = None

    @property
    def ready(self):
        return self.models_loaded and self.db_connected


readiness = ReadinessState()


# =====================================================
# RESPONSE HELPERS
# =====================================================

def api_success(data):

    return {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": dt.now(timezone.utc).isoformat(),
    }


def api_error(message):

    return {
        "success": False,
        "data": None,
        "error": message,
        "timestamp": dt.now(timezone.utc).isoformat(),
    }


# =====================================================
# GLOBAL EXCEPTION HANDLER
# =====================================================

async def global_exception_handler(request: Request, exc: Exception):

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=api_error(exc.detail),
        )

    logger.exception("Unhandled API error")

    return JSONResponse(
        status_code=500,
        content=api_error("internal_server_error"),
    )


# =====================================================
# BACKGROUND TASKS
# =====================================================

async def _daily_sync_loop():
    """
    Runs delta data sync every weekday at 6:30pm.
    Only fetches new rows since last stored date — fast after first run.
    """
    while True:
        now = datetime.datetime.now()
        target = now.replace(hour=18, minute=30, second=0, microsecond=0)
        if now >= target:
            target += datetime.timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())

        if datetime.datetime.now().weekday() < 5:
            try:
                logger.info("Running scheduled daily data sync")
                svc = DataSyncService()
                report = await asyncio.to_thread(svc.sync_universe)
                logger.info(
                    "Daily sync complete | synced=%d skipped=%d errors=%d",
                    report.get("synced", 0),
                    report.get("skipped", 0),
                    report.get("errors", 0),
                )
            except Exception as e:
                logger.warning("Daily sync failed | error=%s", e)


async def _background_snapshot_loop():
    """
    Pre-computes the full snapshot every SNAPSHOT_PRECOMPUTE_INTERVAL seconds
    and stores the result in Redis. This means POST /snapshot returns instantly
    from cache instead of waiting 10-30s for inference.
    """
    # Wait for initial startup to complete before starting
    await asyncio.sleep(30)

    while True:
        try:
            from app.api.routes.predict import get_pipeline, load_default_universe

            pipeline = get_pipeline()
            tickers = load_default_universe()

            result = await asyncio.to_thread(pipeline.run_snapshot, tickers)

            cache = RedisCache()
            if cache.enabled:
                key = cache.build_key({"type": "background_snapshot"})
                cache.set(key, result)
                logger.info(
                    "Background snapshot cached | signals=%d | model=%s",
                    len(result.get("signals", [])),
                    result.get("model_version", "unknown"),
                )

        except Exception as e:
            logger.warning("Background snapshot failed | error=%s", e)

        await asyncio.sleep(SNAPSHOT_PRECOMPUTE_INTERVAL)


# =====================================================
# LIFESPAN MANAGEMENT
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    boot_start = time.time()

    try:

        logger.info("===================================")
        logger.info(" MarketSentinel Boot Sequence v3.1 ")
        logger.info(" Boot ID: %s", BOOT_ID)
        logger.info("===================================")

        # ── Step 1: Initialize PostgreSQL ────────────────

        try:
            init_db()
            db_health = check_db_health()
            readiness.db_connected = db_health["status"] == "healthy"

            logger.info(
                "Database ready | status=%s latency=%.1fms",
                db_health["status"],
                db_health.get("latency_ms", 0),
            )

        except Exception:
            readiness.db_connected = False
            logger.exception("Database initialization failed")

        # ── Step 2: Sync market data (Yahoo → PostgreSQL) ──

        if readiness.db_connected and not SKIP_DATA_SYNC:

            try:
                sync_service = DataSyncService()
                readiness.sync_report = sync_service.sync_universe()
                readiness.data_synced = True

                logger.info(
                    "Data sync complete | synced=%d skipped=%d errors=%d",
                    readiness.sync_report.get("synced", 0),
                    readiness.sync_report.get("skipped", 0),
                    readiness.sync_report.get("errors", 0),
                )

            except Exception:
                readiness.data_synced = False
                logger.exception("Data sync failed — DB may have stale data")

        elif SKIP_DATA_SYNC:
            logger.info("Data sync skipped (SKIP_DATA_SYNC=1)")

        # ── Step 3: Load model ───────────────────────────

        loader = ModelLoader()
        _ = loader.xgb
        loader.warmup()

        readiness.models_loaded = True
        readiness.schema_signature = loader.schema_signature
        readiness.model_version = loader.xgb_version
        readiness.artifact_hash = loader.artifact_hash
        readiness.dataset_hash = loader.dataset_hash
        readiness.training_code_hash = loader.training_code_hash

        if readiness.schema_signature != get_schema_signature():
            logger.warning("Runtime schema mismatch detected.")

        # ── Step 4: Redis ────────────────────────────────

        try:
            cache = RedisCache()
            readiness.redis_connected = cache.enabled
        except Exception:
            readiness.redis_connected = False
            logger.warning("Redis unavailable — degraded mode.")

        # ── Step 5: Drift baseline ───────────────────────

        try:
            detector = DriftDetector()
            detector._load_verified_baseline()
            readiness.drift_baseline_loaded = True
        except Exception:
            readiness.drift_baseline_loaded = False
            logger.warning("Drift baseline not loaded.")

        # ── Step 6: LLM config ───────────────────────────

        readiness.llm_enabled = get_bool("LLM_ENABLED", False)
        readiness.llm_model = get_env("OPENAI_MODEL", None)
        readiness.llm_rate_limit = get_int("LLM_RATE_LIMIT_PER_MIN", 30)
        readiness.llm_cache_enabled = get_bool("LLM_CACHE_ENABLED", True)

        # ── Step 7: Memory + fingerprint ─────────────────

        process = psutil.Process(os.getpid())
        readiness.boot_memory_mb = round(
            process.memory_info().rss / (1024 * 1024), 2
        )

        readiness.config_fingerprint = hashlib.sha256(
            str(sorted(os.environ.items())).encode()
        ).hexdigest()[:16]

        gc.collect()

        boot_time = round(time.time() - boot_start, 2)

        if boot_time > STARTUP_TIMEOUT_SEC:
            logger.warning("Startup exceeded expected time.")

        logger.info(
            "Startup complete | time=%ss | db=%s | redis=%s | drift=%s | synced=%s",
            boot_time,
            readiness.db_connected,
            readiness.redis_connected,
            readiness.drift_baseline_loaded,
            readiness.data_synced,
        )

        # ── Step 8: Start background tasks ───────────────

        asyncio.create_task(_daily_sync_loop())
        logger.info("Daily sync scheduler started (runs weekdays at 18:30)")

        if readiness.models_loaded and readiness.db_connected:
            asyncio.create_task(_background_snapshot_loop())
            logger.info(
                "Background snapshot pre-computation started (interval=%ds)",
                SNAPSHOT_PRECOMPUTE_INTERVAL,
            )

        yield

        # ── Shutdown ─────────────────────────────────────

        logger.info("Shutting down MarketSentinel")
        dispose_engine()
        gc.collect()

    except Exception:
        logger.exception("CRITICAL STARTUP FAILURE")
        raise


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="MarketSentinel Portfolio API",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_exception_handler(Exception, global_exception_handler)


# =====================================================
# MIDDLEWARE (order matters — outermost first)
# =====================================================

# GZip — compresses responses > 1KB (cuts JSON payload 60-80%)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# =====================================================
# REQUEST MIDDLEWARE
# =====================================================

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):

    path = request.url.path

    if path not in PUBLIC_PATHS and API_KEY:

        client_key = request.headers.get("X-API-KEY")

        if client_key != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
            )

    forwarded = request.headers.get("X-Forwarded-For")

    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host

    now = time.time()

    queue = request_store[client_ip]

    while queue and queue[0] < now - WINDOW_SECONDS:
        queue.popleft()

    if len(queue) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )

    queue.append(now)

    if len(request_store) > MAX_TRACKED_IPS:
        request_store.clear()

    request_id = str(uuid.uuid4())[:12]
    start = time.time()

    try:
        response = await call_next(request)
        latency = time.time() - start

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(latency, 4))

        return response

    except Exception:
        logger.exception("Request failure | id=%s", request_id)
        raise


# =====================================================
# ROUTERS
# =====================================================

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(universe.router)
app.include_router(model_info.router, prefix="/model")   # FIX: /model/info, /model/model-info → /model/info
app.include_router(drift.router)
app.include_router(portfolio.router)
app.include_router(performance.router)
app.include_router(equity.router)
app.include_router(agent.router)


# =====================================================
# ROOT-LEVEL SNAPSHOT ALIAS
# FIX: predict.router uses prefix=/predict so POST /snapshot
# was returning 404. This aliases it at the root level.
# =====================================================

@app.post("/snapshot")
async def snapshot():
    """
    Full inference run — all tickers, multi-agent pipeline, portfolio construction.
    Alias for POST /predict/live-snapshot at the root level.
    """
    from app.api.routes.predict import live_snapshot
    return await live_snapshot()


# =====================================================
# ROOT
# =====================================================

@app.get("/")
async def root():

    uptime = int(time.time()) - readiness.start_time

    return api_success({
        "service": "MarketSentinel Hybrid Portfolio Engine",
        "architecture": "multi-agent-ml-governed",
        "status": "running",
        "boot_id": readiness.boot_id,
        "model_version": readiness.model_version,
        "artifact_hash": readiness.artifact_hash,
        "db": readiness.db_connected,
        "redis": readiness.redis_connected,
        "data_synced": readiness.data_synced,
        "drift_baseline": readiness.drift_baseline_loaded,
        "uptime_seconds": uptime,
        "version": APP_VERSION,
        "docs": "/docs",
        "metrics": "/metrics",
    })


# =====================================================
# METRICS
# =====================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )