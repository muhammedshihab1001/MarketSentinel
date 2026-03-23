# =====================================================
# MARKET SENTINEL APPLICATION ENTRYPOINT v3.3
# Hybrid Multi-Agent | DB-Backed | Auth-Enabled
#
# FIXES in v3.3:
#   FIX 1: Startup sync wrapped in asyncio.to_thread()
#           + staleness check — only syncs if DB data
#           is older than 24h. Prevents blocking the
#           event loop for 4-8 minutes on every restart.
#   FIX 2: Rate limiter moved to Redis INCR+EXPIRE.
#           In-memory dict reset on every restart and
#           was bypassable by crashing the API.
#   FIX 3: /admin/retrain removed — docker-compose is
#           not installed inside the inference image so
#           the endpoint always crashed with FileNotFoundError.
#           Use: docker-compose run --rm training instead.
#   FIX 4: request_store.clear() replaced with LRU eviction
#           — clearing all IPs let attackers bypass limits.
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

from app.api.routes import auth as auth_router
from app.core.auth.middleware import AuthMiddleware

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

CORS_ORIGINS = get_env("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",")]

# Legacy API key — optional, for external tool access only
API_KEY = os.getenv("API_KEY")

# FIX: Rate limit config (Redis-backed)
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MIN", "180"))
WINDOW_SECONDS = 180
RATE_LIMIT_KEY_PREFIX = "ms:ratelimit:"
RATE_LIMIT_TTL = WINDOW_SECONDS + 10

SKIP_DATA_SYNC = os.getenv("SKIP_DATA_SYNC", "0") == "1"
SNAPSHOT_PRECOMPUTE_INTERVAL = int(os.getenv("SNAPSHOT_PRECOMPUTE_INTERVAL", "120"))

# Staleness threshold — only sync if DB data is older than this
DATA_STALENESS_HOURS = int(os.getenv("DATA_STALENESS_HOURS", "24"))

PUBLIC_PATHS = {
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
    "/health/live",
    "/health/ready",
    "/health/db",
    "/health/model",
    "/universe",
    "/auth/owner-login",
    "/auth/demo-login",
    "/auth/me",
    "/auth/logout",
    "/favicon.ico",
}


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
# DATA STALENESS CHECK
# FIX: Only sync if DB data is older than DATA_STALENESS_HOURS.
# This prevents the 4-8 minute blocking sync on every restart.
# =====================================================

async def _is_data_stale() -> bool:
    """
    Returns True if the DB has no data or data is older than
    DATA_STALENESS_HOURS (default 24h).
    Fast check — only queries MAX(date) from ohlcv_daily.
    """
    try:
        from core.db.repository import OHLCVRepository
        # Get the latest date for any ticker
        stored = await asyncio.to_thread(OHLCVRepository.get_stored_tickers)
        if not stored:
            logger.info("DB is empty — sync required")
            return True

        # Check the most recent ticker's latest date
        latest = await asyncio.to_thread(OHLCVRepository.get_latest_date, stored[0])
        if not latest:
            return True

        age_hours = (dt.utcnow().date() - latest).days * 24
        stale = age_hours >= DATA_STALENESS_HOURS

        logger.info(
            "Data staleness check | latest=%s age_hours=%d stale=%s",
            latest, age_hours, stale
        )
        return stale

    except Exception as e:
        logger.warning("Staleness check failed — assuming stale | error=%s", e)
        return True


# =====================================================
# BACKGROUND TASKS
# =====================================================

async def _daily_sync_loop():
    """Runs delta data sync every weekday at 6:30pm."""
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
    """Pre-computes full snapshot every SNAPSHOT_PRECOMPUTE_INTERVAL seconds."""
    await asyncio.sleep(30)

    while True:
        try:
            from app.api.routes.predict import get_pipeline, load_default_universe

            pipeline = get_pipeline()
            tickers = load_default_universe()
            result = await asyncio.to_thread(pipeline.run_snapshot, tickers)

            cache = RedisCache()
            if cache.enabled:
                key = "ms:background_snapshot:latest"
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
        logger.info(" MarketSentinel Boot Sequence v3.3 ")
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

        # ── Step 2: Sync market data (non-blocking) ───────
        # FIX: Wrapped in asyncio.to_thread + staleness check.
        # Only runs if data is > 24h old. Does NOT block the event loop.
        if readiness.db_connected and not SKIP_DATA_SYNC:
            try:
                stale = await _is_data_stale()
                if stale:
                    logger.info("Data is stale — starting background sync")
                    # Run sync in thread so it doesn't block startup
                    # API starts serving immediately, sync runs in background
                    async def _run_sync():
                        try:
                            sync_service = DataSyncService()
                            report = await asyncio.to_thread(sync_service.sync_universe)
                            readiness.data_synced = True
                            readiness.sync_report = report
                            logger.info(
                                "Background sync complete | synced=%d skipped=%d errors=%d",
                                report.get("synced", 0),
                                report.get("skipped", 0),
                                report.get("errors", 0),
                            )
                        except Exception:
                            logger.exception("Background sync failed")

                    asyncio.create_task(_run_sync())
                    logger.info("Background sync started — API serving immediately")
                else:
                    logger.info("Data is fresh — skipping sync")
                    readiness.data_synced = True

            except Exception:
                logger.exception("Staleness check failed — skipping sync")

        elif SKIP_DATA_SYNC:
            logger.info("Data sync skipped (SKIP_DATA_SYNC=1)")
            readiness.data_synced = True

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
        logger.info(
            "Startup complete | time=%ss | db=%s | redis=%s | drift=%s",
            boot_time,
            readiness.db_connected,
            readiness.redis_connected,
            readiness.drift_baseline_loaded,
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
# MIDDLEWARE
# =====================================================

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.add_middleware(AuthMiddleware)


# =====================================================
# REQUEST MIDDLEWARE
# FIX: Rate limiter now uses Redis INCR+EXPIRE per IP.
# In-memory dict was reset on every restart and could be
# bypassed by triggering a crash. Redis state survives restarts.
# FIX: Removed request_store.clear() which nuked all IPs
# when MAX_TRACKED_IPS was exceeded — replaced with LRU eviction.
# =====================================================

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):

    path = request.url.path

    # Legacy API key check — only if API_KEY is set
    if path not in PUBLIC_PATHS and API_KEY:
        client_key = request.headers.get("X-API-KEY")
        has_jwt = (
            request.cookies.get("ms_token") or
            request.headers.get("Authorization", "").startswith("Bearer ")
        )
        if not has_jwt and client_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Get client IP
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(",")[0].strip() if forwarded else request.client.host

    # FIX: Redis-backed rate limiting — survives restarts
    cache = RedisCache()
    if cache.enabled and cache._redis:
        try:
            redis_client = cache._redis
            key = f"{RATE_LIMIT_KEY_PREFIX}{client_ip}"
            count = redis_client.incr(key)
            if count == 1:
                redis_client.expire(key, RATE_LIMIT_TTL)
            if count > RATE_LIMIT:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except HTTPException:
            raise
        except Exception:
            # Redis down — fall through, don't block requests
            pass
    else:
        # Fallback: in-memory rate limiting when Redis unavailable
        # Less secure but keeps API functional
        now = time.time()
        if not hasattr(request_context_middleware, "_store"):
            request_context_middleware._store = defaultdict(deque)
        store = request_context_middleware._store
        queue = store[client_ip]
        while queue and queue[0] < now - WINDOW_SECONDS:
            queue.popleft()
        if len(queue) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        queue.append(now)
        # FIX: LRU eviction — remove oldest entry, not clear all
        if len(store) > 5000:
            oldest_ip = next(iter(store))
            del store[oldest_ip]

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

app.include_router(auth_router.router)
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(universe.router)
app.include_router(model_info.router, prefix="/model")
app.include_router(drift.router)
app.include_router(portfolio.router)
app.include_router(performance.router)
app.include_router(equity.router)
app.include_router(agent.router)


# =====================================================
# ROOT-LEVEL SNAPSHOT ALIAS
# =====================================================

@app.post("/snapshot")
async def snapshot():
    """Full inference run — alias for /predict/live-snapshot."""
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
        # FIX: Expose these so the frontend Health page can read them
        "models_loaded": readiness.models_loaded,
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