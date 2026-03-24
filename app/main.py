# =========================================================
# MARKET SENTINEL APPLICATION ENTRYPOINT v3.4
# FIX: ModelLoader attribute names aligned to v2.8
#      (loader.xgb → loader.model, loader.xgb_version →
#       loader.version, loader.warmup() → loader.load())
# FIX: app.state.model_loader, app.state.cache,
#      app.state.startup_time now set — health/portfolio
#      routes all depend on these.
# FIX: Added POST /admin/sync endpoint (item 44)
# =========================================================

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

API_KEY = os.getenv("API_KEY")

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MIN", "180"))
WINDOW_SECONDS = 180
RATE_LIMIT_KEY_PREFIX = "ms:ratelimit:"
RATE_LIMIT_TTL = WINDOW_SECONDS + 10

SKIP_DATA_SYNC = os.getenv("SKIP_DATA_SYNC", "0") == "1"
SNAPSHOT_PRECOMPUTE_INTERVAL = int(os.getenv("SNAPSHOT_PRECOMPUTE_INTERVAL", "120"))
DATA_STALENESS_HOURS = int(os.getenv("DATA_STALENESS_HOURS", "24"))

PUBLIC_PATHS = {
    "/", "/docs", "/openapi.json", "/redoc", "/metrics",
    "/health/live", "/health/ready", "/health/db", "/health/model",
    "/universe", "/auth/owner-login", "/auth/demo-login",
    "/auth/me", "/auth/logout", "/favicon.ico",
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
    return JSONResponse(status_code=500, content=api_error("internal_server_error"))


# =====================================================
# DATA STALENESS CHECK
# =====================================================

async def _is_data_stale() -> bool:
    try:
        from core.db.repository import OHLCVRepository
        stored = await asyncio.to_thread(OHLCVRepository.get_stored_tickers)
        if not stored:
            logger.info("DB is empty — sync required")
            return True
        latest = await asyncio.to_thread(OHLCVRepository.get_latest_date, stored[0])
        if not latest:
            return True
        age_hours = (dt.utcnow().date() - latest).days * 24
        stale = age_hours >= DATA_STALENESS_HOURS
        logger.info("Data staleness check | latest=%s age_hours=%d stale=%s", latest, age_hours, stale)
        return stale
    except Exception as e:
        logger.warning("Staleness check failed — assuming stale | error=%s", e)
        return True


# =====================================================
# BACKGROUND TASKS
# =====================================================

async def _daily_sync_loop():
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
                    report.get("synced", 0), report.get("skipped", 0), report.get("errors", 0),
                )
            except Exception as e:
                logger.warning("Daily sync failed | error=%s", e)


async def _background_snapshot_loop():
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
                    "Background snapshot cached | signals=%d",
                    len(result.get("signals", [])),
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
        logger.info("=================================")
        logger.info(" MarketSentinel Boot v3.4")
        logger.info(" Boot ID: %s", BOOT_ID)
        logger.info("=================================")

        # ── Step 1: Record startup time ───────────────
        # FIX: Set before anything else — health.py reads this
        app.state.startup_time = time.time()

        # ── Step 2: Initialize PostgreSQL ─────────────
        try:
            init_db()
            db_health = check_db_health()
            readiness.db_connected = db_health["status"] == "healthy"
            logger.info(
                "Database ready | status=%s latency=%.1fms",
                db_health["status"], db_health.get("latency_ms", 0),
            )
        except Exception:
            readiness.db_connected = False
            logger.exception("Database initialization failed")

        # ── Step 3: Sync market data (non-blocking) ───
        if readiness.db_connected and not SKIP_DATA_SYNC:
            try:
                stale = await _is_data_stale()
                if stale:
                    logger.info("Data is stale — starting background sync")
                    async def _run_sync():
                        try:
                            sync_service = DataSyncService()
                            report = await asyncio.to_thread(sync_service.sync_universe)
                            readiness.data_synced = True
                            readiness.sync_report = report
                            logger.info(
                                "Background sync complete | synced=%d skipped=%d errors=%d",
                                report.get("synced", 0), report.get("skipped", 0), report.get("errors", 0),
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

        # ── Step 4: Load model ────────────────────────
        # FIX: Use loader.load() not loader.xgb / loader.warmup()
        # FIX: Use loader.version not loader.xgb_version
        # FIX: Use loader.metadata dict for dataset_hash, training_code_hash
        loader = ModelLoader()
        load_success = loader.load()

        readiness.models_loaded = load_success and loader.is_loaded()
        readiness.schema_signature = loader.schema_signature
        readiness.model_version = loader.version                        # FIX
        readiness.artifact_hash = loader.artifact_hash

        meta = loader.metadata or {}
        readiness.dataset_hash = meta.get("dataset_hash")              # FIX
        readiness.training_code_hash = meta.get("training_code_hash")  # FIX

        if readiness.schema_signature and readiness.schema_signature != get_schema_signature():
            logger.warning("Runtime schema mismatch detected.")

        # FIX: Set app.state.model_loader — health.py + model_info.py depend on this
        app.state.model_loader = loader

        # ── Step 5: Redis ──────────────────────────────
        try:
            cache = RedisCache()
            readiness.redis_connected = cache.enabled
            # FIX: Set app.state.cache — health.py + portfolio.py depend on this
            app.state.cache = cache
        except Exception:
            readiness.redis_connected = False
            app.state.cache = RedisCache()
            logger.warning("Redis unavailable — degraded mode.")

        # ── Step 6: Drift baseline ────────────────────
        try:
            detector = DriftDetector()
            detector._load_verified_baseline()
            readiness.drift_baseline_loaded = True
        except Exception:
            readiness.drift_baseline_loaded = False
            logger.warning("Drift baseline not loaded.")

        # ── Step 7: LLM config ────────────────────────
        readiness.llm_enabled = get_bool("LLM_ENABLED", False)
        readiness.llm_model = get_env("OPENAI_MODEL", None)
        readiness.llm_rate_limit = get_int("LLM_RATE_LIMIT_PER_MIN", 30)
        readiness.llm_cache_enabled = get_bool("LLM_CACHE_ENABLED", True)

        # ── Step 8: Memory + fingerprint ─────────────
        process = psutil.Process(os.getpid())
        readiness.boot_memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        readiness.config_fingerprint = hashlib.sha256(
            str(sorted(os.environ.items())).encode()
        ).hexdigest()[:16]

        gc.collect()

        boot_time = round(time.time() - boot_start, 2)
        logger.info(
            "Startup complete | time=%ss | db=%s | redis=%s | model=%s | drift=%s",
            boot_time, readiness.db_connected, readiness.redis_connected,
            readiness.models_loaded, readiness.drift_baseline_loaded,
        )

        # ── Step 9: Start background tasks ───────────
        asyncio.create_task(_daily_sync_loop())
        logger.info("Daily sync scheduler started (runs weekdays at 18:30)")

        if readiness.models_loaded and readiness.db_connected:
            asyncio.create_task(_background_snapshot_loop())
            logger.info(
                "Background snapshot pre-computation started (interval=%ds)",
                SNAPSHOT_PRECOMPUTE_INTERVAL,
            )

        yield

        # ── Shutdown ──────────────────────────────────
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


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):

    path = request.url.path

    if path not in PUBLIC_PATHS and API_KEY:
        client_key = request.headers.get("X-API-KEY")
        has_jwt = (
            request.cookies.get("ms_token") or
            request.headers.get("Authorization", "").startswith("Bearer ")
        )
        if not has_jwt and client_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(",")[0].strip() if forwarded else request.client.host

    # Redis-backed rate limiting
    try:
        cache = app.state.cache
        if cache.enabled and cache._redis:
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
        pass  # Redis down — don't block requests

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
# ADMIN: MANUAL SYNC TRIGGER  (item 44)
# POST /admin/sync — triggers delta data sync on demand.
# Requires owner JWT. Useful after market close to ensure
# latest prices without waiting for scheduled 18:30 run.
# =====================================================

@app.post("/admin/sync")
async def admin_sync(request: Request):
    """
    Trigger a manual delta data sync.
    Only callable by authenticated owners.
    Does not block — sync runs in background thread.
    """
    if getattr(request.state, "role", None) != "owner":
        raise HTTPException(status_code=403, detail="Owner access required")

    async def _run():
        try:
            svc = DataSyncService()
            report = await asyncio.to_thread(svc.sync_universe)
            readiness.data_synced = True
            readiness.sync_report = report
            logger.info(
                "Manual sync complete | synced=%d skipped=%d errors=%d",
                report.get("synced", 0), report.get("skipped", 0), report.get("errors", 0),
            )
        except Exception:
            logger.exception("Manual sync failed")

    asyncio.create_task(_run())

    return api_success({
        "message": "Data sync started in background",
        "status": "running",
        "timestamp": dt.now(timezone.utc).isoformat(),
    })


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
        "models_loaded": readiness.models_loaded,
    })


# =====================================================
# METRICS
# =====================================================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)