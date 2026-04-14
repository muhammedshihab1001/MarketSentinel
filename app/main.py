# =========================================================
# MARKET SENTINEL APPLICATION ENTRYPOINT v3.7
#
# Changes from v3.6:
# FIX #19: Per-endpoint IP rate limits added to middleware.
#   Previously only a global rate limit existed (180 req/3min).
#   An attacker could hammer /health/live millions of times
#   with no per-endpoint protection — generating cloud costs.
#
#   Per-endpoint limits now enforced via Redis per IP:
#     /auth/owner-login          5  req / 60s  (brute force guard)
#     /auth/demo-login          10  req / 60s
#     /predict/live-snapshot    10  req / 60s  (expensive inference)
#     /snapshot                 10  req / 60s  (alias)
#     /agent/explain            20  req / 60s
#     /agent/political-risk     20  req / 60s
#     /performance              20  req / 60s
#     /health/live              60  req / 60s  (Docker probe)
#     /health/ready             60  req / 60s
#     ALL OTHER paths           60  req / 60s  (global fallback)
#
#   On limit exceeded: 429 with Retry-After header.
#   Redis unavailable: fails open (no block during downtime).
#
# FIX: _check_rate_limit no longer accesses cache._redis directly.
#      Uses cache.get_redis_client() public method instead.
#
# All other fixes from v3.6 retained.
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

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from core.config.env_loader import init_env, get_env, get_bool
from core.schema.feature_schema import get_schema_signature
from core.logging.logger import get_logger

from app.api.routes import (
    drift, model_info, portfolio, universe,
    health, predict, performance, equity, agent,
)
from app.api.routes import auth as auth_router
from app.core.auth.middleware import AuthMiddleware
from app.inference.model_loader import get_model_loader
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
APP_VERSION = get_env("APP_VERSION", "5.0.0")

CORS_ORIGINS = get_env("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",")]

API_KEY = os.getenv("API_KEY")

SKIP_DATA_SYNC = os.getenv("SKIP_DATA_SYNC", "0") == "1"
SNAPSHOT_PRECOMPUTE_INTERVAL = int(os.getenv("SNAPSHOT_PRECOMPUTE_INTERVAL", "300"))
DATA_STALENESS_HOURS = int(os.getenv("DATA_STALENESS_HOURS", "24"))

PUBLIC_PATHS = {
    "/", "/docs", "/openapi.json", "/redoc", "/metrics",
    "/health/live", "/health/ready", "/health/db", "/health/model",
    "/universe", "/auth/owner-login", "/auth/demo-login",
    "/auth/me", "/auth/logout", "/favicon.ico",
    "/model/info",
    "/agent/agents",
}

# =====================================================
# PER-ENDPOINT RATE LIMITS
#
# Format: { path_prefix: (max_requests, window_seconds) }
# First prefix match wins. Unlisted paths use default.
# =====================================================

PER_PATH_RATE_LIMITS = {
    "/auth/owner-login":        (5,  60),
    "/auth/demo-login":         (10, 60),
    "/predict/live-snapshot":   (10, 60),
    "/snapshot":                (10, 60),
    "/agent/explain":           (20, 60),
    "/agent/political-risk":    (20, 60),
    "/performance":             (20, 60),
    "/health/live":             (60, 60),
    "/health/ready":            (60, 60),
}

RATE_LIMIT_DEFAULT = (60, 60)
RATE_LIMIT_KEY_PREFIX = "ms:ratelimit:"

_snapshot_lock = asyncio.Lock()


# =====================================================
# RATE LIMIT HELPERS
# =====================================================

def _get_path_limit(path: str) -> tuple:
    """Return (max_requests, window_seconds) for the given path."""
    for prefix, limits in PER_PATH_RATE_LIMITS.items():
        if (
            path == prefix
            or path.startswith(prefix + "/")
            or path.startswith(prefix + "?")
        ):
            return limits
    return RATE_LIMIT_DEFAULT


def _check_rate_limit(redis_client, client_ip: str, path: str) -> tuple:
    """
    Check and increment rate limit counter for (ip, path).

    Returns:
        (allowed: bool, count: int, limit: int, retry_after: int)

    Fails open if Redis is unavailable.

    FIX: No longer accesses cache._redis directly.
         redis_client is passed in explicitly.
    """
    max_requests, window_seconds = _get_path_limit(path)
    safe_path = path.replace("/", "_").replace("?", "_").strip("_")
    key = f"{RATE_LIMIT_KEY_PREFIX}{client_ip}:{safe_path}"

    try:
        count = int(redis_client.incr(key))
        if count == 1:
            redis_client.expire(key, window_seconds)
        if count > max_requests:
            ttl = int(redis_client.ttl(key))
            return False, count, max_requests, max(0, ttl)
        return True, count, max_requests, 0
    except Exception:
        return True, 0, max_requests, 0


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
        self.llm_enabled = False
        self.boot_id = BOOT_ID
        self.start_time = int(time.time())
        self.boot_memory_mb = None
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
            logger.info("DB is empty — full sync required")
            return True
        latest = await asyncio.to_thread(OHLCVRepository.get_latest_date, stored[0])
        if not latest:
            return True
        age_hours = (dt.utcnow().date() - latest).days * 24
        stale = age_hours >= DATA_STALENESS_HOURS
        logger.info(
            "Data staleness check | latest=%s age_hours=%d stale=%s",
            latest, age_hours, stale,
        )
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
                    report.get("synced", 0),
                    report.get("skipped", 0),
                    report.get("errors", 0),
                )
            except Exception as e:
                logger.warning("Daily sync failed | error=%s", e)


async def _background_snapshot_loop():
    """
    Pre-computes snapshot every SNAPSHOT_PRECOMPUTE_INTERVAL seconds.
    asyncio.Lock prevents concurrent runs.
    """
    await asyncio.sleep(30)

    while True:
        if _snapshot_lock.locked():
            logger.warning(
                "Background snapshot skipped — previous run still in progress | "
                "interval=%ds",
                SNAPSHOT_PRECOMPUTE_INTERVAL,
            )
            await asyncio.sleep(SNAPSHOT_PRECOMPUTE_INTERVAL)
            continue

        async with _snapshot_lock:
            snapshot_start = time.time()
            try:
                pipeline = predict.get_pipeline()
                result = await asyncio.to_thread(pipeline.run_snapshot)
                cache = app.state.cache
                if cache.set_background_snapshot(
                    result, ttl=SNAPSHOT_PRECOMPUTE_INTERVAL + 60
                ):
                    logger.info(
                        "Background snapshot cached | signals=%d | model=%s | took=%ss",
                        len(result.get("snapshot", {}).get("signals", [])),
                        result.get("meta", {}).get("model_version", "unknown"),
                        round(time.time() - snapshot_start, 1),
                    )
                else:
                    logger.warning(
                        "Background snapshot cache write failed (memory fallback)"
                    )
            except Exception as e:
                logger.warning(
                    "Background snapshot failed | error=%s | took=%ss",
                    e, round(time.time() - snapshot_start, 1),
                )

        await asyncio.sleep(SNAPSHOT_PRECOMPUTE_INTERVAL)


# =====================================================
# LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    boot_start = time.time()

    try:
        logger.info("=================================")
        logger.info(" MarketSentinel Boot v3.7")
        logger.info(" Boot ID: %s", BOOT_ID)
        logger.info("=================================")

        app.state.startup_time = time.time()

        # ── PostgreSQL ────────────────────────────────
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

        # ── Data sync ─────────────────────────────────
        if readiness.db_connected and not SKIP_DATA_SYNC:
            try:
                stale = await _is_data_stale()
                if stale:
                    async def _run_sync():
                        try:
                            svc = DataSyncService()
                            report = await asyncio.to_thread(svc.sync_universe)
                            readiness.data_synced = True
                            readiness.sync_report = report
                            logger.info(
                                "Background sync complete | "
                                "synced=%d skipped=%d errors=%d",
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

        # ── Model ─────────────────────────────────────
        loader = get_model_loader()
        load_success = loader.load()
        readiness.models_loaded = load_success and loader.is_loaded()
        readiness.schema_signature = loader.schema_signature
        readiness.model_version = loader.version
        readiness.artifact_hash = loader.artifact_hash

        if (
            readiness.schema_signature
            and readiness.schema_signature != get_schema_signature()
        ):
            logger.warning(
                "Runtime schema mismatch — retrain with: "
                "docker-compose run --rm training"
            )

        app.state.model_loader = loader

        # ── Redis ─────────────────────────────────────
        try:
            cache = RedisCache()
            readiness.redis_connected = cache.ping()
            if not readiness.redis_connected:
                logger.warning("Redis unavailable — degraded mode (memory fallback)")
            app.state.cache = cache
        except Exception:
            logger.warning("Redis init failed — degraded mode")
            cache = RedisCache()
            readiness.redis_connected = False
            app.state.cache = cache

        # ── InferencePipeline ─────────────────────────
        if readiness.models_loaded:
            try:
                predict.init_pipeline(loader, cache)
                logger.info(
                    "InferencePipeline initialized | model=%s", loader.version
                )
            except Exception:
                logger.exception("InferencePipeline init failed")

        # ── Drift baseline ────────────────────────────
        try:
            detector = DriftDetector()
            detector._load_verified_baseline()
            readiness.drift_baseline_loaded = True
        except Exception:
            readiness.drift_baseline_loaded = False
            logger.warning("Drift baseline not loaded — retrain to create one")

        # ── LLM + memory ──────────────────────────────
        readiness.llm_enabled = get_bool("LLM_ENABLED", False)
        process = psutil.Process(os.getpid())
        readiness.boot_memory_mb = round(
            process.memory_info().rss / (1024 * 1024), 2
        )
        gc.collect()

        boot_time = round(time.time() - boot_start, 2)
        logger.info(
            "Startup complete | time=%ss | db=%s | redis=%s | model=%s | drift=%s",
            boot_time, readiness.db_connected, readiness.redis_connected,
            readiness.models_loaded, readiness.drift_baseline_loaded,
        )

        logger.info(
            "Rate limits active | endpoints=%d | default=%d req/%ds | fails-open",
            len(PER_PATH_RATE_LIMITS),
            RATE_LIMIT_DEFAULT[0],
            RATE_LIMIT_DEFAULT[1],
        )

        if not API_KEY:
            logger.warning(
                "API_KEY is not set — programmatic access blocked. "
                "Only JWT-authenticated requests (browser login) are accepted. "
                "Generate a key: python scripts/generate_api_key.py"
            )
        else:
            logger.info("API key configured | programmatic access enabled")

        # ── Background tasks ──────────────────────────
        asyncio.create_task(_daily_sync_loop())
        logger.info("Daily sync scheduler started (runs weekdays at 18:30)")

        if readiness.models_loaded and readiness.db_connected:
            asyncio.create_task(_background_snapshot_loop())
            logger.info(
                "Background snapshot pre-computation started | "
                "interval=%ds | lock=enabled",
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
# REQUEST MIDDLEWARE — per-endpoint rate limit
# =====================================================

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):

    path = request.url.path

    # ── Auth check ────────────────────────────────────
    if path not in PUBLIC_PATHS:
        has_jwt = (
            request.cookies.get("ms_token")
            or request.headers.get("Authorization", "").startswith("Bearer ")
        )
        if not has_jwt:
            client_key = request.headers.get("X-API-KEY")
            if not API_KEY:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Please log in.",
                )
            if client_key != API_KEY:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing API key.",
                )

    # ── Client IP ─────────────────────────────────────
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )

    # ── Per-endpoint rate limit ───────────────────────
    # FIX: uses cache.get_redis_client() not cache._redis directly
    try:
        cache = app.state.cache
        redis_client = cache.get_redis_client()
        if redis_client is not None:
            allowed, count, limit, retry_after = _check_rate_limit(
                redis_client, client_ip, path
            )
            if not allowed:
                logger.warning(
                    "Rate limit exceeded | ip=%s path=%s count=%d limit=%d",
                    client_ip, path, count, limit,
                )
                resp = JSONResponse(
                    status_code=429,
                    content=api_error(
                        f"Rate limit exceeded. Max {limit} requests/60s "
                        f"for this endpoint. Retry after {retry_after}s."
                    ),
                )
                resp.headers["Retry-After"] = str(retry_after)
                resp.headers["X-RateLimit-Limit"] = str(limit)
                resp.headers["X-RateLimit-Remaining"] = "0"
                return resp
    except HTTPException:
        raise
    except Exception:
        pass  # Redis down — fail open

    # ── Request tracking ──────────────────────────────
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
# SNAPSHOT ALIAS
# FIX: /snapshot now tracks demo quota via predict route
# =====================================================

@app.post("/snapshot")
async def snapshot(request: Request):
    """Full inference run — alias for POST /predict/live-snapshot."""
    from app.api.routes.predict import live_snapshot
    return await live_snapshot(request)


# =====================================================
# ADMIN: MANUAL SYNC
# =====================================================

@app.post("/admin/sync")
async def admin_sync(request: Request):
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
                report.get("synced", 0),
                report.get("skipped", 0),
                report.get("errors", 0),
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
