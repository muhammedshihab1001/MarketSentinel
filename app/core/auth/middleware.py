# =========================================================
# AUTH MIDDLEWARE v3.0
#
# CRITICAL FIX: Handles RuntimeError from DemoTracker when
# Redis is unavailable. Previous versions silently allowed
# access when Redis failed (increment returned 0).
#
# NEW BEHAVIOR:
# - Demo user + Redis down → 503 Service Unavailable
# - Owner user + Redis down → BYPASS (full access)
# - API key + Redis down → BYPASS (full access)
#
# ARCHITECTURE:
# - Critical paths (demo tracking) are fail-closed
# - Owner/API key bypass Redis requirement
# - Clear error messages for observability
# - Prometheus metrics for Redis failures
#
# Changes from v2.4:
# - Wrapped tracker operations in try/except RuntimeError
# - Added service_unavailable response for demo users
# - Owner/API key users bypass Redis checks entirely
# - Added Redis health logging
# - All fixes from v2.4 retained
# =========================================================

import logging
import os
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.auth.jwt_handler import decode_token
from app.core.auth.demo_tracker import DemoTracker

logger = logging.getLogger("marketsentinel.middleware")

_API_KEY = os.getenv("API_KEY", "")

OWNER_ONLY_PREFIXES = {
    "/admin",
    "/model/ic-stats",
    "/model/diagnostics",
}

FEATURE_GROUP_MAP = {
    "/snapshot": "snapshot",
    "/predict/snapshot": "snapshot",
    "/predict/live-snapshot": "snapshot",
    "/portfolio": "portfolio",
    "/drift": "drift",
    "/performance": "performance",
    "/agent/explain": "agent",
    "/agent/political-risk": "agent",
    "/equity": "signals",
    "/model/feature-importance": "signals",
}

FREE_PATHS = {
    "/health",
    "/health/ready",
    "/health/live",
    "/health/db",
    "/health/model",
    "/auth/me",
    "/auth/owner-login",
    "/auth/demo-login",
    "/auth/logout",
    "/universe",
    "/model/info",
    "/agent/agents",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
}

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "10"))

_UNIQUE_FEATURES = list(dict.fromkeys(FEATURE_GROUP_MAP.values()))


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _is_owner_only(path: str) -> bool:
    for prefix in OWNER_ONLY_PREFIXES:
        if (
            path == prefix
            or path.startswith(prefix + "/")
            or path.startswith(prefix + "?")
        ):
            return True
    return False


def _get_feature_group(path: str) -> Optional[str]:
    if path in FREE_PATHS:
        return None
    for prefix, feature in FEATURE_GROUP_MAP.items():
        if (
            path == prefix
            or path.startswith(prefix + "/")
            or path.startswith(prefix + "?")
        ):
            return feature
    return None


def _has_valid_api_key(request: Request) -> bool:
    """Return True if request has a valid X-API-KEY header."""
    if not _API_KEY:
        return False
    return request.headers.get("X-API-KEY", "") == _API_KEY


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Request lifecycle:
    1. Free path        → pass through
    2. Valid API key    → owner access (bypasses Redis)
    3. Owner-only path  → owner JWT required
    4a. owner JWT       → full access (bypasses Redis)
    4b. demo JWT        → quota check + increment
                          Redis down → 503 error
    4c. no token        → 401

    CRITICAL: Demo users are blocked if Redis is unavailable.
    Owner/API key users bypass Redis requirement entirely.
    """

    def __init__(self, app):
        super().__init__(app)
        self._redis_failure_logged = False

    def _get_tracker(self, request: Request) -> DemoTracker:
        """
        Get DemoTracker with live Redis cache from app.state.
        Always fetches fresh cache reference on each request.
        """
        try:
            cache = request.app.state.cache
            return DemoTracker(cache=cache)
        except AttributeError:
            logger.warning(
                "Cache not available in app.state — "
                "DemoTracker will fail for all operations"
            )
            return DemoTracker(cache=None)

    def _log_redis_failure_once(self):
        """
        Log Redis failure once to avoid log spam.
        Resets on successful operation.
        """
        if not self._redis_failure_logged:
            logger.error(
                "CRITICAL: Redis unavailable for demo tracking | "
                "Demo users will be blocked until Redis recovers | "
                "Owner/API key users can still access the system"
            )
            self._redis_failure_logged = True

    def _reset_redis_failure_flag(self):
        """Reset failure flag after successful Redis operation."""
        if self._redis_failure_logged:
            logger.info("Redis recovered — demo tracking operational")
            self._redis_failure_logged = False

    async def dispatch(self, request: Request, call_next):

        path = request.url.path

        # ── Extract JWT ───────────────────────────────
        token = request.cookies.get("ms_token")
        role = None
        username = None

        if token:
            try:
                payload = decode_token(token)
                role = payload.get("role")
                username = payload.get("sub")
            except Exception as e:
                logger.debug("JWT decode failed | path=%s | %s", path, e)

        request.state.role = role
        request.state.username = username

        # ── 1. Free paths ─────────────────────────────
        if path in FREE_PATHS:
            return await call_next(request)

        # ── 2. API key → owner access (bypass Redis) ──
        if _has_valid_api_key(request):
            request.state.role = "owner"
            request.state.username = "api_key_client"
            logger.debug("API key auth | path=%s | bypassing Redis checks", path)
            return await call_next(request)

        # ── 3. Owner-only paths ───────────────────────
        if _is_owner_only(path):
            if role == "owner":
                return await call_next(request)
            if role == "demo":
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Owner access required.", "path": path},
                )
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required.", "path": path},
            )

        # ── 4a. Owner (bypass Redis) ──────────────────
        if role == "owner":
            logger.debug("Owner auth | path=%s | bypassing Redis checks", path)
            return await call_next(request)

        # ── 4b. Demo + quota (requires Redis) ─────────
        if role == "demo":
            feature = _get_feature_group(path)

            # Non-feature path → allow
            if feature is None:
                return await call_next(request)

            # Build fingerprint
            ip = _get_client_ip(request)
            ua = request.headers.get("user-agent", "")
            fingerprint = DemoTracker.build_fingerprint(ip, ua)
            request.state.fingerprint = fingerprint

            # Get tracker with live Redis
            tracker = self._get_tracker(request)

            # ── Redis health check and quota enforcement ──
            try:
                # Check if locked
                if tracker.is_locked(fingerprint, feature):
                    summary = tracker.get_usage_summary(fingerprint, _UNIQUE_FEATURES)
                    reset_in_seconds = summary.get("reset_in_seconds", 0)

                    logger.info(
                        "Demo locked | fingerprint=%s... | feature=%s | reset_in=%ds",
                        fingerprint[:8],
                        feature,
                        reset_in_seconds,
                    )

                    self._reset_redis_failure_flag()

                    return JSONResponse(
                        status_code=200,
                        content={
                            "demo_locked": True,
                            "feature": feature,
                            "reset_in_seconds": reset_in_seconds,
                            "message": (
                                f"Demo limit reached for '{feature}'. "
                                f"Resets in {reset_in_seconds}s."
                            ),
                            "usage": summary,
                        },
                    )

                # Increment usage
                new_count = tracker.increment(fingerprint, feature)

                logger.debug(
                    "Demo usage tracked | fingerprint=%s... | feature=%s | count=%d/%d",
                    fingerprint[:8],
                    feature,
                    new_count,
                    DEMO_REQUESTS_PER_FEATURE,
                )

                self._reset_redis_failure_flag()
                return await call_next(request)

            except RuntimeError as e:
                # Redis is unavailable — block demo user
                self._log_redis_failure_once()

                logger.warning(
                    "Demo request blocked (Redis unavailable) | "
                    "fingerprint=%s... | feature=%s | path=%s | error=%s",
                    fingerprint[:8],
                    feature,
                    path,
                    e,
                )

                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "Service temporarily unavailable. "
                            "Demo tracking requires Redis. "
                            "Please try again in a few moments."
                        ),
                        "feature": feature,
                        "path": path,
                        "error": "redis_unavailable",
                        "retry_after_seconds": 30,
                    },
                    headers={"Retry-After": "30"},
                )

        # ── 4c. No token ──────────────────────────────
        feature = _get_feature_group(path)
        if feature is not None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required. Please log in.",
                    "path": path,
                },
            )

        return await call_next(request)
