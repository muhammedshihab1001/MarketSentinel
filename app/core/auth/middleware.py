# =========================================================
# AUTH MIDDLEWARE v2.4
#
# Changes from v2.3:
# FIX: DemoTracker was initialized with cache=None at startup.
#      Redis cache is set up AFTER middleware init in lifespan.
#      Result: all demo usage tracking silently failed-open.
#      used count never incremented in Redis.
#      Fix: get cache from request.app.state.cache dynamically
#      on each request — always gets live Redis client.
#
# FIX: API key authentication now handled here.
#      When X-API-KEY matches API_KEY env var, request gets
#      owner-level access. Previously the API key passed
#      main.py but AuthMiddleware rejected it (no JWT = no role).
#
# All other fixes from v2.3 retained.
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
    "/snapshot":                  "snapshot",
    "/predict/snapshot":          "snapshot",
    "/predict/live-snapshot":     "snapshot",
    "/portfolio":                 "portfolio",
    "/drift":                     "drift",
    "/performance":               "performance",
    "/agent/explain":             "agent",
    "/agent/political-risk":      "agent",
    "/equity":                    "signals",
    "/model/feature-importance":  "signals",
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
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return True
    return False


def _get_feature_group(path: str) -> Optional[str]:
    if path in FREE_PATHS:
        return None
    for prefix, feature in FEATURE_GROUP_MAP.items():
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
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
    2. Valid API key    → owner access (FIX v2.4)
    3. Owner-only path  → owner JWT required
    4a. owner JWT       → full access
    4b. demo JWT        → quota check + increment (FIX v2.4: live Redis)
    4c. no token        → 401
    """

    def __init__(self, app):
        super().__init__(app)

    def _get_tracker(self, request: Request) -> DemoTracker:
        """
        FIX v2.4: Get DemoTracker with live Redis cache from app.state.
        Previously: DemoTracker(cache=None) stored at init time.
        Cache was not yet available when middleware was created.
        Now: fetch from request.app.state.cache on every request.
        """
        try:
            cache = request.app.state.cache
            return DemoTracker(cache=cache)
        except AttributeError:
            return DemoTracker(cache=None)

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

        # ── 2. API key → owner access ─────────────────
        if _has_valid_api_key(request):
            request.state.role = "owner"
            request.state.username = "api_key_client"
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

        # ── 4a. Owner ─────────────────────────────────
        if role == "owner":
            return await call_next(request)

        # ── 4b. Demo + quota ──────────────────────────
        if role == "demo":
            feature = _get_feature_group(path)

            if feature is None:
                return await call_next(request)

            ip = _get_client_ip(request)
            ua = request.headers.get("user-agent", "")
            fingerprint = DemoTracker.build_fingerprint(ip, ua)
            request.state.fingerprint = fingerprint

            tracker = self._get_tracker(request)

            if tracker.is_locked(fingerprint, feature):
                summary = tracker.get_usage_summary(fingerprint, _UNIQUE_FEATURES)
                reset_in_seconds = summary.get("reset_in_seconds", 0)

                logger.info(
                    "Demo locked | fingerprint=%s... | feature=%s | reset_in=%ds",
                    fingerprint[:8], feature, reset_in_seconds,
                )

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

            tracker.increment(fingerprint, feature)
            return await call_next(request)

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
