# =========================================================
# AUTH MIDDLEWARE v2.3
#
# Changes from v2.2:
# NEW: OWNER_ONLY_PATHS — routes that return 403 for demo
#      users and 401 for unauthenticated users:
#        /admin/*       — sync, retrain triggers
#        /model/ic-stats — IC monitoring (owner tool)
#        /model/diagnostics — raw diagnostics
# FIX: reset_in_seconds at top level of demo_locked so
#      frontend DemoLockedError.resetInSeconds works.
# FIX: feature in demo_locked response so frontend
#      DemoLockedError.feature works.
# FIX: /predict/live-snapshot added to FEATURE_GROUP_MAP.
# FIX: Unauthenticated requests to protected routes now
#      return 401 immediately rather than passing through.
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

# =========================================================
# OWNER-ONLY PATHS
# Demo users get 403. Unauthenticated users get 401.
# These are administrative or sensitive endpoints.
# =========================================================

OWNER_ONLY_PREFIXES = {
    "/admin",                  # /admin/sync, /admin/retrain etc
    "/model/ic-stats",         # IC monitoring — owner tool
    "/model/diagnostics",      # Raw model diagnostics
}

# =========================================================
# FEATURE GROUP MAP
# Demo users get 3 requests per feature group then locked.
# =========================================================

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

# Paths always free — no auth check, no demo counter
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

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "3"))

# Unique ordered feature list for usage summaries
_UNIQUE_FEATURES = list(dict.fromkeys(FEATURE_GROUP_MAP.values()))


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _is_owner_only(path: str) -> bool:
    """Return True if path requires owner role."""
    for prefix in OWNER_ONLY_PREFIXES:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return True
    return False


def _get_feature_group(path: str) -> Optional[str]:
    """Return feature group for demo quota, or None if free."""
    if path in FREE_PATHS:
        return None
    for prefix, feature in FEATURE_GROUP_MAP.items():
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return feature
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Request lifecycle:

    1. Free path           → pass through (no token needed)
    2. Owner-only path     → owner token required, else 401/403
    3. Authenticated path:
       a. owner token      → full access, no limits
       b. demo token       → check + increment demo counter
       c. no token         → 401
    """

    def __init__(self, app, cache=None):
        super().__init__(app)
        self._cache = cache
        self._tracker = DemoTracker(cache=cache)

    async def dispatch(self, request: Request, call_next):

        path = request.url.path

        # ── Extract JWT from cookie ───────────────────
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

        # ── 2. Owner-only paths ───────────────────────
        if _is_owner_only(path):
            if role == "owner":
                return await call_next(request)
            if role == "demo":
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "This feature is restricted to owner accounts.",
                        "path": path,
                    },
                )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required.",
                    "path": path,
                },
            )

        # ── 3a. Owner — full access ───────────────────
        if role == "owner":
            return await call_next(request)

        # ── 3b. Demo — feature quota ──────────────────
        if role == "demo":
            feature = _get_feature_group(path)

            if feature is None:
                # No quota for this path — pass through
                return await call_next(request)

            ip = _get_client_ip(request)
            ua = request.headers.get("user-agent", "")
            fingerprint = DemoTracker.build_fingerprint(ip, ua)
            request.state.fingerprint = fingerprint

            if self._tracker.is_locked(fingerprint, feature):
                summary = self._tracker.get_usage_summary(
                    fingerprint, _UNIQUE_FEATURES
                )
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

            self._tracker.increment(fingerprint, feature)
            return await call_next(request)

        # ── 3c. No valid token ────────────────────────
        feature = _get_feature_group(path)
        if feature is not None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required. Please log in.",
                    "path": path,
                },
            )

        # Unknown path with no token — let route handler decide
        return await call_next(request)
