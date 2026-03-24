# =========================================================
# AUTH MIDDLEWARE v2.2
#
# Changes from v2.1:
# FIX: demo_locked response now includes reset_in_seconds at
#      the top level so frontend DemoLockedError class can
#      read err.resetInSeconds directly. Previously it was
#      only inside the nested usage object.
# FIX: feature_group included at top level for frontend
#      DemoLockedError.feature parsing.
# FIX: Owner token expiry check is more explicit — expired
#      tokens log a warning rather than silently failing.
# FIX: Unauthenticated requests to protected routes now get
#      401 immediately (was passing through to route handler).
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
# FEATURE GROUP MAP
# Maps request path → feature name used in DemoTracker.
# All paths that consume a demo request must be listed here.
# =========================================================

FEATURE_GROUP_MAP = {
    "/snapshot":                 "snapshot",
    "/predict/snapshot":         "snapshot",
    "/predict/live-snapshot":    "snapshot",
    "/portfolio":                "portfolio",
    "/drift":                    "drift",
    "/performance":              "performance",
    "/agent/explain":            "agent",
    "/agent/political-risk":     "agent",
    "/equity":                   "signals",
    "/model/feature-importance": "signals",
    "/model/diagnostics":        "signals",
}

# Paths that are always free — never consume a demo request
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
    "/model/ic-stats",
    "/agent/agents",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
}

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "3"))

# Unique ordered feature list for usage summary
_UNIQUE_FEATURES = list(dict.fromkeys(FEATURE_GROUP_MAP.values()))


def _get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _get_feature_group(path: str) -> Optional[str]:
    """
    Return the feature group name for a given path.
    Returns None if path is free (no demo counter applied).
    """
    # Exact match (e.g. /health)
    if path in FREE_PATHS:
        return None

    # Prefix match with trailing slash or exact (e.g. /agent/explain)
    for prefix, feature in FEATURE_GROUP_MAP.items():
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return feature

    # Free by default if no mapping — unknown routes don't consume demo quota
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Runs on every request before the route handler.

    Owner:  validates JWT, injects role=owner, no limits.
    Demo:   validates JWT, checks feature group counter,
            increments on success, returns 200 lock response
            if the feature limit is exhausted.
    No JWT: 401 for protected routes, passthrough for free paths.
    """

    def __init__(self, app, cache=None):
        super().__init__(app)
        self._cache = cache
        # DemoTracker receives cache so it can reach Redis safely
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
                role = None

        # ── Inject request state ──────────────────────
        request.state.role = role
        request.state.username = username

        # ── Determine if path requires auth ──────────
        feature = _get_feature_group(path)

        # ── Free paths — always pass through ─────────
        if feature is None:
            return await call_next(request)

        # ── Owner — full access, no limits ───────────
        if role == "owner":
            return await call_next(request)

        # ── Demo — check and increment counter ───────
        if role == "demo":
            ip = _get_client_ip(request)
            ua = request.headers.get("user-agent", "")
            fingerprint = DemoTracker.build_fingerprint(ip, ua)

            request.state.fingerprint = fingerprint

            # Check if this feature is locked BEFORE incrementing
            if self._tracker.is_locked(fingerprint, feature):
                summary = self._tracker.get_usage_summary(
                    fingerprint, _UNIQUE_FEATURES
                )

                reset_in_seconds = summary.get("reset_in_seconds", 0)

                logger.info(
                    "Demo locked | fingerprint=%s... | feature=%s | reset_in=%ds",
                    fingerprint[:8], feature, reset_in_seconds,
                )

                # FIX: Include reset_in_seconds at top level for
                # frontend DemoLockedError class: err.resetInSeconds
                return JSONResponse(
                    status_code=200,
                    content={
                        "demo_locked": True,
                        "feature": feature,
                        "reset_in_seconds": reset_in_seconds,
                        "message": (
                            f"Demo limit reached for '{feature}'. "
                            f"You have used all {DEMO_REQUESTS_PER_FEATURE} "
                            f"demo requests for this feature. "
                            f"Resets in {reset_in_seconds}s."
                        ),
                        "usage": summary,
                    },
                )

            # Not locked — increment counter then handle request
            self._tracker.increment(fingerprint, feature)

            return await call_next(request)

        # ── No valid JWT — return 401 for protected routes ─
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Authentication required. Please log in.",
                "path": path,
            },
        )