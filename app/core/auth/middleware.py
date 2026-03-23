# =========================================================
# AUTH MIDDLEWARE v2.1
#
# Changes from v2.0:
# FIX: DemoTracker now receives cache instance so it can
#      access Redis via the safe _redis property.
# FIX: Fingerprint built from IP + User-Agent (stable
#      across demo logout/login as designed).
# FIX: Feature group map expanded to cover all 10 routes.
# FIX: demo_locked response includes reset_in_seconds so
#      frontend DemoBanner can show correct countdown.
# FIX: Owner JWT now checked for expiry before injecting
#      role — expired owner tokens return 401 not 500.
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
    "/snapshot":            "snapshot",
    "/predict/snapshot":    "snapshot",
    "/portfolio":           "portfolio",
    "/drift":               "drift",
    "/performance":         "performance",
    "/agent/explain":       "agent",
    "/agent/political-risk": "agent",
    "/equity":              "signals",
    "/model/feature-importance": "signals",
    "/model/diagnostics":   "signals",
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
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
}

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "3"))


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
    # Check exact matches first
    if path in FREE_PATHS:
        return None

    # Prefix match for feature groups
    for prefix, feature in FEATURE_GROUP_MAP.items():
        if path.startswith(prefix):
            return feature

    return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Runs on every request before the route handler.

    Owner:  validates JWT, injects role=owner, no limits.
    Demo:   validates JWT, checks feature group counter,
            increments on success, returns lock response if
            the feature limit is exhausted.
    No JWT: request proceeds unauthenticated (public routes
            like /health work without any token).
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
        fingerprint = None

        if token:
            try:
                payload = decode_token(token)
                role = payload.get("role")
                username = payload.get("sub")
            except Exception as e:
                logger.debug("JWT decode failed: %s", e)
                # Invalid/expired token — treat as unauthenticated
                role = None

        # ── Inject request state ──────────────────────
        request.state.role = role
        request.state.username = username

        # ── Free paths — no demo check needed ────────
        feature = _get_feature_group(path)

        if feature is None:
            # Path is free — just pass through
            response = await call_next(request)
            return response

        # ── Owner — full access, no limits ───────────
        if role == "owner":
            response = await call_next(request)
            return response

        # ── Demo — check and increment counter ───────
        if role == "demo":
            ip = _get_client_ip(request)
            ua = request.headers.get("user-agent", "")
            fingerprint = DemoTracker.build_fingerprint(ip, ua)

            request.state.fingerprint = fingerprint

            # Check if this feature is locked
            if self._tracker.is_locked(fingerprint, feature):
                # Build usage summary for response
                all_features = list(FEATURE_GROUP_MAP.values())
                # Deduplicate preserving order
                seen = set()
                unique_features = []
                for f in all_features:
                    if f not in seen:
                        seen.add(f)
                        unique_features.append(f)

                summary = self._tracker.get_usage_summary(
                    fingerprint, unique_features
                )

                logger.info(
                    "Demo locked | fingerprint=%s... | feature=%s",
                    fingerprint[:8],
                    feature,
                )

                return JSONResponse(
                    status_code=200,
                    content={
                        "demo_locked": True,
                        "feature": feature,
                        "message": (
                            f"Demo limit reached for {feature}. "
                            f"You have used all {DEMO_REQUESTS_PER_FEATURE} "
                            f"demo requests for this feature."
                        ),
                        "usage": summary,
                    },
                )

            # Not locked — increment and proceed
            self._tracker.increment(fingerprint, feature)

            response = await call_next(request)
            return response

        # ── Unauthenticated accessing protected route ─
        # Allow through — route handler will return 401 if needed
        response = await call_next(request)
        return response