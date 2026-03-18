# =========================================================
# AUTH MIDDLEWARE
# Runs on every request before the route handler.
# Owner: validates JWT, injects role=owner, no limits.
# Demo:  validates JWT, checks feature group counter,
#        increments on success, returns lock response if exhausted.
# =========================================================

import time
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.auth.jwt_handler import verify_token
from app.core.auth.demo_tracker import (
    DemoTracker,
    hash_ip,
    fingerprint_from_headers,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.auth_middleware")

# =========================================================
# PATHS THAT NEVER REQUIRE AUTH
# =========================================================

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
    "/health/live",
    "/universe",
    "/auth/owner-login",
    "/auth/demo-login",
    "/auth/me",
    "/auth/logout",
    "/favicon.ico",
}

COOKIE_NAME = "ms_token"


# =========================================================
# HELPERS
# =========================================================

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host or "unknown"


def _get_token(request: Request) -> Optional[str]:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def _demo_locked_response(feature: str, usage: dict) -> JSONResponse:
    """
    Returns a structured JSON response when a demo feature is locked.
    NOT a 403 — the frontend handles this gracefully to show the demo page.
    """
    return JSONResponse(
        status_code=200,
        content={
            "demo_locked": True,
            "feature": feature,
            "message": f"You have used all {os.getenv('DEMO_REQUESTS_PER_FEATURE', '3')} "
                       f"free requests for '{feature}'. "
                       f"Come back in {usage.get('reset_in_seconds', 604800) // 3600} hours "
                       f"or contact Shihab for full access.",
            "usage": usage,
            "contact": {
                "github": "https://github.com/shihab",
                "linkedin": "https://linkedin.com/in/shihab",
                "portfolio": "https://shihab.dev",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def _unauthenticated_response(path: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "success": False,
            "error": "Authentication required. Please log in or use demo access.",
            "login_url": "/auth/owner-login",
            "demo_url": "/auth/demo-login",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# =========================================================
# AUTH MIDDLEWARE
# =========================================================

class AuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app):
        super().__init__(app)
        self._tracker = None

    def _get_tracker(self) -> DemoTracker:
        if self._tracker is None:
            self._tracker = DemoTracker()
        return self._tracker

    async def dispatch(self, request: Request, call_next):

        path = request.url.path

        # ── Always allow public paths ─────────────────────
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # ── Allow /auth/* paths ───────────────────────────
        if path.startswith("/auth/"):
            return await call_next(request)

        # ── Read JWT ──────────────────────────────────────
        token = _get_token(request)

        if not token:
            return _unauthenticated_response(path)

        payload = verify_token(token)

        if not payload:
            return _unauthenticated_response(path)

        role = payload.get("role")

        # ── Owner — full pass-through ─────────────────────
        if role == "owner":
            request.state.role = "owner"
            request.state.username = payload.get("sub", "owner")
            return await call_next(request)

        # ── Demo — check feature group counter ───────────
        if role == "demo":

            request.state.role = "demo"

            ip_hash = payload.get("ip_hash")
            fingerprint = payload.get("fingerprint", "")

            if not ip_hash:
                # Fallback: re-derive from current request
                ip = _get_client_ip(request)
                user_agent = request.headers.get("User-Agent", "")
                accept_lang = request.headers.get("Accept-Language", "")
                fingerprint = fingerprint_from_headers(user_agent, accept_lang)
                ip_hash = hash_ip(ip, fingerprint)

            tracker = self._get_tracker()
            feature = tracker.get_feature_for_path(path)

            if feature:
                # Check if this feature is locked
                if tracker.is_feature_locked(ip_hash, feature):
                    usage = tracker.get_full_status(ip_hash)
                    logger.info(
                        "Demo feature locked | ip=%s... feature=%s",
                        ip_hash[:8],
                        feature,
                    )
                    return _demo_locked_response(feature, usage)

                # Feature available — process request first
                response = await call_next(request)

                # Only increment on successful responses (2xx)
                if 200 <= response.status_code < 300:
                    tracker.increment(ip_hash, feature)

                return response

            # Path not in feature groups — allow freely
            return await call_next(request)

        # ── Unknown role ──────────────────────────────────
        return _unauthenticated_response(path)