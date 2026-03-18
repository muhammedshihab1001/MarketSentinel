# =========================================================
# AUTH ROUTES
# POST /auth/owner-login  — username + password → JWT cookie
# POST /auth/demo-login   — no credentials → demo JWT + Redis register
# GET  /auth/me           — current user role + usage stats
# POST /auth/logout       — clears JWT cookie (Redis survives)
# =========================================================

import os
import time
import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.core.auth.jwt_handler import (
    authenticate_owner,
    create_owner_token,
    create_demo_token,
    verify_token,
)
from app.core.auth.demo_tracker import (
    DemoTracker,
    hash_ip,
    fingerprint_from_headers,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "ms_token"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"  # Set to 1 in production HTTPS
COOKIE_SAMESITE = "lax"


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class OwnerLoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    role: str
    message: str


# =========================================================
# HELPERS
# =========================================================

def _get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host or "unknown"


def _set_auth_cookie(response: Response, token: str, days: int = 1):
    """Set JWT as httpOnly cookie."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=days * 86400,
        path="/",
    )


def _get_token_from_request(request: Request) -> Optional[str]:
    """Read JWT from cookie or Authorization header."""

    # Cookie first
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return token

    # Authorization header fallback (for API clients)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


# =========================================================
# POST /auth/owner-login
# =========================================================

@router.post("/owner-login")
async def owner_login(body: OwnerLoginRequest, response: Response):
    """
    Owner login with username + password.
    Returns 30-day JWT cookie on success.
    Credentials validated against .env (no database).
    """

    if not authenticate_owner(body.username, body.password):
        logger.warning(
            "Failed owner login attempt | username=%s", body.username
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )

    token = create_owner_token(body.username)

    _set_auth_cookie(response, token, days=30)

    logger.info("Owner logged in | username=%s", body.username)

    return {
        "success": True,
        "role": "owner",
        "message": "Welcome back, Shihab.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# POST /auth/demo-login
# =========================================================

@router.post("/demo-login")
async def demo_login(request: Request, response: Response):
    """
    Demo login — no credentials required.
    Registers IP + fingerprint in Redis for usage tracking.
    Returns 24-hour JWT cookie.
    Demo counters persist in Redis for 7 days regardless of token expiry.
    """

    ip = _get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    accept_language = request.headers.get("Accept-Language", "")

    fingerprint = fingerprint_from_headers(user_agent, accept_language)
    ip_hash = hash_ip(ip, fingerprint)

    token = create_demo_token(ip_hash, fingerprint)

    _set_auth_cookie(response, token, days=1)

    tracker = DemoTracker()
    status = tracker.get_full_status(ip_hash)

    logger.info("Demo login | ip=%s... features=%s", ip_hash[:8], status["features"])

    return {
        "success": True,
        "role": "demo",
        "message": "Welcome! You have 3 free requests per feature.",
        "usage": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# GET /auth/me
# =========================================================

@router.get("/me")
async def get_me(request: Request):
    """
    Returns current user role and usage stats.
    Used by frontend on mount to restore auth state.
    """

    token = _get_token_from_request(request)

    if not token:
        return {
            "authenticated": False,
            "role": None,
            "message": "Not logged in",
        }

    payload = verify_token(token)

    if not payload:
        return {
            "authenticated": False,
            "role": None,
            "message": "Session expired",
        }

    role = payload.get("role")

    if role == "owner":
        return {
            "authenticated": True,
            "role": "owner",
            "username": payload.get("sub"),
            "message": "Full access",
        }

    if role == "demo":
        ip_hash = payload.get("ip_hash", "")
        tracker = DemoTracker()
        status = tracker.get_full_status(ip_hash)

        return {
            "authenticated": True,
            "role": "demo",
            "usage": status,
            "message": "Demo access",
        }

    return {
        "authenticated": False,
        "role": None,
        "message": "Unknown role",
    }


# =========================================================
# POST /auth/logout
# =========================================================

@router.post("/logout")
async def logout(response: Response):
    """
    Clears the JWT cookie.
    Does NOT clear Redis demo counters — those persist for 7 days.
    A demo user who logs out and logs back in still has the same usage counts.
    """

    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
    )

    return {
        "success": True,
        "message": "Logged out successfully.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }