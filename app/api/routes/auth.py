# =========================================================
# AUTH ROUTES v2.1
#
# Changes from v2.0:
# FIX: GET /auth/me now calls DemoTracker.get_usage_summary()
#      instead of building usage manually — matches the
#      shape frontend authStore.setAuth() expects.
# FIX: /auth/me returns reset_in_seconds correctly.
# FIX: /auth/demo-login registers fingerprint in DemoTracker
#      so usage tracking starts immediately after login.
# FIX: /auth/logout clears cookie with correct same_site
#      and secure settings from env.
# =========================================================

import logging
import os
from typing import Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.core.auth.jwt_handler import create_owner_token, create_demo_token, decode_token
from app.core.auth.demo_tracker import DemoTracker

logger = logging.getLogger("marketsentinel.auth")

router = APIRouter()

OWNER_USERNAME = os.getenv("OWNER_USERNAME", "shihab")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"

# Feature groups tracked for demo usage display
TRACKED_FEATURES = [
    "snapshot",
    "portfolio",
    "drift",
    "performance",
    "agent",
    "signals",
]


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _get_tracker(request: Request) -> Optional[DemoTracker]:
    """Get DemoTracker from app state if available."""
    try:
        cache = request.app.state.cache
        return DemoTracker(cache=cache)
    except Exception:
        return DemoTracker(cache=None)


# =========================================================
# POST /auth/owner-login
# =========================================================

@router.post("/auth/owner-login")
async def owner_login(request: Request, response: Response):
    """
    Authenticate as owner with username + password.
    Sets a 30-day httpOnly JWT cookie on success.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})

    username = body.get("username", "").strip()
    password = body.get("password", "").strip()

    if not username or not password:
        return JSONResponse(
            status_code=400,
            content={"detail": "username and password are required"},
        )

    # Verify credentials via jwt_handler (bcrypt compare)
    try:
        from app.core.auth.jwt_handler import verify_owner_credentials
        if not verify_owner_credentials(username, password):
            logger.warning("Owner login failed | username=%s", username)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid credentials"},
            )
    except Exception as e:
        logger.error("Owner credential check error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Authentication error"},
        )

    token = create_owner_token(username)

    resp = JSONResponse(
        content={
            "authenticated": True,
            "role": "owner",
            "username": username,
        }
    )

    resp.set_cookie(
        key="ms_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=60 * 60 * 24 * 30,  # 30 days
        path="/",
    )

    logger.info("Owner login success | username=%s", username)
    return resp


# =========================================================
# POST /auth/demo-login
# =========================================================

@router.post("/auth/demo-login")
async def demo_login(request: Request):
    """
    Start a demo session — no credentials required.
    Sets a 24-hour httpOnly JWT cookie.
    Registers fingerprint in DemoTracker for usage tracking.
    """
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    fingerprint = DemoTracker.build_fingerprint(ip, ua)

    # Register fingerprint (starts the 7-day TTL clock)
    tracker = _get_tracker(request)
    tracker.register(fingerprint)

    token = create_demo_token(fingerprint)

    # Build initial usage summary (all zeros)
    summary = tracker.get_usage_summary(fingerprint, TRACKED_FEATURES)

    resp = JSONResponse(
        content={
            "authenticated": True,
            "role": "demo",
            "usage": summary,
        }
    )

    resp.set_cookie(
        key="ms_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=60 * 60 * 24,  # 24 hours
        path="/",
    )

    logger.info("Demo login | fingerprint=%s...", fingerprint[:8])
    return resp


# =========================================================
# GET /auth/me
# =========================================================

@router.get("/auth/me")
async def get_me(request: Request):
    """
    Return current authentication state and usage.

    Owner response:
        { authenticated: true, role: "owner", username: "shihab", usage: null }

    Demo response:
        { authenticated: true, role: "demo", usage: { features: {...}, fully_locked, reset_in_seconds } }

    Unauthenticated:
        { authenticated: false, role: null }
    """
    token = request.cookies.get("ms_token")

    if not token:
        return JSONResponse(content={"authenticated": False, "role": None})

    try:
        payload = decode_token(token)
    except Exception:
        return JSONResponse(content={"authenticated": False, "role": None})

    role = payload.get("role")
    username = payload.get("sub")

    if role == "owner":
        return JSONResponse(content={
            "authenticated": True,
            "role": "owner",
            "username": username,
            "usage": None,
        })

    if role == "demo":
        fingerprint = payload.get("fingerprint") or username

        tracker = _get_tracker(request)
        summary = tracker.get_usage_summary(fingerprint, TRACKED_FEATURES)

        return JSONResponse(content={
            "authenticated": True,
            "role": "demo",
            "username": None,
            "usage": summary,
        })

    # Unknown role in token
    return JSONResponse(content={"authenticated": False, "role": None})


# =========================================================
# POST /auth/logout
# =========================================================

@router.post("/auth/logout")
async def logout():
    """Clear the auth cookie and end the session."""
    resp = JSONResponse(content={"logged_out": True})

    resp.delete_cookie(
        key="ms_token",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )

    return resp