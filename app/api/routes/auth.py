# =========================================================
# AUTH ROUTES v2.2
# FIX: create_demo_token(fingerprint) — was passing wrong args
# FIX: verify_owner_credentials → authenticate_owner (correct name)
# FIX: decode_token now exists in jwt_handler v1.1
# =========================================================

import logging
import os
from typing import Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.core.auth.jwt_handler import (
    create_owner_token,
    create_demo_token,
    decode_token,
    authenticate_owner,
)
from app.core.auth.demo_tracker import DemoTracker

logger = logging.getLogger("marketsentinel.auth")

router = APIRouter()

OWNER_USERNAME = os.getenv("OWNER_USERNAME", "shihab")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"

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


def _get_tracker(request: Request) -> DemoTracker:
    try:
        cache = request.app.state.cache
        return DemoTracker(cache=cache)
    except Exception:
        return DemoTracker(cache=None)


# =========================================================
# POST /auth/owner-login
# =========================================================

@router.post("/auth/owner-login")
async def owner_login(request: Request):
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

    # FIX: use authenticate_owner directly — no more verify_owner_credentials
    if not authenticate_owner(username, password):
        logger.warning("Owner login failed | username=%s", username)
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    token = create_owner_token(username)

    resp = JSONResponse(content={
        "authenticated": True,
        "role": "owner",
        "username": username,
        "message": "Authentication successful.",
    })

    resp.set_cookie(
        key="ms_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=60 * 60 * 24 * 30,
        path="/",
    )

    logger.info("Owner login success | username=%s", username)
    return resp


# =========================================================
# POST /auth/demo-login
# =========================================================

@router.post("/auth/demo-login")
async def demo_login(request: Request):
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    fingerprint = DemoTracker.build_fingerprint(ip, ua)

    tracker = _get_tracker(request)
    tracker.register(fingerprint)

    # FIX: create_demo_token takes fingerprint only — ip_hash is optional
    token = create_demo_token(fingerprint)

    summary = tracker.get_usage_summary(fingerprint, TRACKED_FEATURES)

    resp = JSONResponse(content={
        "authenticated": True,
        "role": "demo",
        "usage": summary,
    })

    resp.set_cookie(
        key="ms_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=60 * 60 * 24,
        path="/",
    )

    logger.info("Demo login | fingerprint=%s...", fingerprint[:8])
    return resp


# =========================================================
# GET /auth/me
# =========================================================

@router.get("/auth/me")
async def get_me(request: Request):
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

    return JSONResponse(content={"authenticated": False, "role": None})


# =========================================================
# POST /auth/logout
# =========================================================

@router.post("/auth/logout")
async def logout():
    resp = JSONResponse(content={"logged_out": True})
    resp.delete_cookie(
        key="ms_token",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )
    return resp