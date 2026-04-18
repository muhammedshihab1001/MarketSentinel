# =========================================================
# AUTH ROUTES v2.3
#
# SWAGGER FIX: Added Pydantic request body models so
#   /auth/owner-login and /auth/demo-login show input
#   fields in Swagger UI instead of raw JSON textarea.
#   Previously used request.json() directly which gives
#   Swagger no schema to render.
#
# FIX: OWNER_USERNAME fallback changed from "shihab" to ""
#   Hardcoded username was a security risk — anyone who
#   knew the default could use it as a username hint.
# =========================================================

import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.auth.jwt_handler import (
    create_owner_token,
    create_demo_token,
    decode_token,
    authenticate_owner,
)
from app.core.auth.demo_tracker import DemoTracker

logger = logging.getLogger("marketsentinel.auth")

router = APIRouter(tags=["auth"])

# FIX: removed hardcoded "shihab" fallback — empty string is safe default
OWNER_USERNAME = os.getenv("OWNER_USERNAME", "")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1")
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"

TRACKED_FEATURES = [
    "snapshot",
    "portfolio",
    "drift",
    "performance",
    "agent",
    "signals",
]


# =========================================================
# REQUEST BODY MODELS
# =========================================================


class OwnerLoginRequest(BaseModel):
    username: str = Field(
        ...,
        example="your_username",
        description="Owner username set in OWNER_USERNAME env var",
    )
    password: str = Field(
        ...,
        example="your_password",
        description="Owner password (plaintext — compared against bcrypt hash)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "username": "your_username",
                "password": "your_password",
            }
        }


class DemoLoginRequest(BaseModel):
    """
    No body fields required. Fingerprint is derived from IP + User-Agent.
    Submit an empty JSON body {} or leave body empty.
    """

    class Config:
        json_schema_extra = {"example": {}}


# =========================================================
# HELPERS
# =========================================================


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


@router.post(
    "/auth/owner-login",
    summary="Owner Login",
    description="""
Authenticate as owner. Returns an httpOnly JWT cookie (`ms_token`).

**Steps in Swagger:**
1. Click **Try it out**
2. Enter your username and password in the request body
3. Click **Execute**
4. The `ms_token` cookie is set automatically — all subsequent
   Swagger requests will use it for authentication.

Owner role unlocks: `/admin/*`, `/model/ic-stats`, `/model/diagnostics`.
""",
    response_description="Sets ms_token cookie. Returns role and username.",
)
async def owner_login(body: OwnerLoginRequest, request: Request):
    username = body.username.strip()
    password = body.password.strip()

    if not username or not password:
        return JSONResponse(
            status_code=400,
            content={"detail": "username and password are required"},
        )

    if not authenticate_owner(username, password):
        logger.warning("Owner login failed | username=%s", username)
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials"},
        )

    token = create_owner_token(username)

    resp = JSONResponse(
        content={
            "authenticated": True,
            "role": "owner",
            "username": username,
            "message": "Authentication successful.",
        }
    )

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


@router.post(
    "/auth/demo-login",
    summary="Demo Login",
    description="""
Start a demo session. No credentials required.

Fingerprint is derived from your IP + User-Agent and persists for 7 days.
Demo accounts get **10 requests per feature group** before being locked.

Feature groups:
- `snapshot` — /snapshot, /predict/live-snapshot
- `portfolio` — /portfolio
- `drift` — /drift
- `performance` — /performance
- `agent` — /agent/explain, /agent/political-risk
- `signals` — /equity, /model/feature-importance
""",
    response_description="Sets ms_token cookie. Returns usage summary.",
)
async def demo_login(request: Request, body: DemoLoginRequest = None):
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    fingerprint = DemoTracker.build_fingerprint(ip, ua)

    tracker = _get_tracker(request)
    tracker.register(fingerprint)

    token = create_demo_token(fingerprint)
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
        max_age=60 * 60 * 24,
        path="/",
    )

    logger.info("Demo login | fingerprint=%s...", fingerprint[:8])
    return resp


# =========================================================
# GET /auth/me
# =========================================================


@router.get(
    "/auth/me",
    summary="Current User",
    description="""
Returns current authentication state from the `ms_token` cookie.

- **Owner**: returns role, username, usage=null
- **Demo**: returns role, usage summary with per-feature request counts
- **Unauthenticated**: returns authenticated=false
""",
)
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
        return JSONResponse(
            content={
                "authenticated": True,
                "role": "owner",
                "username": username,
                "usage": None,
            }
        )

    if role == "demo":
        fingerprint = payload.get("fingerprint") or username
        tracker = _get_tracker(request)
        summary = tracker.get_usage_summary(fingerprint, TRACKED_FEATURES)
        return JSONResponse(
            content={
                "authenticated": True,
                "role": "demo",
                "username": None,
                "usage": summary,
            }
        )

    return JSONResponse(content={"authenticated": False, "role": None})


# =========================================================
# POST /auth/logout
# =========================================================


@router.post(
    "/auth/logout",
    summary="Logout",
    description="Clears the `ms_token` cookie. Works for both owner and demo sessions.",
)
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
