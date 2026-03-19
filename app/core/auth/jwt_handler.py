# =========================================================
# JWT HANDLER
# Owner: 30-day token, full access
# Demo:  24-hour token, limited access tracked by IP in Redis
#
# FIX: JWT_SECRET now raises RuntimeError if not set in .env.
#      Previous code silently used a visible fallback string
#      which anyone reading the source code could use to forge
#      valid tokens. Now the container refuses to start without
#      a proper secret.
#
# Setup: run this once and paste output into .env:
#   openssl rand -hex 32
# =========================================================

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.logging.logger import get_logger

logger = get_logger("marketsentinel.auth")

# =========================================================
# CONFIG — JWT_SECRET is REQUIRED
# =========================================================

_JWT_SECRET_RAW = os.getenv("JWT_SECRET", "")

if not _JWT_SECRET_RAW:
    raise RuntimeError(
        "JWT_SECRET is not set in .env — "
        "this is a security requirement. "
        "Generate one with: openssl rand -hex 32 "
        "then add JWT_SECRET=<output> to your .env file."
    )

JWT_SECRET = _JWT_SECRET_RAW
JWT_ALGORITHM = "HS256"
JWT_OWNER_EXPIRE_DAYS = int(os.getenv("JWT_OWNER_EXPIRE_DAYS", "30"))
JWT_DEMO_EXPIRE_HOURS = int(os.getenv("JWT_DEMO_EXPIRE_HOURS", "24"))

OWNER_USERNAME = os.getenv("OWNER_USERNAME", "shihab")
OWNER_PASSWORD_HASH = os.getenv("OWNER_PASSWORD_HASH", "")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =========================================================
# PASSWORD UTILITIES
# =========================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its bcrypt hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(plain_password: str) -> str:
    """Hash a plain password with bcrypt."""
    return pwd_context.hash(plain_password)


# =========================================================
# TOKEN CREATION
# =========================================================

def create_owner_token(username: str) -> str:
    """
    Create a long-lived JWT for the owner.
    Expires in JWT_OWNER_EXPIRE_DAYS (default 30 days).
    """

    expire = datetime.now(timezone.utc) + timedelta(days=JWT_OWNER_EXPIRE_DAYS)

    payload = {
        "sub": username,
        "role": "owner",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "owner",
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    logger.info(
        "Owner token created | username=%s | expires=%s",
        username, expire.isoformat(),
    )

    return token


def create_demo_token(ip_hash: str, fingerprint: str) -> str:
    """
    Create a short-lived JWT for demo users.
    Expires in JWT_DEMO_EXPIRE_HOURS (default 24 hours).
    Redis usage counters survive token expiry (7-day TTL).
    """

    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_DEMO_EXPIRE_HOURS)

    payload = {
        "sub": f"demo:{ip_hash[:16]}",
        "role": "demo",
        "ip_hash": ip_hash,
        "fingerprint": fingerprint,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "demo",
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token


# =========================================================
# TOKEN VERIFICATION
# =========================================================

def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT.
    Returns the payload dict on success, None on failure.
    """

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload

    except JWTError as e:
        logger.debug("Token verification failed | error=%s", str(e))
        return None


# =========================================================
# OWNER AUTHENTICATION
# =========================================================

def authenticate_owner(username: str, password: str) -> bool:
    """
    Verify owner credentials against .env values.
    Returns True only if both username and password match.
    """

    if not OWNER_PASSWORD_HASH:
        logger.error(
            "OWNER_PASSWORD_HASH not set in .env — "
            "run scripts/generate_owner_hash.py"
        )
        return False

    if username != OWNER_USERNAME:
        return False

    return verify_password(password, OWNER_PASSWORD_HASH)