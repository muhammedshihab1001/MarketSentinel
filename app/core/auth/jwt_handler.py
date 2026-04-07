# =========================================================
# JWT HANDLER v1.1
# FIX: Added decode_token() — alias for verify_token() with
#      raising behaviour. middleware.py + auth.py import this.
# FIX: Added verify_owner_credentials() — alias for
#      authenticate_owner(). auth.py calls this name.
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
# CONFIG
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
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


# =========================================================
# TOKEN CREATION
# =========================================================

def create_owner_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_OWNER_EXPIRE_DAYS)
    payload = {
        "sub": username,
        "role": "owner",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "owner",
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.info("Owner token created | username=%s | expires=%s", username, expire.isoformat())
    return token


def create_demo_token(fingerprint: str, ip_hash: str = "") -> str:
    """
    Create a short-lived JWT for demo users.
    FIX: ip_hash is now optional — auth.py calls create_demo_token(fingerprint).
    Old signature was create_demo_token(ip_hash, fingerprint) — fixed here.
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_DEMO_EXPIRE_HOURS)
    payload = {
        "sub": f"demo:{fingerprint[:16]}",
        "role": "demo",
        "ip_hash": ip_hash or fingerprint[:16],
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
    Returns payload dict on success, None on failure.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug("Token verification failed | error=%s", str(e))
        return None


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT. Raises Exception if invalid.
    FIX: middleware.py and auth.py import decode_token.
    verify_token() returns None on failure; this raises instead.
    """
    payload = verify_token(token)
    if payload is None:
        raise Exception("Invalid or expired token")
    return payload


# =========================================================
# OWNER AUTHENTICATION
# =========================================================

def authenticate_owner(username: str, password: str) -> bool:
    if not OWNER_PASSWORD_HASH:
        logger.error("OWNER_PASSWORD_HASH not set in .env — run scripts/generate_owner_hash.py")
        return False
    if username != OWNER_USERNAME:
        return False
    return verify_password(password, OWNER_PASSWORD_HASH)


def verify_owner_credentials(username: str, password: str) -> bool:
    """
    FIX: auth.py calls verify_owner_credentials — alias for authenticate_owner.
    """
    return authenticate_owner(username, password)
