# =========================================================
# REDIS CACHE v13.1
#
# Changes from v13:
# FIX: Added get_redis_client() public method.
#      main.py rate limiter was accessing cache._redis
#      (private attribute) directly. Now uses public method.
#      _redis property retained for DemoTracker compatibility.
#
# Changes from v12:
# FIX: Removed list validation from set() — ms:portfolio:*
#      keys store full snapshot dicts, not signal lists.
#      The validation was firing every 2 minutes on every
#      background snapshot cycle, blocking all per-request
#      snapshot caching. set_background_snapshot() already
#      handles the background snapshot key correctly.
# FIX: _is_snapshot_key removed — validation was wrong for
#      all use cases. Dict payloads are valid everywhere.
# =========================================================

import redis
import json
import hashlib
import logging
import os
import time
import threading
from typing import Any, Optional

logger = logging.getLogger("marketsentinel.cache")

# =========================================================
# REDIS SENTINEL KEYS — never change these
# =========================================================

BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"
SNAPSHOT_KEY_PREFIX = "ms:portfolio:"

# =========================================================
# REDIS CONFIG
# =========================================================

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "180"))
INFERENCE_CACHE_TTL = int(os.getenv("INFERENCE_CACHE_TTL_SECONDS", "900"))

# Memory fallback when Redis is unavailable
_MEMORY_CACHE: dict = {}
_MEMORY_LOCK = threading.Lock()
_MEMORY_MAX_ITEMS = 256


class RedisCache:
    """
    Redis-backed cache for inference results.

    Key types:
        ms:background_snapshot:latest — background snapshot (fixed key)
        ms:portfolio:{hash}           — per-request snapshot
        ms:agent:{hash}               — agent explain results
        ms:political_risk:{hash}      — GDELT political risk results
        ms:demo:{fingerprint}:{feat}  — demo usage counters (via DemoTracker)

    All key types store JSON-serialisable dicts or lists.
    No structural validation is performed — that is the caller's job.
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._connect()

    # =====================================================
    # CONNECTION
    # =====================================================

    def _connect(self):
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
                decode_responses=False,
            )
            self._client.ping()
            self._connected = True
            logger.info(
                "Redis connected | host=%s port=%d",
                REDIS_HOST, REDIS_PORT,
            )
        except Exception as e:
            self._connected = False
            self._client = None
            logger.warning(
                "Redis unavailable — memory fallback active | %s", e
            )

    def _ensure_connected(self) -> bool:
        """Try to reconnect if disconnected. Returns True if connected."""
        if self._connected and self._client is not None:
            try:
                self._client.ping()
                return True
            except Exception:
                self._connected = False
                self._client = None
        self._connect()
        return self._connected

    # =====================================================
    # PUBLIC REDIS CLIENT ACCESS
    # =====================================================

    def get_redis_client(self) -> Optional[redis.Redis]:
        """
        Return the raw Redis client if connected, else None.

        Use this instead of accessing _redis or _client directly.
        Safe to call at any time — returns None if Redis is down.
        Used by: main.py rate limiter, DemoTracker.
        """
        if self._ensure_connected():
            return self._client
        return None

    @property
    def _redis(self) -> Optional[redis.Redis]:
        """
        Retained for DemoTracker backward compatibility.
        New code should use get_redis_client() instead.
        """
        return self.get_redis_client()

    # =====================================================
    # KEY BUILDER
    # =====================================================

    def build_key(self, payload: Any, prefix: str = SNAPSHOT_KEY_PREFIX) -> str:
        """
        Build a deterministic Redis key from payload.
        Uses SHA256 of JSON-serialised payload.
        """
        try:
            canonical = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(canonical.encode()).hexdigest()[:24]
            return f"{prefix}{digest}"
        except Exception as e:
            logger.warning("Key build failed: %s", e)
            return f"{prefix}fallback"

    # =====================================================
    # GET
    # =====================================================

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value. Returns None on any error or miss.
        Never raises.
        """
        # Try Redis first
        if self._ensure_connected() and self._client is not None:
            try:
                raw = self._client.get(key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as e:
                logger.debug("Redis get failed for %s: %s", key, e)

        # Fallback to memory cache
        with _MEMORY_LOCK:
            entry = _MEMORY_CACHE.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                _MEMORY_CACHE.pop(key, None)
                return None
            return value

    # =====================================================
    # SET
    # =====================================================

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store value in cache with TTL.
        Accepts any JSON-serialisable value (dict, list, str, etc.)
        Never raises — cache failure must not crash inference.
        Returns True on success.
        """
        if ttl is None:
            ttl = CACHE_TTL

        # Try Redis
        if self._ensure_connected() and self._client is not None:
            try:
                serialised = json.dumps(value, default=str)
                self._client.setex(key, ttl, serialised)
                return True
            except Exception as e:
                logger.debug("Redis set failed for %s: %s", key, e)

        # Fallback to memory cache
        try:
            with _MEMORY_LOCK:
                # Evict oldest if at capacity
                if len(_MEMORY_CACHE) >= _MEMORY_MAX_ITEMS:
                    oldest = next(iter(_MEMORY_CACHE))
                    _MEMORY_CACHE.pop(oldest, None)

                expires_at = time.time() + ttl if ttl > 0 else None
                _MEMORY_CACHE[key] = (value, expires_at)
            return True
        except Exception as e:
            logger.debug("Memory cache set failed: %s", e)
            return False

    # =====================================================
    # DELETE
    # =====================================================

    def delete(self, key: str) -> bool:
        """Delete a key from cache. Returns True on success."""
        if self._ensure_connected() and self._client is not None:
            try:
                self._client.delete(key)
                return True
            except Exception:
                pass

        with _MEMORY_LOCK:
            _MEMORY_CACHE.pop(key, None)

        return True

    # =====================================================
    # BACKGROUND SNAPSHOT — FIXED KEY
    # =====================================================

    def get_background_snapshot(self) -> Optional[dict]:
        """Retrieve the pre-computed background snapshot."""
        return self.get(BACKGROUND_SNAPSHOT_KEY)

    def set_background_snapshot(self, payload: dict, ttl: int = 300) -> bool:
        """
        Store the background snapshot.
        Full snapshot dict — routes, signals, executive_summary all included.
        """
        if self._ensure_connected() and self._client is not None:
            try:
                serialised = json.dumps(payload, default=str)
                self._client.setex(BACKGROUND_SNAPSHOT_KEY, ttl, serialised)
                return True
            except Exception as e:
                logger.debug("Background snapshot set failed: %s", e)

        # Memory fallback
        try:
            with _MEMORY_LOCK:
                expires_at = time.time() + ttl
                _MEMORY_CACHE[BACKGROUND_SNAPSHOT_KEY] = (payload, expires_at)
            return True
        except Exception:
            return False

    # =====================================================
    # ATOMIC DEMO COUNTER OPS (for DemoTracker)
    # =====================================================

    def incr(self, key: str) -> int:
        """Atomic increment. Returns new value or -1 on failure."""
        if self._ensure_connected() and self._client is not None:
            try:
                return int(self._client.incr(key))
            except Exception as e:
                logger.debug("Redis incr failed for %s: %s", key, e)
        return -1

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        if self._ensure_connected() and self._client is not None:
            try:
                self._client.expire(key, ttl)
                return True
            except Exception:
                pass
        return False

    def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds. -1 if no expiry, -2 if missing."""
        if self._ensure_connected() and self._client is not None:
            try:
                return int(self._client.ttl(key))
            except Exception:
                pass
        return -2

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            self._connected = False
            return False

    def health(self) -> dict:
        """Return cache health status for /health/ready endpoint."""
        connected = self.ping()
        return {
            "redis_connected": connected,
            "fallback_active": not connected,
            "memory_cache_size": len(_MEMORY_CACHE),
        }