# =========================================================
# REDIS CACHE v14.0
#
# Changes from v13.1:
# CRITICAL FIX: Added strict mode for auth/demo tracking.
#   Previously all Redis operations were fail-open (fallback
#   to memory cache on Redis failure). This was correct for
#   inference caching but WRONG for demo usage tracking.
#
#   When Redis failed, demo counters returned 0 → users got
#   unlimited access during Redis downtime/reconnection.
#
# NEW METHODS:
#   get_strict_client() → Returns Redis client or raises
#                         RuntimeError. NO fallback.
#                         Used by DemoTracker for critical
#                         auth operations.
#
#   is_available()      → Health check. Returns True if
#                         Redis is currently connected.
#
# ARCHITECTURE:
#   - Demo tracking  → STRICT mode (fail-closed)
#   - Inference cache → NORMAL mode (fail-open)
#
# All changes from v13.1 retained.
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
    Redis-backed cache for inference results and critical auth operations.

    TWO MODES:
    ----------
    1. NORMAL (fail-open):
       - Used for: inference caching, background snapshots
       - Redis down → memory fallback
       - Access via: get(), set(), get_redis_client()

    2. STRICT (fail-closed):
       - Used for: demo tracking, rate limiting, auth
       - Redis down → RuntimeError (no fallback)
       - Access via: get_strict_client()

    Key types:
        ms:background_snapshot:latest — background snapshot (fixed key)
        ms:portfolio:{hash}           — per-request snapshot
        ms:agent:{hash}               — agent explain results
        ms:political_risk:{hash}      — GDELT political risk results
        ms:demo:{fingerprint}:{feat}  — demo usage counters (STRICT)
        ms:ratelimit:{ip}:{path}      — rate limit counters (STRICT)

    All key types store JSON-serialisable dicts or lists.
    No structural validation is performed — that is the caller's job.
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._last_connection_attempt = 0
        self._connection_attempt_interval = 5  # seconds
        self._connect()

    # =====================================================
    # CONNECTION
    # =====================================================

    def _connect(self):
        """
        Attempt Redis connection.
        On failure, sets _connected=False and logs warning.
        Called at init and by _ensure_connected().
        """
        current_time = time.time()

        # Rate limit connection attempts to avoid log spam
        if (
            current_time - self._last_connection_attempt
        ) < self._connection_attempt_interval:
            return

        self._last_connection_attempt = current_time

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
                REDIS_HOST,
                REDIS_PORT,
            )
        except Exception as e:
            self._connected = False
            self._client = None
            logger.warning("Redis unavailable — memory fallback active | error=%s", e)

    def _ensure_connected(self) -> bool:
        """
        Check connection and attempt reconnect if needed.
        Returns True if connected after check.
        """
        if self._connected and self._client is not None:
            try:
                self._client.ping()
                return True
            except Exception:
                logger.warning("Redis connection lost — attempting reconnect")
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

        FAIL-OPEN MODE: Returns None on Redis failure.
        Caller should handle None by using memory fallback.

        Used by:
        - main.py rate limiter (fail-open behavior)
        - Inference caching (fail-open behavior)

        For critical operations (demo tracking), use get_strict_client().
        """
        if self._ensure_connected():
            return self._client
        return None

    def get_strict_client(self) -> redis.Redis:
        """
        Return the raw Redis client or raise RuntimeError.

        FAIL-CLOSED MODE: Raises exception on Redis failure.
        NO FALLBACK. Used for critical operations that must not
        silently degrade (demo tracking, auth, strict rate limits).

        Raises:
            RuntimeError: If Redis is unavailable

        Used by:
        - DemoTracker (demo usage counters)
        - Strict rate limiting (future)

        Example:
            try:
                r = cache.get_strict_client()
                r.incr(key)
            except RuntimeError:
                # Handle Redis unavailability explicitly
                return error_response("Service temporarily unavailable")
        """
        if self._ensure_connected() and self._client is not None:
            return self._client

        logger.error(
            "CRITICAL: Redis unavailable in STRICT mode | "
            "operation=get_strict_client | "
            "This will block critical auth/demo operations"
        )
        raise RuntimeError(
            "Redis unavailable — critical operations blocked. "
            "Please check Redis service health."
        )

    @property
    def _redis(self) -> Optional[redis.Redis]:
        """
        Retained for backward compatibility.
        New code should use get_redis_client() or get_strict_client().
        """
        return self.get_redis_client()

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def is_available(self) -> bool:
        """
        Return True if Redis is currently connected.
        Lightweight health check — does not attempt reconnection.
        """
        return self._connected and self._client is not None

    def ping(self) -> bool:
        """
        Return True if Redis is reachable.
        Attempts reconnection if currently disconnected.
        """
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
    # GET (FAIL-OPEN)
    # =====================================================

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value. Returns None on any error or miss.
        FAIL-OPEN: Falls back to memory cache if Redis unavailable.
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
    # SET (FAIL-OPEN)
    # =====================================================

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store value in cache with TTL.
        FAIL-OPEN: Falls back to memory cache if Redis unavailable.
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
    # DELETE (FAIL-OPEN)
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
    # BACKGROUND SNAPSHOT — FIXED KEY (FAIL-OPEN)
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
    # DEPRECATED: DemoTracker should use get_strict_client()
    # =====================================================

    def incr(self, key: str) -> int:
        """
        Atomic increment. Returns new value or -1 on failure.

        DEPRECATED: This is fail-open behavior which is WRONG
        for demo tracking. DemoTracker should use get_strict_client()
        directly instead of these helper methods.

        Retained for backward compatibility only.
        """
        if self._ensure_connected() and self._client is not None:
            try:
                return int(self._client.incr(key))
            except Exception as e:
                logger.debug("Redis incr failed for %s: %s", key, e)
        return -1

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL on existing key.
        DEPRECATED: Use get_strict_client() in DemoTracker.
        """
        if self._ensure_connected() and self._client is not None:
            try:
                self._client.expire(key, ttl)
                return True
            except Exception:
                pass
        return False

    def ttl(self, key: str) -> int:
        """
        Return remaining TTL in seconds. -1 if no expiry, -2 if missing.
        DEPRECATED: Use get_strict_client() in DemoTracker.
        """
        if self._ensure_connected() and self._client is not None:
            try:
                return int(self._client.ttl(key))
            except Exception:
                pass
        return -2
