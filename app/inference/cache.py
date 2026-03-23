# =========================================================
# REDIS CACHE v12
#
# Changes from v11:
# FIX: _validate_payload now only runs for snapshot-type
#      keys (ms:portfolio:*). Non-snapshot keys (political
#      risk, agent explain, etc.) skip validation entirely.
#      This was causing crashes when PoliticalRiskAgent
#      tried to cache GDELT results — payload is a dict,
#      not a list of signal rows.
# FIX: set() catches all exceptions silently — cache
#      failure must never crash the inference pipeline.
# FIX: get() returns None (not raises) on any error.
# =========================================================

import redis
import json
import hashlib
import logging
import os
import random
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

    Only portfolio/snapshot keys are validated for signal structure.
    All other key types bypass payload validation.
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
            return True
        self._connect()
        return self._connected

    # =====================================================
    # REDIS PROPERTY (for DemoTracker access)
    # =====================================================

    @property
    def _redis(self) -> Optional[redis.Redis]:
        """
        Expose raw Redis client for DemoTracker.
        Returns None if Redis is unavailable.
        """
        if self._ensure_connected():
            return self._client
        return None

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
    # PAYLOAD VALIDATION
    # FIX: Only validate snapshot/portfolio payloads.
    #      Skip for all other key types.
    # =====================================================

    def _is_snapshot_key(self, key: str) -> bool:
        """Return True if this key holds snapshot/signal data."""
        return (
            key.startswith(SNAPSHOT_KEY_PREFIX)
            or key == BACKGROUND_SNAPSHOT_KEY
        )

    def _validate_payload(self, payload: Any) -> None:
        """
        Validate snapshot signal list payload.
        Only called for snapshot keys — not for agent/political risk/etc.

        Raises ValueError if payload is structurally invalid.
        """
        if not isinstance(payload, list):
            raise ValueError(
                f"Snapshot payload must be a list, got {type(payload).__name__}"
            )

        for i, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"Row {i} must be a dict")

            weight = row.get("weight")
            if weight is not None:
                try:
                    w = float(weight)
                    if abs(w) > 2.0:
                        raise ValueError(
                            f"Row {i}: Unrealistic weight {w} (expected -2 to 2)"
                        )
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Row {i}: Invalid weight — {e}") from e

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
        FIX: Only validates payload for snapshot keys.
        Never raises — cache failure must not crash inference.
        Returns True on success.
        """
        if ttl is None:
            ttl = CACHE_TTL

        # FIX: Only validate snapshot/portfolio payloads
        if self._is_snapshot_key(key):
            try:
                self._validate_payload(value)
            except ValueError as e:
                logger.warning(
                    "Snapshot payload validation failed for %s: %s", key, e
                )
                return False

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
        NOTE: Uses direct set (not _validate_payload list check)
        because background snapshot is a full dict, not a signal list.
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
    # HEALTH CHECK
    # =====================================================

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    def health(self) -> dict:
        """Return cache health status for /health/ready endpoint."""
        connected = self.ping()
        return {
            "redis_connected": connected,
            "fallback_active": not connected,
            "memory_cache_size": len(_MEMORY_CACHE),
        }