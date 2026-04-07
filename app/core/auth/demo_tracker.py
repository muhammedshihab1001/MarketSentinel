# =========================================================
# DEMO TRACKER v1.4
#
# CRITICAL FIX: self._redis was captured once at __init__
# time from cache._redis. When Redis reconnects (seen in
# logs: Redis connects → disconnects → reconnects), the
# tracker kept the dead client forever. All incr/get/expire
# calls silently failed (fail-open), so demo limits were
# never written to Redis. Result: users could use features
# indefinitely without hitting the limit.
#
# Fix: Remove self._redis entirely. Call cache._redis as a
# property on every operation — it always returns the live
# client or None if Redis is down.
#
# FIX 2: Fixed chained assignment bug in increment():
#   key = _usage_key = self._usage_key(...)
# was shadowing the method name locally (harmless but wrong).
#
# FIX 3: get_usage_summary now always returns valid dict
# even when Redis is completely unavailable (shows 0 usage,
# not locked — correct fail-open behavior).
# =========================================================

import os
import time
import hashlib
from typing import Optional

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "3"))
DEMO_BLOCK_DAYS = int(os.getenv("DEMO_BLOCK_DAYS", "7"))
DEMO_TTL_SECONDS = DEMO_BLOCK_DAYS * 86400


class DemoTracker:
    """
    Tracks per-feature demo usage in Redis.

    Keys:
        demo:usage:{fingerprint}:{feature}  → integer counter
        demo:reg:{fingerprint}              → registration timestamp

    All operations fail-open: if Redis is unavailable,
    usage is not tracked and requests are allowed through.

    CRITICAL: Never cache the Redis client — always fetch
    from cache._redis property which reconnects automatically.
    """

    KEY_PREFIX = "demo:usage"
    REG_PREFIX = "demo:reg"

    def __init__(self, cache=None):
        # FIX: Store cache reference, NOT the Redis client.
        # self._cache._redis is a property that reconnects.
        self._cache = cache

    # =====================================================
    # REDIS CLIENT — always fresh, never cached
    # =====================================================

    @property
    def _redis(self):
        """
        Get live Redis client on every call.
        Returns None if Redis is unavailable.
        NEVER store the return value — always use this property.
        """
        if self._cache is None:
            return None
        try:
            return self._cache._redis
        except Exception:
            return None

    # =====================================================
    # FINGERPRINT
    # =====================================================

    @staticmethod
    def build_fingerprint(ip: str, user_agent: str = "") -> str:
        """
        Build a stable fingerprint from IP + User-Agent.
        Survives logout/login — tied to the browser/client.
        """
        raw = f"{ip}:{user_agent}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    # =====================================================
    # KEY BUILDERS
    # =====================================================

    def _usage_key(self, fingerprint: str, feature: str) -> str:
        return f"{self.KEY_PREFIX}:{fingerprint}:{feature}"

    def _reg_key(self, fingerprint: str) -> str:
        return f"{self.REG_PREFIX}:{fingerprint}"

    # =====================================================
    # SAFE REDIS HELPERS
    # Each helper gets a fresh client — fail-open on None.
    # =====================================================

    def _safe_incr(self, key: str) -> Optional[int]:
        """Increment key, return new value or None on failure."""
        r = self._redis
        if r is None:
            return None
        try:
            return int(r.incr(key))
        except Exception:
            return None

    def _safe_expire(self, key: str, ttl: int) -> bool:
        """Set TTL on key, return True on success."""
        r = self._redis
        if r is None:
            return False
        try:
            r.expire(key, ttl)
            return True
        except Exception:
            return False

    def _safe_get(self, key: str) -> Optional[str]:
        """Get string value of key, return None on failure."""
        r = self._redis
        if r is None:
            return None
        try:
            val = r.get(key)
            return val.decode() if isinstance(val, bytes) else val
        except Exception:
            return None

    def _safe_set(self, key: str, value: str, ex: int) -> bool:
        """Set key with expiry, return True on success."""
        r = self._redis
        if r is None:
            return False
        try:
            r.set(key, value, ex=ex)
            return True
        except Exception:
            return False

    def _safe_ttl(self, key: str) -> int:
        """Return TTL of key in seconds. -2 = key missing, -1 = no expiry."""
        r = self._redis
        if r is None:
            return -2
        try:
            return int(r.ttl(key))
        except Exception:
            return -2

    def _safe_exists(self, key: str) -> bool:
        """Return True if key exists, False on failure."""
        r = self._redis
        if r is None:
            return False
        try:
            return bool(r.exists(key))
        except Exception:
            return False

    # =====================================================
    # REGISTRATION
    # =====================================================

    def register(self, fingerprint: str) -> bool:
        """
        Register a new demo session fingerprint.
        Sets a 7-day TTL on the registration key.
        Returns True if newly registered, False if already exists or Redis down.
        """
        key = self._reg_key(fingerprint)

        if self._safe_exists(key):
            return False

        return self._safe_set(key, str(int(time.time())), ex=DEMO_TTL_SECONDS)

    def is_registered(self, fingerprint: str) -> bool:
        """Return True if this fingerprint has an active demo session."""
        return self._safe_exists(self._reg_key(fingerprint))

    # =====================================================
    # USAGE TRACKING
    # =====================================================

    def increment(self, fingerprint: str, feature: str) -> int:
        """
        Increment usage counter for (fingerprint, feature).
        Returns new count, or 0 if Redis is unavailable (fail-open).
        On first increment, sets a 7-day TTL.

        FIX: Removed chained assignment `key = _usage_key = ...`
        that was shadowing the method name.
        """
        key = self._usage_key(fingerprint, feature)

        new_count = self._safe_incr(key)

        if new_count is None:
            # Redis down — fail open, don't block the request
            return 0

        if new_count == 1:
            # First use — set TTL so counters auto-expire
            self._safe_expire(key, DEMO_TTL_SECONDS)

        return new_count

    def get_count(self, fingerprint: str, feature: str) -> int:
        """Return current usage count for (fingerprint, feature)."""
        val = self._safe_get(self._usage_key(fingerprint, feature))
        try:
            return int(val) if val is not None else 0
        except (ValueError, TypeError):
            return 0

    def is_locked(self, fingerprint: str, feature: str) -> bool:
        """
        Return True if this feature is exhausted for this fingerprint.
        Fails open — returns False (not locked) if Redis is down,
        so Redis downtime never hard-blocks demo users.
        """
        count = self.get_count(fingerprint, feature)
        return count >= DEMO_REQUESTS_PER_FEATURE

    # =====================================================
    # USAGE SUMMARY
    # =====================================================

    def get_usage_summary(
        self,
        fingerprint: str,
        features: list,
    ) -> dict:
        """
        Return usage summary for all tracked features.
        Used by GET /auth/me and demo_locked middleware response.

        Returns:
            {
                "features": {
                    "snapshot": { "used": 2, "limit": 3, "remaining": 1, "locked": False },
                    ...
                },
                "fully_locked": False,
                "reset_in_seconds": 604800,
                "limit_per_feature": 3,
            }
        """
        result = {}
        any_unlocked = False

        for feature in features:
            used = self.get_count(fingerprint, feature)
            remaining = max(0, DEMO_REQUESTS_PER_FEATURE - used)
            locked = used >= DEMO_REQUESTS_PER_FEATURE

            result[feature] = {
                "used": used,
                "limit": DEMO_REQUESTS_PER_FEATURE,
                "remaining": remaining,
                "locked": locked,
            }

            if not locked:
                any_unlocked = True

        # Reset time = TTL of the registration key
        reset_in_seconds = self._safe_ttl(self._reg_key(fingerprint))
        if reset_in_seconds < 0:
            # Key missing or no expiry — default to full block window
            reset_in_seconds = DEMO_TTL_SECONDS

        fully_locked = not any_unlocked and len(features) > 0

        return {
            "features": result,
            "fully_locked": fully_locked,
            "reset_in_seconds": max(0, reset_in_seconds),
            "limit_per_feature": DEMO_REQUESTS_PER_FEATURE,
        }

    # =====================================================
    # RESET (owner tool)
    # =====================================================

    def reset_fingerprint(self, fingerprint: str, features: list) -> bool:
        """
        Reset all usage counters for a fingerprint.
        Called by owner to manually unlock a demo session.
        """
        r = self._redis
        if r is None:
            return False

        try:
            keys = [self._usage_key(fingerprint, f) for f in features]
            keys.append(self._reg_key(fingerprint))
            pipe = r.pipeline()
            for k in keys:
                pipe.delete(k)
            pipe.execute()
            return True
        except Exception:
            return False
