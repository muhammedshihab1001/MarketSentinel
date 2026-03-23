# =========================================================
# DEMO TRACKER v1.3
# FIX: Wrap every _redis call in try/except — fail-open
#      when Redis is down so API doesn't crash.
# FIX: Removed bare _redis attribute access — all ops go
#      through safe helper methods with fallback returns.
# Tracks demo user feature usage in Redis.
# IP + fingerprint keyed, 7-day TTL, survives logout/login.
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
    This prevents Redis downtime from breaking the demo flow.
    """

    KEY_PREFIX = "demo:usage"
    REG_PREFIX = "demo:reg"

    def __init__(self, cache=None):
        self._cache = cache
        self._redis = None

        if cache is not None:
            try:
                self._redis = cache._redis
            except Exception:
                self._redis = None

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
    # All operations wrapped in try/except — fail-open.
    # =====================================================

    def _safe_incr(self, key: str) -> Optional[int]:
        """Increment key, return new value or None on failure."""
        if self._redis is None:
            return None
        try:
            return self._redis.incr(key)
        except Exception:
            return None

    def _safe_expire(self, key: str, ttl: int) -> bool:
        """Set TTL on key, return True on success."""
        if self._redis is None:
            return False
        try:
            self._redis.expire(key, ttl)
            return True
        except Exception:
            return False

    def _safe_get(self, key: str) -> Optional[str]:
        """Get string value of key, return None on failure."""
        if self._redis is None:
            return None
        try:
            val = self._redis.get(key)
            return val.decode() if isinstance(val, bytes) else val
        except Exception:
            return None

    def _safe_set(self, key: str, value: str, ex: int) -> bool:
        """Set key with expiry, return True on success."""
        if self._redis is None:
            return False
        try:
            self._redis.set(key, value, ex=ex)
            return True
        except Exception:
            return False

    def _safe_ttl(self, key: str) -> int:
        """Return TTL of key in seconds, -1 on failure."""
        if self._redis is None:
            return -1
        try:
            return self._redis.ttl(key)
        except Exception:
            return -1

    def _safe_exists(self, key: str) -> bool:
        """Return True if key exists, False on failure."""
        if self._redis is None:
            return False
        try:
            return bool(self._redis.exists(key))
        except Exception:
            return False

    # =====================================================
    # REGISTRATION
    # =====================================================

    def register(self, fingerprint: str) -> bool:
        """
        Register a new demo session fingerprint.
        Sets a 7-day TTL on the registration key.
        Returns True if registered, False if already registered or Redis down.
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
        """
        key = _usage_key = self._usage_key(fingerprint, feature)

        new_count = self._safe_incr(key)

        if new_count is None:
            # Redis down — fail open, don't block the request
            return 0

        if new_count == 1:
            # First use — set expiry
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
        Fails open — returns False (not locked) if Redis is down.
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
        Used by GET /auth/me to populate the demo usage block.

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
        if self._redis is None:
            return False

        try:
            keys = [self._usage_key(fingerprint, f) for f in features]
            keys.append(self._reg_key(fingerprint))
            pipe = self._redis.pipeline()
            for k in keys:
                pipe.delete(k)
            pipe.execute()
            return True
        except Exception:
            return False