import hashlib
import logging
from typing import Optional, Dict, List

logger = logging.getLogger("marketsentinel.demo_tracker")

DEMO_REQUESTS_PER_FEATURE = 3
DEMO_BLOCK_DAYS = 7
DEMO_TTL_SECONDS = DEMO_BLOCK_DAYS * 24 * 60 * 60


class DemoTracker:
    def __init__(self, cache=None):
        self._cache = cache

        if cache is None:
            logger.warning(
                "DemoTracker initialized with cache=None — all operations will fail-open."
            )

    # =====================================================
    # REDIS PROPERTY (issue #10 FIX)
    # =====================================================

    @property
    def _redis(self):
        """
        Always fetch Redis dynamically (no stale connection).
        MUST NOT raise (tests expect None fallback).
        """
        if self._cache is None:
            return None
        try:
            return self._cache._redis
        except Exception:
            logger.warning("Redis access failed — returning None")
            return None

    # =====================================================
    # FINGERPRINT
    # =====================================================

    @staticmethod
    def build_fingerprint(ip: str, user_agent: str) -> str:
        raw = f"{ip}:{user_agent}"
        return hashlib.md5(raw.encode()).hexdigest()

    # =====================================================
    # KEYS
    # =====================================================

    def _usage_key(self, fingerprint: str, feature: str) -> str:
        return f"demo:usage:{fingerprint}:{feature}"

    def _reg_key(self, fingerprint: str) -> str:
        return f"demo:reg:{fingerprint}"

    # =====================================================
    # SAFE REDIS HELPERS (FAIL-OPEN)
    # =====================================================

    def _safe_incr(self, key: str) -> Optional[int]:
        r = self._redis
        if r is None:
            return None
        try:
            return int(r.incr(key))
        except Exception:
            logger.warning("Redis INCR failed")
            return None

    def _safe_get(self, key: str) -> Optional[int]:
        r = self._redis
        if r is None:
            return None
        try:
            val = r.get(key)
            if val is None:
                return 0
            return int(val)
        except Exception:
            logger.warning("Redis GET failed")
            return None

    def _safe_expire(self, key: str, ttl: int):
        r = self._redis
        if r is None:
            return
        try:
            r.expire(key, ttl)
        except Exception:
            logger.warning("Redis EXPIRE failed")

    def _safe_ttl(self, key: str) -> Optional[int]:
        r = self._redis
        if r is None:
            return None
        try:
            return r.ttl(key)
        except Exception:
            logger.warning("Redis TTL failed")
            return None

    # =====================================================
    # INCREMENT
    # =====================================================

    def increment(self, fingerprint: str, feature: str) -> int:
        key = self._usage_key(fingerprint, feature)

        new_count = self._safe_incr(key)

        # FAIL-OPEN
        if new_count is None:
            return 0

        # FIRST USE → SET TTL
        if new_count == 1:
            self._safe_expire(key, DEMO_TTL_SECONDS)
            logger.debug(
                "First usage tracked | fingerprint=%s... | feature=%s | ttl=%ds",
                fingerprint[:6],
                feature,
                DEMO_TTL_SECONDS,
            )

        logger.debug(
            "Usage incremented | fingerprint=%s... | feature=%s | count=%d/%d",
            fingerprint[:6],
            feature,
            new_count,
            DEMO_REQUESTS_PER_FEATURE,
        )

        return new_count

    # =====================================================
    # GET COUNT
    # =====================================================

    def get_count(self, fingerprint: str, feature: str) -> int:
        val = self._safe_get(self._usage_key(fingerprint, feature))
        if val is None:
            return 0
        return val

    # =====================================================
    # LOCK CHECK
    # =====================================================

    def is_locked(self, fingerprint: str, feature: str) -> bool:
        count = self.get_count(fingerprint, feature)
        return count >= DEMO_REQUESTS_PER_FEATURE

    # =====================================================
    # SUMMARY
    # =====================================================

    def get_usage_summary(self, fingerprint: str, features: List[str]) -> Dict:

        result = {}
        fully_locked = True

        for feature in features:
            used = self.get_count(fingerprint, feature)
            remaining = max(0, DEMO_REQUESTS_PER_FEATURE - used)
            locked = used >= DEMO_REQUESTS_PER_FEATURE

            if not locked:
                fully_locked = False

            result[feature] = {
                "used": used,
                "limit": DEMO_REQUESTS_PER_FEATURE,
                "remaining": remaining,
                "locked": locked,
            }

        ttl = self._safe_ttl(self._reg_key(fingerprint))
        if ttl is None or ttl < 0:
            ttl = DEMO_TTL_SECONDS

        return {
            "features": result,
            "fully_locked": fully_locked,
            "reset_in_seconds": ttl,
            "limit_per_feature": DEMO_REQUESTS_PER_FEATURE,
        }

    # =====================================================
    # RESET (kept from original design)
    # =====================================================

    def reset_fingerprint(self, fingerprint: str):
        r = self._redis
        if r is None:
            return

        try:
            keys = r.keys(f"demo:usage:{fingerprint}:*")
            if keys:
                r.delete(*keys)
            r.delete(self._reg_key(fingerprint))
        except Exception:
            logger.warning("Redis reset failed")
