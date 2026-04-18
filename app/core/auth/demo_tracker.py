# =========================================================
# DEMO TRACKER v2.0
#
# CRITICAL FIX: Uses cache.get_strict_client() instead of
# fail-open helpers. When Redis is unavailable, operations
# raise RuntimeError instead of silently returning 0/None.
#
# ARCHITECTURE CHANGE:
# Before: Redis down → return 0 → user gets unlimited access ❌
# After:  Redis down → raise error → middleware blocks access ✅
#
# Changes from v1.4:
# - Removed all _safe_* helpers (fail-open behavior)
# - All Redis operations now use get_strict_client() (fail-closed)
# - increment() raises RuntimeError instead of returning 0
# - get_count() raises RuntimeError instead of returning 0
# - is_locked() raises RuntimeError instead of returning False
# - Added comprehensive error logging
#
# Exception handling strategy:
# - Let exceptions bubble up to middleware
# - Middleware decides: block demo user or allow owner bypass
# - Clear error messages for observability
#
# All fixes from v1.4 retained.
# =========================================================

import os
import time
import hashlib
import logging

logger = logging.getLogger("marketsentinel.demo_tracker")

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "3"))
DEMO_BLOCK_DAYS = int(os.getenv("DEMO_BLOCK_DAYS", "7"))
DEMO_TTL_SECONDS = DEMO_BLOCK_DAYS * 86400


class DemoTracker:
    """
    Tracks per-feature demo usage in Redis using STRICT mode.

    Keys:
        demo:usage:{fingerprint}:{feature}  → integer counter
        demo:reg:{fingerprint}              → registration timestamp

    CRITICAL: All operations use cache.get_strict_client()
    which raises RuntimeError if Redis is unavailable.

    Fail-closed behavior: Redis down → operations blocked.
    This prevents unlimited access during Redis downtime.

    Exception handling:
    - All methods may raise RuntimeError
    - Caller (middleware) must catch and handle appropriately
    - Owner role can bypass Redis requirement
    - Demo role gets blocked with clear error message
    """

    KEY_PREFIX = "demo:usage"
    REG_PREFIX = "demo:reg"

    def __init__(self, cache=None):
        """
        Initialize DemoTracker with Redis cache reference.

        Args:
            cache: RedisCache instance (required for production use)
        """
        if cache is None:
            logger.warning(
                "DemoTracker initialized with cache=None — "
                "all operations will fail. This should only happen "
                "in tests or during app startup before cache is ready."
            )
        self._cache = cache

    # =====================================================
    # REDIS CLIENT — strict mode only
    # =====================================================

    def _get_redis_strict(self):
        """
        Get Redis client in STRICT mode.

        Returns:
            redis.Redis: Connected Redis client

        Raises:
            RuntimeError: If cache is None or Redis unavailable
        """
        if self._cache is None:
            raise RuntimeError(
                "DemoTracker has no cache instance — "
                "cannot track usage. This indicates a configuration error."
            )

        try:
            return self._cache.get_strict_client()
        except RuntimeError as e:
            logger.error(
                "CRITICAL: Redis unavailable for demo tracking | "
                "error=%s | "
                "Demo users will be blocked until Redis recovers",
                e,
            )
            raise

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
    # REGISTRATION
    # =====================================================

    def register(self, fingerprint: str) -> bool:
        """
        Register a new demo session fingerprint.
        Sets a 7-day TTL on the registration key.

        Returns:
            True if newly registered, False if already exists

        Raises:
            RuntimeError: If Redis is unavailable
        """
        r = self._get_redis_strict()
        key = self._reg_key(fingerprint)

        try:
            # Check if already exists
            if r.exists(key):
                return False

            # Set registration timestamp with TTL
            r.set(key, str(int(time.time())), ex=DEMO_TTL_SECONDS)
            logger.info(
                "Demo session registered | fingerprint=%s... | ttl=%ds",
                fingerprint[:8],
                DEMO_TTL_SECONDS,
            )
            return True
        except Exception as e:
            logger.error(
                "Demo registration failed | fingerprint=%s... | error=%s",
                fingerprint[:8],
                e,
            )
            raise RuntimeError(f"Failed to register demo session: {e}")

    def is_registered(self, fingerprint: str) -> bool:
        """
        Return True if this fingerprint has an active demo session.

        Raises:
            RuntimeError: If Redis is unavailable
        """
        r = self._get_redis_strict()
        try:
            return bool(r.exists(self._reg_key(fingerprint)))
        except Exception as e:
            logger.error(
                "Demo registration check failed | fingerprint=%s... | error=%s",
                fingerprint[:8],
                e,
            )
            raise RuntimeError(f"Failed to check demo registration: {e}")

    # =====================================================
    # USAGE TRACKING
    # =====================================================

    def increment(self, fingerprint: str, feature: str) -> int:
        """
        Increment usage counter for (fingerprint, feature).
        On first increment, sets a 7-day TTL.

        Returns:
            New count after increment

        Raises:
            RuntimeError: If Redis is unavailable

        CRITICAL CHANGE: No longer returns 0 on Redis failure.
        Instead raises RuntimeError to prevent silent degradation.
        """
        r = self._get_redis_strict()
        key = self._usage_key(fingerprint, feature)

        try:
            new_count = int(r.incr(key))

            # Set TTL on first use so counters auto-expire
            if new_count == 1:
                r.expire(key, DEMO_TTL_SECONDS)
                logger.debug(
                    "First usage tracked | fingerprint=%s... | feature=%s | ttl=%ds",
                    fingerprint[:8],
                    feature,
                    DEMO_TTL_SECONDS,
                )

            logger.debug(
                "Usage incremented | fingerprint=%s... | feature=%s | count=%d/%d",
                fingerprint[:8],
                feature,
                new_count,
                DEMO_REQUESTS_PER_FEATURE,
            )

            return new_count

        except Exception as e:
            logger.error(
                "CRITICAL: Usage increment failed | "
                "fingerprint=%s... | feature=%s | error=%s | "
                "Demo user will be blocked",
                fingerprint[:8],
                feature,
                e,
            )
            raise RuntimeError(
                f"Failed to track usage for {feature}. "
                f"Service temporarily unavailable."
            )

    def get_count(self, fingerprint: str, feature: str) -> int:
        """
        Return current usage count for (fingerprint, feature).

        Returns:
            Current usage count (0 if key doesn't exist)

        Raises:
            RuntimeError: If Redis is unavailable

        CRITICAL CHANGE: No longer returns 0 on Redis failure.
        """
        r = self._get_redis_strict()
        key = self._usage_key(fingerprint, feature)

        try:
            val = r.get(key)
            if val is None:
                return 0

            # Handle both bytes and string responses
            if isinstance(val, bytes):
                val = val.decode()

            return int(val)
        except Exception as e:
            logger.error(
                "CRITICAL: Usage count retrieval failed | "
                "fingerprint=%s... | feature=%s | error=%s",
                fingerprint[:8],
                feature,
                e,
            )
            raise RuntimeError(
                f"Failed to retrieve usage count for {feature}. "
                f"Service temporarily unavailable."
            )

    def is_locked(self, fingerprint: str, feature: str) -> bool:
        """
        Return True if this feature is exhausted for this fingerprint.

        Returns:
            True if usage >= limit, False otherwise

        Raises:
            RuntimeError: If Redis is unavailable

        CRITICAL CHANGE: No longer returns False (unlocked) on Redis failure.
        Instead raises RuntimeError to prevent unlimited access.
        """
        try:
            count = self.get_count(fingerprint, feature)
            locked = count >= DEMO_REQUESTS_PER_FEATURE

            if locked:
                logger.info(
                    "Demo feature locked | fingerprint=%s... | feature=%s | count=%d/%d",
                    fingerprint[:8],
                    feature,
                    count,
                    DEMO_REQUESTS_PER_FEATURE,
                )

            return locked
        except RuntimeError:
            # get_count already logged the error
            raise

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

        Raises:
            RuntimeError: If Redis is unavailable
        """
        r = self._get_redis_strict()

        try:
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
            reg_key = self._reg_key(fingerprint)
            reset_in_seconds = int(r.ttl(reg_key))

            # TTL returns -2 if key doesn't exist, -1 if no expiry
            if reset_in_seconds < 0:
                reset_in_seconds = DEMO_TTL_SECONDS

            fully_locked = not any_unlocked and len(features) > 0

            return {
                "features": result,
                "fully_locked": fully_locked,
                "reset_in_seconds": max(0, reset_in_seconds),
                "limit_per_feature": DEMO_REQUESTS_PER_FEATURE,
            }
        except Exception as e:
            logger.error(
                "Usage summary retrieval failed | fingerprint=%s... | error=%s",
                fingerprint[:8],
                e,
            )
            raise RuntimeError("Failed to retrieve usage summary. Service temporarily unavailable.")

    # =====================================================
    # RESET (owner tool)
    # =====================================================

    def reset_fingerprint(self, fingerprint: str, features: list) -> bool:
        """
        Reset all usage counters for a fingerprint.
        Called by owner to manually unlock a demo session.

        Returns:
            True on success

        Raises:
            RuntimeError: If Redis is unavailable
        """
        r = self._get_redis_strict()

        try:
            keys = [self._usage_key(fingerprint, f) for f in features]
            keys.append(self._reg_key(fingerprint))

            pipe = r.pipeline()
            for k in keys:
                pipe.delete(k)
            pipe.execute()

            logger.info(
                "Demo session reset | fingerprint=%s... | features=%d",
                fingerprint[:8],
                len(features),
            )
            return True
        except Exception as e:
            logger.error(
                "Demo session reset failed | fingerprint=%s... | error=%s",
                fingerprint[:8],
                e,
            )
            raise RuntimeError(f"Failed to reset demo session: {e}")
