# =========================================================
# DEMO TRACKER
# Tracks demo user feature usage in Redis.
# IP + fingerprint keyed, 7-day TTL, survives logout/login.
# =========================================================

import os
import time
import hashlib
from typing import Dict, Optional

from app.inference.cache import RedisCache
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.demo_tracker")

# =========================================================
# CONFIG
# =========================================================

DEMO_REQUESTS_PER_FEATURE = int(os.getenv("DEMO_REQUESTS_PER_FEATURE", "3"))
DEMO_BLOCK_DAYS = int(os.getenv("DEMO_BLOCK_DAYS", "7"))
DEMO_BLOCK_SECONDS = DEMO_BLOCK_DAYS * 86400

# Feature groups — maps endpoint paths to feature group names
# Any endpoint not listed here is NOT counted (health, docs, metrics, etc.)
FEATURE_GROUPS = {
    "/snapshot": "snapshot",
    "/predict/live-snapshot": "snapshot",
    "/predict/signal-explanation": "signals",
    "/portfolio": "portfolio",
    "/portfolio-summary": "portfolio",
    "/drift": "drift",
    "/drift-status": "drift",
    "/agent/explain": "agent",
    "/performance": "performance",
    "/equity": "equity",
    "/predict/price-history": "equity",
}

ALL_FEATURES = list(set(FEATURE_GROUPS.values()))


# =========================================================
# IP HASHING
# =========================================================

def hash_ip(ip: str, fingerprint: str = "") -> str:
    """
    Hash IP + fingerprint into a stable anonymous key.
    Never stores the raw IP address.
    """
    raw = f"{ip}:{fingerprint}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def fingerprint_from_headers(user_agent: str, accept_language: str = "") -> str:
    """
    Create a simple browser fingerprint from request headers.
    Not foolproof but catches most casual VPN resets.
    """
    raw = f"{user_agent}:{accept_language}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# =========================================================
# DEMO TRACKER CLASS
# =========================================================

class DemoTracker:

    def __init__(self):
        self._cache = RedisCache()

    def _key(self, ip_hash: str, feature: str) -> str:
        return f"demo:{ip_hash}:feature:{feature}"

    def _blocked_key(self, ip_hash: str) -> str:
        return f"demo:{ip_hash}:blocked_until"

    def _first_visit_key(self, ip_hash: str) -> str:
        return f"demo:{ip_hash}:first_visit"

    # =========================================================
    # CORE OPERATIONS
    # =========================================================

    def get_usage(self, ip_hash: str) -> Dict[str, int]:
        """
        Get current usage counts for all feature groups.
        Returns dict like {"snapshot": 2, "portfolio": 0, ...}
        """

        usage = {}

        for feature in ALL_FEATURES:
            key = self._key(ip_hash, feature)
            try:
                raw = self._cache._redis.get(key)
                usage[feature] = int(raw) if raw else 0
            except Exception:
                usage[feature] = 0

        return usage

    def increment(self, ip_hash: str, feature: str) -> int:
        """
        Increment usage counter for a feature group.
        Returns the new count after increment.
        Sets TTL to DEMO_BLOCK_SECONDS if not already set.
        """

        if feature not in ALL_FEATURES:
            return 0

        key = self._key(ip_hash, feature)

        try:
            new_count = self._cache._redis.incr(key)

            # Set TTL on first use
            if new_count == 1:
                self._cache._redis.expire(key, DEMO_BLOCK_SECONDS)
                self._set_first_visit(ip_hash)

            logger.info(
                "Demo usage | ip=%s... feature=%s count=%d/%d",
                ip_hash[:8],
                feature,
                new_count,
                DEMO_REQUESTS_PER_FEATURE,
            )

            return new_count

        except Exception as e:
            logger.warning("Demo tracker increment failed | %s", str(e))
            return 0

    def is_feature_locked(self, ip_hash: str, feature: str) -> bool:
        """Returns True if this IP has exhausted their tries for this feature."""

        key = self._key(ip_hash, feature)

        try:
            raw = self._cache._redis.get(key)
            count = int(raw) if raw else 0
            return count >= DEMO_REQUESTS_PER_FEATURE
        except Exception:
            return False

    def is_fully_locked(self, ip_hash: str) -> bool:
        """Returns True if ALL features are exhausted."""

        for feature in ALL_FEATURES:
            if not self.is_feature_locked(ip_hash, feature):
                return False
        return True

    def get_remaining(self, ip_hash: str, feature: str) -> int:
        """Returns how many requests remain for a feature (0 = locked)."""

        key = self._key(ip_hash, feature)

        try:
            raw = self._cache._redis.get(key)
            count = int(raw) if raw else 0
            return max(0, DEMO_REQUESTS_PER_FEATURE - count)
        except Exception:
            return DEMO_REQUESTS_PER_FEATURE

    def get_reset_in_seconds(self, ip_hash: str) -> int:
        """
        Returns seconds until the first feature key expires (TTL).
        This is when the demo resets.
        """

        min_ttl = DEMO_BLOCK_SECONDS

        for feature in ALL_FEATURES:
            key = self._key(ip_hash, feature)
            try:
                raw = self._cache._redis.get(key)
                if raw and int(raw) > 0:
                    ttl = self._cache._redis.ttl(key)
                    if ttl > 0:
                        min_ttl = min(min_ttl, ttl)
            except Exception:
                pass

        return min_ttl

    def get_full_status(self, ip_hash: str) -> dict:
        """
        Returns complete demo status for the frontend.
        Used by GET /auth/me and the demo profile page.
        """

        usage = self.get_usage(ip_hash)

        features_status = {}
        for feature in ALL_FEATURES:
            count = usage.get(feature, 0)
            features_status[feature] = {
                "used": count,
                "limit": DEMO_REQUESTS_PER_FEATURE,
                "remaining": max(0, DEMO_REQUESTS_PER_FEATURE - count),
                "locked": count >= DEMO_REQUESTS_PER_FEATURE,
            }

        return {
            "ip_hash": ip_hash[:8] + "...",
            "features": features_status,
            "fully_locked": self.is_fully_locked(ip_hash),
            "reset_in_seconds": self.get_reset_in_seconds(ip_hash),
            "limit_per_feature": DEMO_REQUESTS_PER_FEATURE,
            "block_days": DEMO_BLOCK_DAYS,
        }

    def get_feature_for_path(self, path: str) -> Optional[str]:
        """
        Maps a request path to a feature group name.
        Returns None if the path is not a counted endpoint.
        """

        # Exact match first
        if path in FEATURE_GROUPS:
            return FEATURE_GROUPS[path]

        # Prefix match for dynamic routes like /equity/AAPL
        for prefix, feature in FEATURE_GROUPS.items():
            if path.startswith(prefix):
                return feature

        return None

    def _set_first_visit(self, ip_hash: str):
        """Records first visit timestamp — used for demo page greeting."""

        key = self._first_visit_key(ip_hash)

        try:
            if not self._cache._redis.exists(key):
                self._cache._redis.setex(key, DEMO_BLOCK_SECONDS, str(int(time.time())))
        except Exception:
            pass

    def get_first_visit(self, ip_hash: str) -> Optional[int]:
        """Returns first visit timestamp or None."""

        key = self._first_visit_key(ip_hash)

        try:
            raw = self._cache._redis.get(key)
            return int(raw) if raw else None
        except Exception:
            return None