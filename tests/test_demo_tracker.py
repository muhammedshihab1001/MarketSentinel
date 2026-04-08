"""
Tests for DemoTracker — fingerprint, increment, lock, TTL, reset.
Covers issue #10 (dead Redis client cached at init).
"""

from unittest.mock import MagicMock, PropertyMock

from app.core.auth.demo_tracker import DemoTracker, DEMO_REQUESTS_PER_FEATURE, DEMO_TTL_SECONDS


# =====================================================
# FINGERPRINT
# =====================================================

class TestFingerprint:

    def test_same_ip_ua_same_fingerprint(self):
        fp1 = DemoTracker.build_fingerprint("1.2.3.4", "Mozilla/5.0")
        fp2 = DemoTracker.build_fingerprint("1.2.3.4", "Mozilla/5.0")
        assert fp1 == fp2

    def test_different_ip_different_fingerprint(self):
        fp1 = DemoTracker.build_fingerprint("1.2.3.4", "Mozilla/5.0")
        fp2 = DemoTracker.build_fingerprint("5.6.7.8", "Mozilla/5.0")
        assert fp1 != fp2

    def test_fingerprint_length(self):
        fp = DemoTracker.build_fingerprint("1.2.3.4", "ua")
        assert len(fp) == 32

    def test_empty_ua_still_produces_fingerprint(self):
        fp = DemoTracker.build_fingerprint("1.2.3.4", "")
        assert len(fp) == 32


# =====================================================
# REDIS CLIENT PROPERTY (issue #10 fix)
# =====================================================

class TestRedisClientProperty:

    def test_redis_property_returns_none_when_no_cache(self):
        """Without cache, _redis must return None — not raise."""
        tracker = DemoTracker(cache=None)
        assert tracker._redis is None

    def test_redis_property_fetches_from_cache_each_time(self):
        """
        Regression test for issue #10.
        _redis must be fetched from cache on every access,
        not cached at __init__ time.
        """
        mock_cache = MagicMock()
        mock_redis_1 = MagicMock()
        mock_redis_2 = MagicMock()

        # Simulate Redis reconnect — property returns different objects
        mock_cache._redis = mock_redis_1
        tracker = DemoTracker(cache=mock_cache)
        tracker._redis  # access to verify live client
        mock_cache._redis = mock_redis_2   # simulates reconnect
        second_call = tracker._redis

        # Must reflect the new client — not cached old one
        assert second_call is mock_redis_2

    def test_redis_property_returns_none_on_cache_exception(self):
        """If cache._redis raises, _redis must return None."""
        mock_cache = MagicMock()
        type(mock_cache)._redis = PropertyMock(side_effect=Exception("conn error"))
        tracker = DemoTracker(cache=mock_cache)
        assert tracker._redis is None


# =====================================================
# INCREMENT + LOCK
# =====================================================

class TestIncrementAndLock:

    def _make_tracker_with_counts(self, counts: dict):
        """Helper: creates tracker with mocked Redis returning given counts."""
        mock_redis = MagicMock()
        mock_cache = MagicMock()

        call_count = {"n": 0}

        def incr_side_effect(key):
            feature = key.split(":")[-1]
            call_count["n"] += 1
            return counts.get(feature, 1)

        def get_side_effect(key):
            feature = key.split(":")[-1]
            val = counts.get(feature, 0)
            return str(val).encode()

        mock_redis.incr.side_effect = incr_side_effect
        mock_redis.get.side_effect = get_side_effect
        mock_redis.expire.return_value = True
        mock_redis.ttl.return_value = DEMO_TTL_SECONDS
        mock_redis.exists.return_value = False
        mock_cache._redis = mock_redis

        return DemoTracker(cache=mock_cache)

    def test_not_locked_before_limit(self):
        tracker = self._make_tracker_with_counts({"snapshot": 2})
        assert tracker.is_locked("fp123", "snapshot") is False

    def test_locked_at_limit(self):
        tracker = self._make_tracker_with_counts(
            {"snapshot": DEMO_REQUESTS_PER_FEATURE}
        )
        assert tracker.is_locked("fp123", "snapshot") is True

    def test_locked_above_limit(self):
        tracker = self._make_tracker_with_counts(
            {"snapshot": DEMO_REQUESTS_PER_FEATURE + 5}
        )
        assert tracker.is_locked("fp123", "snapshot") is True

    def test_increment_returns_new_count(self):
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True
        mock_cache._redis = mock_redis

        tracker = DemoTracker(cache=mock_cache)
        result = tracker.increment("fp123", "snapshot")
        assert result == 1

    def test_increment_sets_ttl_on_first_use(self):
        """First increment must set TTL = DEMO_TTL_SECONDS."""
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        mock_redis.incr.return_value = 1  # first use
        mock_cache._redis = mock_redis

        tracker = DemoTracker(cache=mock_cache)
        tracker.increment("fp123", "snapshot")

        mock_redis.expire.assert_called_once()
        call_args = mock_redis.expire.call_args
        assert call_args[0][1] == DEMO_TTL_SECONDS

    def test_increment_no_ttl_on_subsequent_use(self):
        """Subsequent increments must NOT reset TTL."""
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        mock_redis.incr.return_value = 2  # not first use
        mock_cache._redis = mock_redis

        tracker = DemoTracker(cache=mock_cache)
        tracker.increment("fp123", "snapshot")

        mock_redis.expire.assert_not_called()


# =====================================================
# FAIL OPEN
# =====================================================

class TestFailOpen:

    def test_is_locked_returns_false_when_redis_down(self):
        """Redis unavailable must not block demo users."""
        tracker = DemoTracker(cache=None)
        assert tracker.is_locked("fp123", "snapshot") is False

    def test_increment_returns_zero_when_redis_down(self):
        """Redis unavailable must return 0 — fail open."""
        tracker = DemoTracker(cache=None)
        result = tracker.increment("fp123", "snapshot")
        assert result == 0

    def test_get_usage_summary_returns_defaults_when_redis_down(self):
        """Summary must return valid structure even with no Redis."""
        tracker = DemoTracker(cache=None)
        summary = tracker.get_usage_summary("fp123", ["snapshot", "portfolio"])

        assert "features" in summary
        assert "fully_locked" in summary
        assert "reset_in_seconds" in summary
        assert summary["features"]["snapshot"]["used"] == 0
        assert summary["features"]["snapshot"]["remaining"] == DEMO_REQUESTS_PER_FEATURE
        assert summary["features"]["snapshot"]["locked"] is False


# =====================================================
# USAGE SUMMARY
# =====================================================

class TestUsageSummary:

    def test_summary_structure(self):
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        mock_redis.get.return_value = b"2"
        mock_redis.ttl.return_value = 500000
        mock_cache._redis = mock_redis

        tracker = DemoTracker(cache=mock_cache)
        summary = tracker.get_usage_summary("fp123", ["snapshot", "drift"])

        assert "features" in summary
        assert "snapshot" in summary["features"]
        assert "drift" in summary["features"]
        assert "fully_locked" in summary
        assert "reset_in_seconds" in summary
        assert "limit_per_feature" in summary

    def test_fully_locked_when_all_features_exhausted(self):
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        # All features at limit
        mock_redis.get.return_value = str(DEMO_REQUESTS_PER_FEATURE).encode()
        mock_redis.ttl.return_value = 500000
        mock_cache._redis = mock_redis

        tracker = DemoTracker(cache=mock_cache)
        summary = tracker.get_usage_summary("fp123", ["snapshot", "drift"])

        assert summary["fully_locked"] is True

    def test_not_fully_locked_when_some_remaining(self):
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        mock_redis.get.return_value = b"1"  # below limit
        mock_redis.ttl.return_value = 500000
        mock_cache._redis = mock_redis

        tracker = DemoTracker(cache=mock_cache)
        summary = tracker.get_usage_summary("fp123", ["snapshot"])

        assert summary["fully_locked"] is False
