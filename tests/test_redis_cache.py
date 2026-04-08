"""
Tests for RedisCache v13.

cache.py v13 changes that broke old tests:
  - _is_snapshot_key() REMOVED — tests calling it must be deleted
  - _validate_payload() REMOVED — tests calling it must be deleted
  - set() accepts ANY JSON-serialisable value (dict, list, str)
    No structural validation — that is the caller's job.

All tests in this file match the current v13 API exactly.
"""

import time
import json
from unittest.mock import MagicMock, patch

from app.inference.cache import (
    RedisCache,
    BACKGROUND_SNAPSHOT_KEY,
    SNAPSHOT_KEY_PREFIX,
    _MEMORY_CACHE,
    _MEMORY_LOCK,
)


# =====================================================
# HELPERS
# =====================================================

def _clear_memory_cache():
    """Clear memory fallback between tests."""
    with _MEMORY_LOCK:
        _MEMORY_CACHE.clear()


def _make_cache_with_mock_redis():
    """
    Return a RedisCache with a mocked Redis client.
    Avoids real Redis connection in unit tests.
    """
    cache = RedisCache.__new__(RedisCache)
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    cache._client = mock_redis
    cache._connected = True
    return cache, mock_redis


# =====================================================
# CONNECTION + PING
# =====================================================

class TestConnection:

    def test_ping_returns_false_when_no_client(self):
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False
        assert cache.ping() is False

    def test_ping_returns_true_with_mock_redis(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.ping.return_value = True
        assert cache.ping() is True

    def test_ping_returns_false_on_redis_error(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.ping.side_effect = Exception("connection refused")
        assert cache.ping() is False
        assert cache._connected is False

    def test_redis_property_returns_client_when_connected(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        assert cache._redis is mock_redis

    def test_redis_property_returns_none_when_disconnected(self):
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False
        # Patch _ensure_connected directly — it calls _connect internally
        # and _connect exception propagates. Patching the outer method
        # is cleaner and avoids implementation detail coupling.
        with patch.object(cache, "_ensure_connected", return_value=False):
            result = cache._redis
        assert result is None

    def test_health_returns_correct_structure(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        health = cache.health()
        assert "redis_connected" in health
        assert "fallback_active" in health
        assert "memory_cache_size" in health


# =====================================================
# KEY BUILDER
# =====================================================

class TestKeyBuilder:

    def test_build_key_returns_string(self):
        cache, _ = _make_cache_with_mock_redis()
        key = cache.build_key({"ticker": "AAPL"})
        assert isinstance(key, str)

    def test_build_key_starts_with_prefix(self):
        cache, _ = _make_cache_with_mock_redis()
        key = cache.build_key({"ticker": "AAPL"})
        assert key.startswith(SNAPSHOT_KEY_PREFIX)

    def test_same_payload_same_key(self):
        cache, _ = _make_cache_with_mock_redis()
        key1 = cache.build_key({"ticker": "AAPL", "type": "snapshot"})
        key2 = cache.build_key({"ticker": "AAPL", "type": "snapshot"})
        assert key1 == key2

    def test_different_payload_different_key(self):
        cache, _ = _make_cache_with_mock_redis()
        key1 = cache.build_key({"ticker": "AAPL"})
        key2 = cache.build_key({"ticker": "MSFT"})
        assert key1 != key2

    def test_key_order_independent(self):
        """Payload key order must not affect the cache key."""
        cache, _ = _make_cache_with_mock_redis()
        key1 = cache.build_key({"a": 1, "b": 2})
        key2 = cache.build_key({"b": 2, "a": 1})
        assert key1 == key2

    def test_custom_prefix(self):
        cache, _ = _make_cache_with_mock_redis()
        key = cache.build_key({"ticker": "AAPL"}, prefix="ms:agent:")
        assert key.startswith("ms:agent:")


# =====================================================
# SET + GET
# =====================================================

class TestSetGet:

    def setup_method(self):
        _clear_memory_cache()

    def test_set_and_get_dict(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        payload = {"signals": [1, 2, 3], "score": 0.9}
        serialised = json.dumps(payload).encode()
        mock_redis.get.return_value = serialised
        mock_redis.setex.return_value = True

        cache.set("testkey", payload, ttl=60)
        result = cache.get("testkey")

        assert result == payload

    def test_set_and_get_list(self):
        """v13: set() accepts lists — no list validation blocking."""
        cache, mock_redis = _make_cache_with_mock_redis()
        payload = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        mock_redis.get.return_value = json.dumps(payload).encode()
        mock_redis.setex.return_value = True

        cache.set("testkey", payload, ttl=60)
        result = cache.get("testkey")

        assert result == payload

    def test_set_dict_returns_true(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.setex.return_value = True
        result = cache.set("key", {"data": "value"}, ttl=60)
        assert result is True

    def test_set_list_returns_true(self):
        """v13: lists are valid — set() must return True for list payloads."""
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.setex.return_value = True
        result = cache.set("key", [1, 2, 3], ttl=60)
        assert result is True

    def test_set_string_returns_true(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.setex.return_value = True
        result = cache.set("key", "just a string", ttl=60)
        assert result is True

    def test_get_returns_none_on_miss(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.get.return_value = None
        result = cache.get("missing_key")
        assert result is None

    def test_get_returns_none_on_redis_error(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.get.side_effect = Exception("connection lost")
        result = cache.get("anykey")
        # Must not raise — returns None or falls back to memory
        assert result is None or isinstance(result, (dict, list, str))

    def test_set_uses_default_ttl_when_none_given(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.setex.return_value = True
        cache.set("key", {"data": 1})
        mock_redis.setex.assert_called_once()
        # TTL arg is second positional arg to setex(key, ttl, value)
        call_args = mock_redis.setex.call_args[0]
        assert call_args[1] > 0   # TTL must be positive


# =====================================================
# MEMORY FALLBACK
# =====================================================

class TestMemoryFallback:

    def setup_method(self):
        _clear_memory_cache()

    def test_set_and_get_in_memory_when_redis_down(self):
        """When Redis is unavailable, memory fallback must work."""
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False

        with patch.object(cache, "_connect"):
            cache.set("memkey", {"data": "fallback"}, ttl=60)
            result = cache.get("memkey")

        assert result == {"data": "fallback"}

    def test_memory_cache_respects_ttl(self):
        """Expired memory entries must return None."""
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False

        with patch.object(cache, "_connect"):
            # TTL of 1 second
            cache.set("expiring_key", {"val": 1}, ttl=1)
            # Manually expire by backdating
            with _MEMORY_LOCK:
                val, _ = _MEMORY_CACHE["expiring_key"]
                _MEMORY_CACHE["expiring_key"] = (val, time.time() - 1)

            result = cache.get("expiring_key")

        assert result is None

    def test_memory_fallback_evicts_when_full(self):
        """Memory cache must not grow beyond _MEMORY_MAX_ITEMS."""
        from app.inference.cache import _MEMORY_MAX_ITEMS
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False

        with patch.object(cache, "_connect"):
            for i in range(_MEMORY_MAX_ITEMS + 10):
                cache.set(f"key_{i}", {"i": i}, ttl=300)

        with _MEMORY_LOCK:
            assert len(_MEMORY_CACHE) <= _MEMORY_MAX_ITEMS


# =====================================================
# BACKGROUND SNAPSHOT
# =====================================================

class TestBackgroundSnapshot:

    def setup_method(self):
        _clear_memory_cache()

    def test_set_and_get_background_snapshot(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        payload = {
            "snapshot": {"signals": [], "snapshot_date": "2026-03-24"},
            "meta": {"model_version": "xgb_test"},
            "executive_summary": {"top_5_tickers": []},
        }
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = json.dumps(payload).encode()

        cache.set_background_snapshot(payload, ttl=300)
        result = cache.get_background_snapshot()

        assert result == payload

    def test_set_background_snapshot_uses_fixed_key(self):
        """Background snapshot must always use BACKGROUND_SNAPSHOT_KEY."""
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.setex.return_value = True

        cache.set_background_snapshot({"test": 1}, ttl=300)

        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == BACKGROUND_SNAPSHOT_KEY

    def test_set_background_snapshot_returns_true(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.setex.return_value = True
        result = cache.set_background_snapshot({"meta": {}}, ttl=300)
        assert result is True

    def test_get_background_snapshot_returns_none_on_miss(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.get.return_value = None
        result = cache.get_background_snapshot()
        assert result is None

    def test_background_snapshot_memory_fallback(self):
        """set_background_snapshot must fall back to memory when Redis down."""
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False
        payload = {"snapshot": {"signals": []}, "meta": {}}

        with patch.object(cache, "_connect"):
            result = cache.set_background_snapshot(payload, ttl=300)
            fetched = cache.get(BACKGROUND_SNAPSHOT_KEY)

        assert result is True
        assert fetched == payload


# =====================================================
# DELETE
# =====================================================

class TestDelete:

    def test_delete_returns_true(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.delete.return_value = 1
        result = cache.delete("somekey")
        assert result is True

    def test_delete_calls_redis_delete(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        cache.delete("targetkey")
        mock_redis.delete.assert_called_once_with("targetkey")


# =====================================================
# ATOMIC OPS (for DemoTracker)
# =====================================================

class TestAtomicOps:

    def test_incr_returns_new_value(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.incr.return_value = 3
        result = cache.incr("counter_key")
        assert result == 3

    def test_incr_returns_minus_one_when_redis_down(self):
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False
        with patch.object(cache, "_connect"):
            result = cache.incr("counter_key")
        assert result == -1

    def test_expire_returns_true_on_success(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.expire.return_value = True
        result = cache.expire("somekey", 300)
        assert result is True

    def test_expire_returns_false_when_redis_down(self):
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False
        with patch.object(cache, "_connect"):
            result = cache.expire("somekey", 300)
        assert result is False

    def test_ttl_returns_remaining_seconds(self):
        cache, mock_redis = _make_cache_with_mock_redis()
        mock_redis.ttl.return_value = 604800
        result = cache.ttl("demo:reg:abc123")
        assert result == 604800

    def test_ttl_returns_minus_two_when_redis_down(self):
        cache = RedisCache.__new__(RedisCache)
        cache._client = None
        cache._connected = False
        with patch.object(cache, "_connect"):
            result = cache.ttl("anykey")
        assert result == -2
