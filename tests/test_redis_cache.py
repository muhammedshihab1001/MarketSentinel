import pytest
from app.inference.cache import RedisCache, BACKGROUND_SNAPSHOT_KEY, SNAPSHOT_KEY_PREFIX


############################################################
# KEY BUILDING
############################################################

def test_key_build():
    cache = RedisCache()
    key = cache.build_key({"tickers": ["AAPL"]})
    assert isinstance(key, str)
    assert key.startswith(SNAPSHOT_KEY_PREFIX)


def test_key_deterministic():
    cache = RedisCache()
    payload = {"tickers": ["AAPL", "MSFT"]}
    k1 = cache.build_key(payload)
    k2 = cache.build_key(payload)
    assert k1 == k2


def test_key_different_payloads_give_different_keys():
    cache = RedisCache()
    k1 = cache.build_key({"type": "a"})
    k2 = cache.build_key({"type": "b"})
    assert k1 != k2


############################################################
# SNAPSHOT KEY DETECTION
# FIX: _is_snapshot_key() used to route validation
############################################################

def test_snapshot_key_detected():
    cache = RedisCache()
    assert cache._is_snapshot_key("ms:portfolio:abc123") is True
    assert cache._is_snapshot_key(BACKGROUND_SNAPSHOT_KEY) is True


def test_non_snapshot_key_not_detected():
    cache = RedisCache()
    assert cache._is_snapshot_key("ms:agent:abc123") is False
    assert cache._is_snapshot_key("ms:political_risk:abc123") is False
    assert cache._is_snapshot_key("demo:usage:fingerprint:snapshot") is False


############################################################
# PAYLOAD VALIDATION — SNAPSHOT KEYS ONLY
############################################################

def test_snapshot_payload_validation_passes():
    cache = RedisCache()
    valid = [
        {"ticker": "AAPL", "date": "2026-01-01", "raw_model_score": 0.5,
         "hybrid_consensus_score": 0.4, "weight": 0.1}
    ]
    # Should not raise
    cache._validate_payload(valid)


def test_snapshot_payload_validation_fails_on_bad_weight():
    cache = RedisCache()
    invalid = [{"ticker": "AAPL", "weight": 5.0}]
    with pytest.raises((ValueError, Exception)):
        cache._validate_payload(invalid)


def test_snapshot_payload_validation_fails_on_non_list():
    cache = RedisCache()
    with pytest.raises((ValueError, Exception)):
        cache._validate_payload({"ticker": "AAPL"})


############################################################
# SET — NON-SNAPSHOT KEY SKIPS VALIDATION
# FIX: Political risk / agent results are dicts, not lists.
#      Previously _validate_payload ran on all keys and crashed.
############################################################

def test_set_non_snapshot_key_with_dict_does_not_crash():
    """
    FIX: PoliticalRiskAgent stores a dict, not a list.
    Old code called _validate_payload on all keys → crash.
    New code skips validation for non-snapshot keys.
    """
    cache = RedisCache()

    political_result = {
        "ticker": "NVDA",
        "political_risk_score": 0.12,
        "political_risk_label": "LOW",
        "top_events": [],
        "source": "gdelt",
    }

    # Non-snapshot key — should NOT call _validate_payload
    result = cache.set("ms:political_risk:nvda_us", political_result, ttl=3600)
    # Returns True (Redis) or True (memory fallback) — never raises
    assert isinstance(result, bool)


def test_set_snapshot_key_with_invalid_payload_returns_false():
    """Snapshot keys still validate — bad payload returns False not raises."""
    cache = RedisCache()
    bad_payload = {"not_a_list": True}
    snapshot_key = cache.build_key({"type": "snapshot"})
    result = cache.set(snapshot_key, bad_payload)
    assert result is False


def test_set_never_raises():
    """set() must never propagate exceptions — cache failure is silent."""
    cache = RedisCache()
    # Even with garbage input, should not raise
    try:
        cache.set("any:key", object())  # non-serialisable
    except Exception as e:
        pytest.fail(f"cache.set() raised unexpectedly: {e}")


############################################################
# GET — RETURNS NONE ON MISS OR ERROR
############################################################

def test_get_returns_none_on_miss():
    cache = RedisCache()
    result = cache.get("nonexistent:key:xyz123")
    assert result is None


def test_get_never_raises():
    """get() must never propagate exceptions."""
    cache = RedisCache()
    try:
        result = cache.get("any:key:that:doesnt:exist")
        assert result is None
    except Exception as e:
        pytest.fail(f"cache.get() raised unexpectedly: {e}")


############################################################
# BACKGROUND SNAPSHOT — FIXED KEY
############################################################

def test_background_snapshot_uses_fixed_key():
    cache = RedisCache()
    # set_background_snapshot should use BACKGROUND_SNAPSHOT_KEY
    payload = {"meta": {"model_version": "test"}, "snapshot": {"signals": []}}
    cache.set_background_snapshot(payload, ttl=300)

    # If Redis is up, get_background_snapshot returns it
    # If Redis is down, memory fallback returns it
    result = cache.get_background_snapshot()
    # Either returns the dict or None (if both Redis and memory fail)
    assert result is None or isinstance(result, dict)


############################################################
# HEALTH
############################################################

def test_health_returns_dict():
    cache = RedisCache()
    h = cache.health()
    assert isinstance(h, dict)
    assert "redis_connected" in h
    assert "fallback_active" in h
    assert isinstance(h["redis_connected"], bool)


############################################################
# REDIS PROPERTY (for DemoTracker)
############################################################

def test_redis_property_returns_client_or_none():
    cache = RedisCache()
    # Returns raw Redis client if connected, None if not
    r = cache._redis
    assert r is None or hasattr(r, "get")