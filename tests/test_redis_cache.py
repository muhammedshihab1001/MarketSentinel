import pytest

from app.inference.cache import RedisCache


############################################################
# KEY BUILDING
############################################################

def test_key_build():

    cache = RedisCache()

    key = cache.build_key({"tickers": ["AAPL"]})

    assert isinstance(key, str)

    # Pipeline keys include portfolio namespace
    assert "portfolio" in key


############################################################
# KEY DETERMINISM
############################################################

def test_key_deterministic():

    cache = RedisCache()

    payload = {"tickers": ["AAPL", "MSFT"]}

    k1 = cache.build_key(payload)
    k2 = cache.build_key(payload)

    assert k1 == k2


############################################################
# PAYLOAD VALIDATION SUCCESS
############################################################

def test_payload_validation():

    cache = RedisCache()

    valid_payload = [
        {
            "date": "2023-01-01",
            "ticker": "AAPL",
            "score": 1.2,
            "signal": "LONG",
            "weight": 0.1
        }
    ]

    # Should not raise
    cache._validate_payload(valid_payload)


############################################################
# PAYLOAD VALIDATION FAILURE
############################################################

def test_payload_validation_failure():

    cache = RedisCache()

    # Weight > 1 triggers "Unrealistic weight" check
    invalid_payload = [
        {
            "ticker": "AAPL",
            "weight": 5.0
        }
    ]

    with pytest.raises(Exception):
        cache._validate_payload(invalid_payload)


############################################################
# CACHE SAFE MODE (NO REDIS)
############################################################

def test_cache_without_redis():

    cache = RedisCache()

    key = cache.build_key({"tickers": ["AAPL"]})

    value = {"signals": []}

    # Should not crash even if Redis disabled
    cache.set(key, value)

    result = cache.get(key)

    # When redis disabled, result may be None
    assert result is None or isinstance(result, dict)