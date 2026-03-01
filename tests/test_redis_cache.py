from app.inference.cache import RedisCache


def test_key_build():

    cache = RedisCache()

    key = cache.build_key({"tickers": ["AAPL"]})

    assert isinstance(key, str)
    assert "portfolio" in key


def test_payload_validation():

    cache = RedisCache()

    valid_payload = [{
        "date": "2023-01-01",
        "ticker": "AAPL",
        "score": 1.2,
        "signal": "LONG",
        "weight": 0.1
    }]

    cache._validate_payload(valid_payload)