import redis
import json
import hashlib


class RedisCache:
    """
    Production-safe Redis cache for inference results.
    """

    def __init__(self):
        self.client = redis.Redis(
            host="redis",
            port=6379,
            decode_responses=True
        )

    # ----------------------------------

    def build_key(self, payload: dict) -> str:
        """
        Create deterministic cache key.
        """
        raw = json.dumps(payload, sort_keys=True)
        return "prediction:" + hashlib.sha256(raw.encode()).hexdigest()

    # ----------------------------------

    def get(self, key: str):

        data = self.client.get(key)

        if data:
            return json.loads(data)

        return None

    # ----------------------------------

    def set(self, key: str, value: dict, ttl=3600):
        """
        ttl = 1 hour (ideal for market data)
        """
        self.client.setex(
            key,
            ttl,
            json.dumps(value)
        )
