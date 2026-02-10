import redis
import json
import hashlib


class RedisCache:
    """
    Production-safe Redis cache.

    Design Goals:
    ✔ Never crash inference
    ✔ Deterministic keys
    ✔ Fast connection pooling
    ✔ Timeout protection
    ✔ Silent fallback if Redis fails
    """

    def __init__(self):

        try:
            pool = redis.ConnectionPool(
                host="redis",
                port=6379,
                socket_timeout=2,          # prevents API freeze
                socket_connect_timeout=2,
                max_connections=10,
                decode_responses=True
            )

            self.client = redis.Redis(connection_pool=pool)

            # Test connection
            self.client.ping()

            self.enabled = True
            print("✅ Redis cache connected")

        except Exception:
            self.enabled = False
            print("⚠️ Redis not available — running without cache")

    # ----------------------------------

    def build_key(self, payload: dict) -> str:
        """
        Deterministic hash key.
        Prevents collisions.
        """

        raw = json.dumps(payload, sort_keys=True, default=str)

        return "prediction:" + hashlib.sha256(raw.encode()).hexdigest()

    # ----------------------------------

    def get(self, key: str):

        if not self.enabled:
            return None

        try:

            data = self.client.get(key)

            if data:
                return json.loads(data)

        except Exception:
            return None

        return None

    # ----------------------------------

    def set(self, key: str, value: dict, ttl=900):
        """
        ttl default = 15 minutes
        Perfect for stock prediction.
        """

        if not self.enabled:
            return

        try:

            self.client.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )

        except Exception:
            pass
