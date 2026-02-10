import redis
import json
import hashlib
import asyncio
import logging
import os
import random
import threading


logger = logging.getLogger("marketsentinel.cache")


class RedisCache:
    """
    Institutional Redis cache.

    Guarantees:
    - never crashes inference
    - deterministic keys
    - connection singleton
    - thread-safe single-flight
    - TTL jitter (stampede protection)
    """

    _client = None
    _pool = None

    _locks = {}
    _locks_guard = threading.Lock()

    # prevent unbounded lock growth
    MAX_LOCKS = 10_000

    # ----------------------------------

    def __init__(self):

        if RedisCache._client is not None:
            self.client = RedisCache._client
            self.enabled = True
            return

        try:

            host = os.getenv("REDIS_HOST", "redis")
            port = int(os.getenv("REDIS_PORT", "6379"))

            RedisCache._pool = redis.ConnectionPool(
                host=host,
                port=port,
                socket_timeout=2,
                socket_connect_timeout=2,
                max_connections=20,
                decode_responses=True
            )

            RedisCache._client = redis.Redis(
                connection_pool=RedisCache._pool
            )

            RedisCache._client.ping()

            self.client = RedisCache._client
            self.enabled = True

            logger.info("Redis cache connected.")

        except Exception:

            self.enabled = False
            logger.warning(
                "Redis unavailable — running without cache."
            )

    # ----------------------------------

    def build_key(self, payload: dict) -> str:

        raw = json.dumps(payload, sort_keys=True, default=str)

        return "prediction:" + hashlib.sha256(raw.encode()).hexdigest()

    # ----------------------------------
    # THREAD SAFE SINGLE-FLIGHT
    # ----------------------------------

    def get_lock(self, key: str):

        with RedisCache._locks_guard:

            if len(self._locks) > self.MAX_LOCKS:
                # cheap pruning
                self._locks.clear()

            if key not in self._locks:
                self._locks[key] = asyncio.Lock()

            return self._locks[key]

    # ----------------------------------

    def get(self, key: str):

        if not self.enabled:
            return None

        try:

            data = self.client.get(key)

            if data:
                return json.loads(data)

        except Exception:
            logger.exception("Redis GET failure.")
            self.enabled = False  # fail closed

        return None

    # ----------------------------------

    def set(self, key: str, value: dict, ttl=900):

        if not self.enabled:
            return

        try:

            # TTL jitter prevents stampedes
            jitter = random.randint(0, int(ttl * 0.1))
            final_ttl = ttl + jitter

            payload = json.dumps(value, default=str)

            self.client.setex(
                key,
                final_ttl,
                payload
            )

        except Exception:
            logger.exception("Redis SET failure.")
            self.enabled = False
