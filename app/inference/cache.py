import redis
import json
import hashlib
import logging
import os
import random
import threading
import time

from core.schema.feature_schema import get_schema_signature


logger = logging.getLogger("marketsentinel.cache")


class RedisCache:
    """
    Institutional Redis cache.

    Guarantees:
    - thread-safe single-flight
    - schema-bound keys
    - safe degradation
    - stampede protection
    """

    _client = None
    _pool = None

    _locks = {}
    _locks_guard = threading.Lock()

    MAX_LOCKS = 10_000

    RETRY_SECONDS = 30

    # ----------------------------------

    def __init__(self):

        if RedisCache._client is not None:
            self.client = RedisCache._client
            self.enabled = True
            self._disabled_until = 0
            return

        self._connect()

    # ----------------------------------

    def _connect(self):

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
            self._disabled_until = 0

            logger.info("Redis cache connected.")

        except Exception:

            self.enabled = False
            self._disabled_until = time.time() + self.RETRY_SECONDS

            logger.warning(
                "Redis unavailable — retrying soon."
            )

    # ----------------------------------

    def _maybe_reconnect(self):

        if self.enabled:
            return

        if time.time() < self._disabled_until:
            return

        logger.info("Attempting Redis reconnect...")
        self._connect()

    # ----------------------------------

    def build_key(self, payload: dict) -> str:

        raw = json.dumps(payload, sort_keys=True, default=str)

        schema = get_schema_signature()

        return "prediction:" + schema + ":" + hashlib.sha256(raw.encode()).hexdigest()

    # ----------------------------------
    # THREAD SAFE SINGLE-FLIGHT
    # ----------------------------------

    def get_lock(self, key: str):

        with RedisCache._locks_guard:

            if key not in self._locks:

                if len(self._locks) >= self.MAX_LOCKS:
                    # prune oldest
                    RedisCache._locks.pop(
                        next(iter(self._locks))
                    )

                self._locks[key] = threading.Lock()

            return self._locks[key]

    # ----------------------------------

    def get(self, key: str):

        self._maybe_reconnect()

        if not self.enabled:
            return None

        try:

            data = self.client.get(key)

            if data is None:
                return None

            try:
                return json.loads(data)

            except Exception:

                logger.warning(
                    "Corrupted cache entry removed."
                )

                self.client.delete(key)
                return None

        except Exception:

            logger.exception("Redis GET failure.")

            self.enabled = False
            self._disabled_until = time.time() + self.RETRY_SECONDS

            return None

    # ----------------------------------

    def set(self, key: str, value: dict, ttl=900):

        self._maybe_reconnect()

        if not self.enabled:
            return

        try:

            jitter = int(ttl * 0.1)
            final_ttl = ttl + random.randint(-jitter, jitter)

            payload = json.dumps(value, default=str)

            self.client.setex(
                key,
                max(60, final_ttl),
                payload
            )

        except Exception:

            logger.exception("Redis SET failure.")

            self.enabled = False
            self._disabled_until = time.time() + self.RETRY_SECONDS
