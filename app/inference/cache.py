import redis
import json
import hashlib
import logging
import os
import random
import time
import zlib

from core.schema.feature_schema import get_schema_signature
from app.inference.model_loader import ModelLoader


logger = logging.getLogger("marketsentinel.cache")


class RedisCache:

    _client = None
    _pool = None

    BASE_RETRY = 15
    MAX_RETRY = 120

    LOCK_TIMEOUT = 10  # seconds

    def __init__(self):

        self.enabled = False
        self._disabled_until = 0
        self._retry_delay = self.BASE_RETRY

        self._connect()

    # ------------------------------------------------

    def _connect(self):

        try:

            host = os.getenv("REDIS_HOST", "redis")
            port = int(os.getenv("REDIS_PORT", "6379"))

            RedisCache._pool = redis.ConnectionPool(
                host=host,
                port=port,
                socket_timeout=2,
                socket_connect_timeout=2,
                max_connections=50,
                decode_responses=False  # required for compression
            )

            RedisCache._client = redis.Redis(
                connection_pool=RedisCache._pool
            )

            RedisCache._client.ping()

            self.client = RedisCache._client
            self.enabled = True
            self._retry_delay = self.BASE_RETRY

            logger.info("Redis connected.")

        except Exception:

            self.enabled = False
            self._disabled_until = time.time() + self._retry_delay

            self._retry_delay = min(
                self._retry_delay * 2,
                self.MAX_RETRY
            )

            logger.warning(
                f"Redis unavailable. Retry in {self._retry_delay}s"
            )

    # ------------------------------------------------

    def _maybe_reconnect(self):

        if self.enabled:
            return

        if time.time() < self._disabled_until:
            return

        logger.info("Attempting Redis reconnect...")
        self._connect()

    # ------------------------------------------------
    #  MODEL VERSION SAFE KEY
    # ------------------------------------------------

    def build_key(self, payload: dict) -> str:

        raw = json.dumps(payload, sort_keys=True, default=str)

        schema = get_schema_signature()

        model_version = ModelLoader().get_production_version(
            "xgboost"
        )

        fingerprint = hashlib.sha256(raw.encode()).hexdigest()

        return f"prediction:{schema}:{model_version}:{fingerprint}"

    # ------------------------------------------------
    # DISTRIBUTED LOCK
    # ------------------------------------------------

    def acquire_lock(self, key: str):

        if not self.enabled:
            return None

        lock_key = f"lock:{key}"

        try:

            lock = self.client.lock(
                lock_key,
                timeout=self.LOCK_TIMEOUT,
                blocking_timeout=3
            )

            acquired = lock.acquire()

            if acquired:
                return lock

            return None

        except Exception:

            logger.exception("Redis lock failure.")
            return None

    # ------------------------------------------------

    def get(self, key: str):

        self._maybe_reconnect()

        if not self.enabled:
            return None

        try:

            data = self.client.get(key)

            if not data:
                return None

            try:

                decompressed = zlib.decompress(data)

                return json.loads(decompressed)

            except Exception:

                logger.warning("Corrupted cache entry removed.")
                self.client.delete(key)
                return None

        except Exception:

            logger.exception("Redis GET failure.")

            self.enabled = False
            self._disabled_until = time.time() + self._retry_delay

            return None

    # ------------------------------------------------

    def set(self, key: str, value: dict, ttl=None):

        self._maybe_reconnect()

        if not self.enabled:
            return

        try:

            ttl = ttl or int(
                os.getenv("CACHE_TTL_SECONDS", "180")
            )

            jitter = int(ttl * 0.15)
            final_ttl = ttl + random.randint(-jitter, jitter)

            payload = zlib.compress(
                json.dumps(value).encode()
            )

            self.client.setex(
                key,
                max(30, final_ttl),
                payload
            )

        except Exception:

            logger.exception("Redis SET failure.")

            self.enabled = False
            self._disabled_until = time.time() + self._retry_delay
