import redis
import json
import hashlib
import logging
import os
import random
import time
import zlib


logger = logging.getLogger("marketsentinel.cache")


class RedisCache:

    _client = None
    _pool = None

    BASE_RETRY = 15
    MAX_RETRY = 120

    def __init__(self):

        self.enabled = False
        self._disabled_until = 0
        self._retry_delay = self.BASE_RETRY

        self._connect()

    def _connect(self):

        try:

            host = os.getenv("REDIS_HOST", "redis")
            port = int(os.getenv("REDIS_PORT", "6379"))

            RedisCache._pool = redis.ConnectionPool(
                host=host,
                port=port,
                socket_timeout=2,
                socket_connect_timeout=2,
                max_connections=10,
                decode_responses=False
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

    def _maybe_reconnect(self):

        if self.enabled:
            return

        if time.time() < self._disabled_until:
            return

        logger.info("Attempting Redis reconnect...")
        self._connect()

    def _model_fingerprint(self):

        path = os.getenv(
            "XGB_MODEL_PATH",
            "artifacts/xgboost/model.pkl"
        )

        try:
            mtime = os.path.getmtime(path)
            return str(int(mtime))
        except Exception:
            return "unknown"

    def build_key(self, payload: dict) -> str:

        raw = json.dumps(payload, sort_keys=True, default=str)

        fingerprint = hashlib.sha256(raw.encode()).hexdigest()

        model_fp = self._model_fingerprint()

        return f"prediction:{model_fp}:{fingerprint}"

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
