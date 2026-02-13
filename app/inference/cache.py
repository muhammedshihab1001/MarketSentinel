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

    MAX_TTL = 900
    MIN_TTL = 30

    CACHE_NAMESPACE_VERSION = "v2"

    ###################################################

    def __init__(self):

        self.enabled = False
        self._disabled_until = 0
        self._retry_delay = self.BASE_RETRY

        self.schema_sig = get_schema_signature()

        self._connect()

    ###################################################
    # CONNECTION
    ###################################################

    def _connect(self):

        try:

            host = os.getenv("REDIS_HOST", "redis")
            port = int(os.getenv("REDIS_PORT", "6379"))

            if RedisCache._pool is None:

                RedisCache._pool = redis.ConnectionPool(
                    host=host,
                    port=port,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                    health_check_interval=30,
                    max_connections=20,
                    decode_responses=False
                )

            if RedisCache._client is None:

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

            jitter = random.randint(0, 5)
            self._disabled_until = time.time() + self._retry_delay + jitter

            self._retry_delay = min(
                self._retry_delay * 2,
                self.MAX_RETRY
            )

            logger.warning(
                "Redis unavailable. Retry in %ss",
                self._retry_delay
            )

    ###################################################

    def _maybe_reconnect(self):

        if self.enabled:
            return

        if time.time() < self._disabled_until:
            return

        logger.info("Attempting Redis reconnect...")
        self._connect()

    ###################################################
    # MODEL FINGERPRINT
    ###################################################

    def _model_fingerprint(self):

        try:
            loader = ModelLoader()
            container = loader._xgb_container

            if container:
                return container.version

        except Exception:
            pass

        return "unknown"

    ###################################################
    # CANONICAL JSON
    ###################################################

    def _canonical_json(self, payload):

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str
        )

    ###################################################
    # CACHE KEY
    ###################################################

    def build_key(self, payload: dict) -> str:

        raw = self._canonical_json(payload)

        fingerprint = hashlib.sha256(raw.encode()).hexdigest()

        model_fp = self._model_fingerprint()

        return (
            f"{self.CACHE_NAMESPACE_VERSION}:"
            f"prediction:"
            f"{model_fp}:"
            f"{self.schema_sig}:"
            f"{fingerprint}"
        )

    ###################################################
    # PAYLOAD VALIDATION
    ###################################################

    def _validate_payload(self, value):

        if not isinstance(value, dict):
            raise RuntimeError("Cache payload must be dict.")

        if not value:
            raise RuntimeError("Refusing to cache empty payload.")

        required = {"ticker", "signal_today", "confidence"}

        if not required.issubset(value.keys()):
            raise RuntimeError("Cache payload missing required fields.")

    ###################################################
    # GET
    ###################################################

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
                obj = json.loads(decompressed)

                self._validate_payload(obj)

                return obj

            except Exception:

                logger.warning("Corrupted cache entry removed.")
                self.client.delete(key)
                return None

        except Exception:

            logger.exception("Redis GET failure.")

            self.enabled = False
            self._retry_delay = min(
                self._retry_delay * 2,
                self.MAX_RETRY
            )
            self._disabled_until = time.time() + self._retry_delay

            return None

    ###################################################
    # SET
    ###################################################

    def set(self, key: str, value: dict, ttl=None):

        self._maybe_reconnect()

        if not self.enabled:
            return

        try:

            self._validate_payload(value)

            ttl = ttl or int(
                os.getenv("CACHE_TTL_SECONDS", "180")
            )

            ttl = max(self.MIN_TTL, min(ttl, self.MAX_TTL))

            jitter = int(ttl * 0.15)
            final_ttl = ttl + random.randint(-jitter, jitter)
            final_ttl = max(self.MIN_TTL, min(final_ttl, self.MAX_TTL))

            serialized = self._canonical_json(value).encode()

            payload = zlib.compress(serialized)

            # round-trip validation
            test = json.loads(zlib.decompress(payload))

            self._validate_payload(test)

            self.client.setex(
                key,
                final_ttl,
                payload
            )

        except Exception:

            logger.exception("Redis SET failure.")

            self.enabled = False
            self._retry_delay = min(
                self._retry_delay * 2,
                self.MAX_RETRY
            )
            self._disabled_until = time.time() + self._retry_delay
