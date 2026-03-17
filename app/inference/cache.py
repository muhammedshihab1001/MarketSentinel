# =========================================================
# REDIS CACHE v10
# Stable | Drift-Safe | CV-Optimized | Memory Fallback
# =========================================================

import redis
import json
import hashlib
import logging
import os
import random
import time
import zlib
import numpy as np
from typing import Optional

from core.schema.feature_schema import get_schema_signature
from core.market.universe import MarketUniverse
from app.inference.model_loader import ModelLoader

from app.monitoring.metrics import CACHE_HITS, CACHE_MISSES

logger = logging.getLogger("marketsentinel.cache")


class RedisCache:

    _client = None
    _pool = None

    # 🔥 NEW: in-memory fallback cache
    _memory_cache = {}

    BASE_RETRY = 15
    MAX_RETRY = 180

    MAX_TTL = 900
    MIN_TTL = 30

    MAX_PAYLOAD_BYTES = 256_000
    CACHE_NAMESPACE_VERSION = "v10"

    ###################################################

    def __init__(self):

        self.enabled = False
        self._disabled_until = 0
        self._retry_delay = self.BASE_RETRY

        self.schema_sig = get_schema_signature()
        self.universe_fp = MarketUniverse.fingerprint()

        try:
            loader = ModelLoader()
            container = loader._xgb_container
            self.model_fp = container.version if container else "unknown"
        except Exception:
            self.model_fp = "unknown"

        self._connect()

    ###################################################
    # CONNECTION
    ###################################################

    def _connect(self):

        try:

            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))

            if RedisCache._pool is None:

                RedisCache._pool = redis.ConnectionPool(
                    host=host,
                    port=port,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                    max_connections=8,
                    retry_on_timeout=True,
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

        except Exception as exc:

            self.enabled = False

            jitter = random.randint(0, 5)

            self._disabled_until = (
                time.time() +
                self._retry_delay +
                jitter
            )

            self._retry_delay = min(
                self._retry_delay * 2,
                self.MAX_RETRY
            )

            logger.warning(
                "Redis unavailable (%s). Retry in %ss",
                exc,
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

        return (
            f"{self.CACHE_NAMESPACE_VERSION}:"
            f"{self.model_fp}:"
            f"{self.schema_sig}:"
            f"{self.universe_fp}:"
            f"{fingerprint}"
        )

    ###################################################
    # MEMORY CACHE HELPERS
    ###################################################

    def _memory_get(self, key):

        entry = self._memory_cache.get(key)

        if not entry:
            return None

        value, expiry = entry

        if time.time() > expiry:
            self._memory_cache.pop(key, None)
            return None

        return value

    def _memory_set(self, key, value, ttl):

        expiry = time.time() + ttl

        # limit memory size (simple LRU-like cleanup)
        if len(self._memory_cache) > 500:
            self._memory_cache.pop(next(iter(self._memory_cache)))

        self._memory_cache[key] = (value, expiry)

    ###################################################
    # VALIDATION (RELAXED)
    ###################################################

    def _validate_snapshot(self, value):

        if not isinstance(value, dict):
            return False

        if "signals" not in value:
            return False

        if not isinstance(value["signals"], list):
            return False

        return True

    ###################################################
    # GET
    ###################################################

    def get(self, key: str):

        self._maybe_reconnect()

        # 🔥 1. Try memory cache first
        mem = self._memory_get(key)
        if mem:
            CACHE_HITS.inc()
            return mem

        # 🔥 2. Try Redis
        if not self.enabled:
            CACHE_MISSES.inc()
            return None

        try:

            data = self.client.get(key)

            if not data:
                CACHE_MISSES.inc()
                return None

            decompressed = zlib.decompress(data)
            obj = json.loads(decompressed)

            if not self._validate_snapshot(obj):
                self.client.delete(key)
                CACHE_MISSES.inc()
                return None

            # 🔥 store in memory cache
            self._memory_set(key, obj, 60)

            CACHE_HITS.inc()
            return obj

        except Exception:

            logger.exception("Redis GET failure.")

            self.enabled = False
            self._disabled_until = time.time() + self._retry_delay

            return None

    ###################################################
    # SET
    ###################################################

    def set(
        self,
        key: str,
        value,
        ttl: Optional[int] = None,
        ex: Optional[int] = None
    ):

        self._maybe_reconnect()

        ttl_value = ex if ex is not None else ttl

        if ttl_value is None:
            ttl_value = int(os.getenv("CACHE_TTL_SECONDS", "180"))

        ttl_value = max(self.MIN_TTL, min(ttl_value, self.MAX_TTL))

        # 🔥 Always store in memory
        self._memory_set(key, value, ttl_value)

        if not self.enabled:
            return

        try:

            serialized = self._canonical_json(value).encode()

            if len(serialized) > self.MAX_PAYLOAD_BYTES:
                return

            payload = zlib.compress(serialized)

            self.client.set(key, payload, ex=ttl_value)

        except Exception:

            logger.exception("Redis SET failure.")

            self.enabled = False
            self._disabled_until = time.time() + self._retry_delay