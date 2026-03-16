# =========================================================
# REDIS CACHE v9
# Stable | Drift-Safe | CV-Optimized
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

    BASE_RETRY = 15
    MAX_RETRY = 120

    MAX_TTL = 900
    MIN_TTL = 30

    MAX_PAYLOAD_BYTES = 256_000
    CACHE_NAMESPACE_VERSION = "v9"

    ###################################################

    def __init__(self):

        self.enabled = False
        self._disabled_until = 0
        self._retry_delay = self.BASE_RETRY

        self.schema_sig = get_schema_signature()
        self.universe_fp = MarketUniverse.fingerprint()

        try:
            loader = ModelLoader()
            self.model_fp = loader.xgb_version
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
                    socket_keepalive=True,
                    health_check_interval=30,
                    max_connections=16,
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
            f"portfolio:"
            f"{self.model_fp}:"
            f"{self.schema_sig}:"
            f"{self.universe_fp}:"
            f"{fingerprint}"
        )

    ###################################################
    # PAYLOAD VALIDATION
    ###################################################

    def _validate_payload(self, payload):

        if not isinstance(payload, list):
            raise RuntimeError("Payload must be a list.")

        if len(payload) == 0:
            raise RuntimeError("Payload cannot be empty.")

        for row in payload:

            if not isinstance(row, dict):
                raise RuntimeError("Invalid payload row.")

            ticker = row.get("ticker")
            weight = row.get("weight", 0)

            if not isinstance(ticker, str):
                raise RuntimeError("Invalid ticker.")

            if not isinstance(weight, (int, float)):
                raise RuntimeError("Invalid weight.")

            if not np.isfinite(weight):
                raise RuntimeError("Non-finite weight.")

            if abs(weight) > 1:
                raise RuntimeError("Unrealistic weight.")

    ###################################################
    # SNAPSHOT VALIDATION
    ###################################################

    def _validate_snapshot(self, value):

        if not isinstance(value, dict):
            raise RuntimeError("Cache payload must be dict.")

        if "signals" not in value:
            raise RuntimeError("Snapshot missing signals.")

        if not isinstance(value["signals"], list):
            raise RuntimeError("Signals must be list.")

        self._validate_payload(value["signals"])

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
                CACHE_MISSES.inc()
                return None

            if len(data) > self.MAX_PAYLOAD_BYTES:
                logger.warning("Raw cache payload too large. Deleting.")
                self.client.delete(key)
                CACHE_MISSES.inc()
                return None

            try:
                decompressed = zlib.decompress(data)
            except Exception:
                logger.warning("Corrupted compressed cache entry removed.")
                self.client.delete(key)
                CACHE_MISSES.inc()
                return None

            if len(decompressed) > self.MAX_PAYLOAD_BYTES:
                logger.warning("Oversized decompressed cache entry removed.")
                self.client.delete(key)
                CACHE_MISSES.inc()
                return None

            try:
                obj = json.loads(decompressed)
            except Exception:
                logger.warning("Invalid JSON cache entry removed.")
                self.client.delete(key)
                CACHE_MISSES.inc()
                return None

            self._validate_snapshot(obj)

            CACHE_HITS.inc()

            return obj

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

    def set(
        self,
        key: str,
        value,
        ttl: Optional[int] = None,
        ex: Optional[int] = None
    ):

        self._maybe_reconnect()

        if not self.enabled:
            return

        try:

            self._validate_snapshot(value)

            ttl_value = ex if ex is not None else ttl

            if ttl_value is None:
                ttl_value = int(
                    os.getenv("CACHE_TTL_SECONDS", "180")
                )

            ttl_value = max(self.MIN_TTL, min(ttl_value, self.MAX_TTL))

            jitter = int(ttl_value * 0.15)

            final_ttl = ttl_value + random.randint(-jitter, jitter)

            final_ttl = max(
                self.MIN_TTL,
                min(final_ttl, self.MAX_TTL)
            )

            serialized = self._canonical_json(value).encode()

            if len(serialized) > self.MAX_PAYLOAD_BYTES:
                logger.warning("Cache payload too large. Skipping cache.")
                return

            payload = zlib.compress(serialized)

            self.client.set(
                key,
                payload,
                ex=final_ttl
            )

        except Exception:

            logger.exception("Redis SET failure.")

            self.enabled = False

            self._retry_delay = min(
                self._retry_delay * 2,
                self.MAX_RETRY
            )

            self._disabled_until = time.time() + self._retry_delay