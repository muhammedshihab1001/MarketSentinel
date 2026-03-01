import redis
import json
import hashlib
import logging
import os
import random
import time
import zlib
import numpy as np

from core.schema.feature_schema import get_schema_signature
from core.market.universe import MarketUniverse
from app.inference.model_loader import ModelLoader

logger = logging.getLogger("marketsentinel.cache")


class RedisCache:

    _client = None
    _pool = None

    BASE_RETRY = 15
    MAX_RETRY = 120

    MAX_TTL = 900
    MIN_TTL = 30

    MAX_PAYLOAD_BYTES = 256_000
    CACHE_NAMESPACE_VERSION = "v6"  # bumped

    ###################################################

    def __init__(self):

        self.enabled = False
        self._disabled_until = 0
        self._retry_delay = self.BASE_RETRY

        self.schema_sig = get_schema_signature()
        self.universe_fp = MarketUniverse.fingerprint()

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

        except Exception:

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
            return loader.xgb_version
        except Exception:
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
            f"portfolio:"
            f"{model_fp}:"
            f"{self.schema_sig}:"
            f"{self.universe_fp}:"
            f"{fingerprint}"
        )

    ###################################################
    # SNAPSHOT VALIDATION
    ###################################################

    def _validate_snapshot(self, value):

        if not isinstance(value, dict):
            raise RuntimeError("Cache payload must be snapshot dict.")

        required_top = {
            "snapshot_date",
            "universe_size",
            "top_5",
            "signals"
        }

        if not required_top.issubset(value.keys()):
            raise RuntimeError("Snapshot missing required fields.")

        if not isinstance(value["signals"], list):
            raise RuntimeError("Signals must be list.")

        for row in value["signals"]:

            if not isinstance(row, dict):
                raise RuntimeError("Invalid signal row.")

            required_row = {
                "ticker",
                "score",
                "weight"
            }

            if not required_row.issubset(row.keys()):
                raise RuntimeError("Signal row missing fields.")

            if not isinstance(row["ticker"], str):
                raise RuntimeError("Invalid ticker type.")

            for field in ["score", "weight"]:
                val = row[field]
                if not isinstance(val, (int, float)):
                    raise RuntimeError("Invalid numeric field.")
                if not np.isfinite(val):
                    raise RuntimeError("Non-finite numeric value.")

            if abs(row["score"]) > 20:
                raise RuntimeError("Unrealistic score detected.")

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

            decompressed = zlib.decompress(data)

            if len(decompressed) > self.MAX_PAYLOAD_BYTES:
                logger.warning("Oversized cache entry removed.")
                self.client.delete(key)
                return None

            obj = json.loads(decompressed)

            self._validate_snapshot(obj)

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

    def set(self, key: str, value, ttl=None):

        self._maybe_reconnect()

        if not self.enabled:
            return

        try:

            self._validate_snapshot(value)

            ttl = ttl or int(
                os.getenv("CACHE_TTL_SECONDS", "180")
            )

            ttl = max(self.MIN_TTL, min(ttl, self.MAX_TTL))

            # volatility-sensitive TTL (shorter if high risk)
            risk_mult = value.get("risk_multiplier", 1.0)
            if risk_mult < 0.8:
                ttl = max(self.MIN_TTL, int(ttl * 0.5))

            jitter = int(ttl * 0.15)
            final_ttl = ttl + random.randint(-jitter, jitter)
            final_ttl = max(self.MIN_TTL, min(final_ttl, self.MAX_TTL))

            serialized = self._canonical_json(value).encode()

            if len(serialized) > self.MAX_PAYLOAD_BYTES:
                raise RuntimeError("Payload exceeds safe cache size.")

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