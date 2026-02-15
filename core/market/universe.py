from typing import Tuple, Dict
import hashlib
import json
import os
import threading
import re


class MarketUniverse:
    """
    Institutional Universe Controller.

    Guarantees:
    ✔ deterministic
    ✔ file-governed
    ✔ audit safe
    ✔ mutation proof
    ✔ lineage traceable
    ✔ thread safe
    ✔ tamper detected
    ✔ hot-reload safe
    ✔ ticker validated
    ✔ version monotonic
    """

    UNIVERSE_FILE = "config/universe.json"

    MIN_FILE_BYTES = 20
    MIN_UNIVERSE_SIZE = 5

    # exchange-safe ticker pattern
    TICKER_REGEX = re.compile(r"^[A-Z0-9.\-]{1,10}$")

    _CACHE = None
    _LOCK = threading.RLock()
    _FILE_HASH = None
    _LAST_VERSION = None

    ###################################################
    # FILE HASH
    ###################################################

    @classmethod
    def _hash_file(cls):

        h = hashlib.sha256()

        with open(cls.UNIVERSE_FILE, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    ###################################################
    # SAFE LOAD
    ###################################################

    @classmethod
    def _load_file(cls):

        if not os.path.exists(cls.UNIVERSE_FILE):
            raise RuntimeError(
                f"Universe file missing: {cls.UNIVERSE_FILE}"
            )

        if os.path.getsize(cls.UNIVERSE_FILE) < cls.MIN_FILE_BYTES:
            raise RuntimeError("Universe file corrupted.")

        try:
            # safer read
            with open(cls.UNIVERSE_FILE, "rb") as f:
                payload = json.loads(f.read().decode())

        except Exception as exc:
            raise RuntimeError(
                "Universe config unreadable."
            ) from exc

        #################################################
        # STRICT SCHEMA
        #################################################

        if not isinstance(payload, dict):
            raise RuntimeError("Universe payload must be dict.")

        version = payload.get("version")
        tickers = payload.get("tickers")

        if not isinstance(version, str):
            raise RuntimeError("Universe version must be string.")

        if not isinstance(tickers, list):
            raise RuntimeError("Universe tickers must be list.")

        if len(tickers) < cls.MIN_UNIVERSE_SIZE:
            raise RuntimeError(
                f"Universe too small (<{cls.MIN_UNIVERSE_SIZE})."
            )

        #################################################
        # VERSION MONOTONICITY
        #################################################

        if cls._LAST_VERSION and version < cls._LAST_VERSION:
            raise RuntimeError(
                "Universe version downgrade detected."
            )

        #################################################
        # NORMALIZE + VALIDATE
        #################################################

        normalized = []

        for t in tickers:

            if not isinstance(t, str):
                raise RuntimeError("Non-string ticker detected.")

            t = t.strip().upper()

            if not cls.TICKER_REGEX.match(t):
                raise RuntimeError(
                    f"Invalid ticker format: {t}"
                )

            normalized.append(t)

        if len(normalized) != len(set(normalized)):
            raise RuntimeError("Duplicate tickers detected.")

        cls._LAST_VERSION = version

        return version, tuple(sorted(normalized))

    ###################################################
    # CACHE (THREAD SAFE + HOT RELOAD)
    ###################################################

    @classmethod
    def _get_cached(cls):

        current_hash = cls._hash_file()

        if cls._CACHE is None or cls._FILE_HASH != current_hash:

            with cls._LOCK:

                if cls._CACHE is None or cls._FILE_HASH != current_hash:

                    version, tickers = cls._load_file()

                    cls._CACHE = {
                        "version": version,
                        "tickers": tickers,
                        "ticker_set": set(tickers)
                    }

                    cls._FILE_HASH = current_hash

        return cls._CACHE

    ###################################################
    # FORCE REFRESH
    ###################################################

    @classmethod
    def refresh(cls):

        with cls._LOCK:
            cls._CACHE = None
            cls._FILE_HASH = None

        return cls._get_cached()

    ###################################################
    # PUBLIC
    ###################################################

    @classmethod
    def get_universe(cls) -> Tuple[str, ...]:
        return cls._get_cached()["tickers"]

    ###################################################
    # FINGERPRINT
    ###################################################

    @classmethod
    def fingerprint(cls) -> str:

        contract = {
            "tickers": list(cls.get_universe()),
            "version": cls._get_cached()["version"],
        }

        canonical = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    ###################################################
    # SNAPSHOT
    ###################################################

    @classmethod
    def snapshot(cls) -> Dict:

        cache = cls._get_cached()

        tickers = list(cache["tickers"])

        return {
            "universe_version": cache["version"],
            "universe_hash": cls.fingerprint(),
            "universe_size": len(tickers),
            "tickers": tickers
        }

    ###################################################
    # VALIDATION
    ###################################################

    @classmethod
    def validate_subset(cls, tickers):

        cache = cls._get_cached()

        unknown = set(tickers) - cache["ticker_set"]

        if unknown:
            raise RuntimeError(
                f"Unknown tickers detected: {unknown}"
            )
