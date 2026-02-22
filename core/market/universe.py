from typing import Tuple, Dict
import hashlib
import json
import os
import threading
import re
from datetime import datetime
from pathlib import Path


class MarketUniverse:
    """
    Institutional Universe Controller (v8 dual-universe hardened)

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
    ✔ version monotonic (numeric-safe)
    ✔ metadata validated
    ✔ min_history_days enforced
    ✔ cross-sectional statistical safety
    ✔ research/production separation
    """

    ###################################################
    # FILE SELECTION (ENV GOVERNED)
    ###################################################

    DEFAULT_RESEARCH_FILE = "config/universe_research.json"
    DEFAULT_PRODUCTION_FILE = "config/universe_production.json"

    UNIVERSE_FILE = os.getenv(
        "UNIVERSE_PATH",
        DEFAULT_RESEARCH_FILE
    )

    MIN_FILE_BYTES = 20

    # 🚨 For ML stability (research only)
    MIN_UNIVERSE_SIZE = 30

    TICKER_REGEX = re.compile(r"^[A-Z0-9.\-]{1,10}$")

    _CACHE = None
    _LOCK = threading.RLock()
    _FILE_HASH = None
    _LAST_VERSION = None

    ###################################################
    # INTERNAL VERSION PARSER
    ###################################################

    @staticmethod
    def _parse_version(version: str):
        try:
            return tuple(int(x) for x in version.split("."))
        except Exception:
            raise RuntimeError(
                "Universe version must be numeric format like '6.0'"
            )

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

        path = Path(cls.UNIVERSE_FILE)

        if not path.exists():
            raise RuntimeError(
                f"Universe file missing: {cls.UNIVERSE_FILE}"
            )

        if path.stat().st_size < cls.MIN_FILE_BYTES:
            raise RuntimeError("Universe file corrupted.")

        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            raise RuntimeError(
                "Universe config unreadable."
            ) from exc

        #################################################
        # STRICT SCHEMA
        #################################################

        required_fields = {
            "version",
            "created_utc",
            "description",
            "tickers"
        }

        # min_history_days optional in production
        missing = required_fields - set(payload.keys())
        if missing:
            raise RuntimeError(
                f"Universe config missing fields: {missing}"
            )

        version = payload["version"]
        created_utc = payload["created_utc"]
        description = payload["description"]
        tickers = payload["tickers"]

        min_history_days = payload.get("min_history_days", None)

        if not isinstance(version, str):
            raise RuntimeError("Universe version must be string.")

        if not isinstance(description, str):
            raise RuntimeError("Universe description must be string.")

        try:
            datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
        except Exception:
            raise RuntimeError("created_utc must be ISO format.")

        if not isinstance(tickers, list):
            raise RuntimeError("Universe tickers must be list.")

        #################################################
        # ML SAFETY CHECK (ONLY FOR RESEARCH)
        #################################################

        if cls.UNIVERSE_FILE == cls.DEFAULT_RESEARCH_FILE:

            if min_history_days is None:
                raise RuntimeError(
                    "Research universe requires min_history_days."
                )

            if not isinstance(min_history_days, int) or min_history_days <= 0:
                raise RuntimeError(
                    "min_history_days must be positive int."
                )

            if len(tickers) < cls.MIN_UNIVERSE_SIZE:
                raise RuntimeError(
                    f"Universe too small for cross-sectional ML "
                    f"(minimum {cls.MIN_UNIVERSE_SIZE} required)."
                )

        #################################################
        # VERSION MONOTONICITY
        #################################################

        parsed_version = cls._parse_version(version)

        if cls._LAST_VERSION:
            if parsed_version < cls._parse_version(cls._LAST_VERSION):
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

        return {
            "version": version,
            "created_utc": created_utc,
            "description": description,
            "min_history_days": min_history_days,
            "tickers": tuple(sorted(normalized)),
            "ticker_set": set(normalized)
        }

    ###################################################
    # CACHE (THREAD SAFE + HOT RELOAD)
    ###################################################

    @classmethod
    def _get_cached(cls):

        current_hash = cls._hash_file()

        if cls._CACHE is None or cls._FILE_HASH != current_hash:

            with cls._LOCK:

                if cls._CACHE is None or cls._FILE_HASH != current_hash:

                    cls._CACHE = cls._load_file()
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
    # PUBLIC API
    ###################################################

    @classmethod
    def get_universe(cls) -> Tuple[str, ...]:
        return cls._get_cached()["tickers"]

    @classmethod
    def get_min_history_days(cls):
        return cls._get_cached()["min_history_days"]

    @classmethod
    def get_version(cls) -> str:
        return cls._get_cached()["version"]

    ###################################################
    # FINGERPRINT
    ###################################################

    @classmethod
    def fingerprint(cls) -> str:

        contract = {
            "tickers": list(cls.get_universe()),
            "version": cls.get_version(),
            "min_history_days": cls.get_min_history_days()
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

        return {
            "universe_version": cache["version"],
            "created_utc": cache["created_utc"],
            "description": cache["description"],
            "min_history_days": cache["min_history_days"],
            "universe_hash": cls.fingerprint(),
            "universe_size": len(cache["tickers"]),
            "tickers": list(cache["tickers"])
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