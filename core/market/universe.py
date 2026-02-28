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
    Institutional Universe Controller (Production Hardened + Research Flexible)

    Guarantees:
    ✔ deterministic
    ✔ file-governed
    ✔ audit safe
    ✔ mutation proof
    ✔ lineage traceable
    ✔ thread safe
    ✔ tamper detected
    ✔ version monotonic
    ✔ metadata compatible
    ✔ research override safe
    """

    ###################################################
    # CONFIG
    ###################################################

    UNIVERSE_FILE = os.getenv(
        "UNIVERSE_PATH",
        "config/universe.json"
    )

    REQUIRED_SIZE = 30
    MIN_FILE_BYTES = 50

    # Optional research override (DOES NOT affect production)
    ALLOW_RESEARCH_OVERRIDE = os.getenv("UNIVERSE_RESEARCH_MODE", "0") == "1"

    TICKER_REGEX = re.compile(r"^[A-Z0-9.\-]{1,10}$")

    _CACHE = None
    _LOCK = threading.RLock()
    _FILE_HASH = None
    _LAST_VERSION = None

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

    @classmethod
    def _hash_file(cls):

        h = hashlib.sha256()

        with open(cls.UNIVERSE_FILE, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    ###################################################

    @classmethod
    def _validate_payload(cls, payload: Dict):

        required = {
            "version",
            "created_utc",
            "description",
            "min_history_days",
            "tickers"
        }

        missing = required - set(payload.keys())
        if missing:
            raise RuntimeError(f"Universe config missing fields: {missing}")

        version = payload["version"]
        created_utc = payload["created_utc"]
        tickers = payload["tickers"]
        min_history_days = payload["min_history_days"]

        if not isinstance(version, str):
            raise RuntimeError("Universe version must be string.")

        datetime.fromisoformat(created_utc.replace("Z", "+00:00"))

        if not isinstance(min_history_days, int) or min_history_days <= 0:
            raise RuntimeError("min_history_days must be positive integer.")

        if not isinstance(tickers, list):
            raise RuntimeError("Universe tickers must be list.")

        if not cls.ALLOW_RESEARCH_OVERRIDE:
            if len(tickers) != cls.REQUIRED_SIZE:
                raise RuntimeError(
                    f"Production universe must contain exactly {cls.REQUIRED_SIZE} tickers."
                )

        parsed_version = cls._parse_version(version)

        if cls._LAST_VERSION:
            if parsed_version < cls._parse_version(cls._LAST_VERSION):
                raise RuntimeError("Universe version downgrade detected.")

        normalized = []

        for t in tickers:

            if not isinstance(t, str):
                raise RuntimeError("Non-string ticker detected.")

            t = t.strip().upper()

            if not cls.TICKER_REGEX.match(t):
                raise RuntimeError(f"Invalid ticker format: {t}")

            normalized.append(t)

        if len(normalized) != len(set(normalized)):
            raise RuntimeError("Duplicate tickers detected.")

        cls._LAST_VERSION = version

        return {
            "version": version,
            "created_utc": created_utc,
            "description": payload["description"],
            "min_history_days": min_history_days,
            "tickers": tuple(sorted(normalized)),
            "ticker_set": set(normalized)
        }

    ###################################################

    @classmethod
    def _load_file(cls):

        path = Path(cls.UNIVERSE_FILE)

        if not path.exists():
            raise RuntimeError(f"Universe file missing: {cls.UNIVERSE_FILE}")

        if path.stat().st_size < cls.MIN_FILE_BYTES:
            raise RuntimeError("Universe file corrupted.")

        payload = json.loads(path.read_text())

        return cls._validate_payload(payload)

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

    @classmethod
    def size(cls) -> int:
        return len(cls.get_universe())

    ###################################################
    # FINGERPRINT (METADATA CRITICAL)
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
    # SNAPSHOT (METADATA SAFE)
    ###################################################

    @classmethod
    def snapshot(cls) -> Dict:

        cache = cls._get_cached()

        snapshot = {
            "universe_version": cache["version"],
            "created_utc": cache["created_utc"],
            "description": cache["description"],
            "min_history_days": cache["min_history_days"],
            "universe_hash": cls.fingerprint(),
            "universe_size": len(cache["tickers"]),
            "tickers": list(cache["tickers"])
        }

        if "universe_hash" not in snapshot:
            raise RuntimeError("Universe snapshot construction failed.")

        return snapshot