# =========================================================
# MARKET UNIVERSE CONTROLLER v2.3
# Hybrid Multi-Agent Compatible | CV-Optimized
# =========================================================

from typing import Tuple, Dict, Optional, List
import hashlib
import json
import os
import threading
import re
from datetime import datetime
from pathlib import Path


class MarketUniverse:
    """
    Deterministic Universe Controller

    Guarantees:
    ✔ deterministic
    ✔ file-governed
    ✔ thread safe
    ✔ tamper detected
    ✔ hybrid compatible
    ✔ CV-polished architecture

    Softened for:
    - Personal project usage
    - yfinance data noise
    - Non-critical production use
    """

    CONTROLLER_VERSION = "2.3"

    PRODUCTION_FILE = os.getenv(
        "UNIVERSE_PATH",
        "config/universe.json"
    )

    RESEARCH_FILE = os.getenv(
        "UNIVERSE_RESEARCH_PATH",
        "config/universe_research.json"
    )

    REQUIRED_SIZE = int(os.getenv("UNIVERSE_MIN_SIZE", "20"))

    MAX_UNIVERSE_SIZE = int(os.getenv("UNIVERSE_MAX_SIZE", "200"))

    TRAINING_SOFT_LIMIT = int(os.getenv("UNIVERSE_TRAINING_LIMIT", "120"))

    # NEW: runtime safety cap
    RUNTIME_FETCH_LIMIT = int(os.getenv("UNIVERSE_RUNTIME_LIMIT", "150"))

    MIN_FILE_BYTES = 50

    ALLOW_RESEARCH_OVERRIDE = os.getenv(
        "UNIVERSE_RESEARCH_MODE",
        "0"
    ) == "1"

    TICKER_REGEX = re.compile(r"^[A-Z0-9.\-]{1,12}$")

    _CACHE: Optional[Dict] = None

    _LOCK = threading.RLock()

    _FILE_HASH: Optional[str] = None

    _LAST_VERSION: Optional[str] = None

    _LAST_LOADED_AT: Optional[str] = None

    # -----------------------------------------------------
    # VERSION PARSER
    # -----------------------------------------------------

    @staticmethod
    def _parse_version(version: str):

        try:
            return tuple(int(x) for x in version.split("."))

        except Exception:
            return (0,)

    # -----------------------------------------------------
    # FILE RESOLUTION
    # -----------------------------------------------------

    @classmethod
    def _active_file(cls):

        if cls.ALLOW_RESEARCH_OVERRIDE and Path(cls.RESEARCH_FILE).exists():
            return cls.RESEARCH_FILE

        return cls.PRODUCTION_FILE

    # -----------------------------------------------------
    # HASH FILE
    # -----------------------------------------------------

    @classmethod
    def _hash_file(cls):

        path = Path(cls._active_file())

        h = hashlib.sha256()

        with open(path, "rb") as f:

            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    # -----------------------------------------------------
    # VALIDATE PAYLOAD
    # -----------------------------------------------------

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

        version = str(payload["version"])

        created_utc = payload["created_utc"]

        tickers = payload["tickers"]

        min_history_days = payload["min_history_days"]

        try:
            datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
        except Exception:
            raise RuntimeError("Invalid created_utc timestamp format.")

        if not isinstance(min_history_days, int) or min_history_days <= 0:
            raise RuntimeError("min_history_days must be positive integer.")

        if not isinstance(tickers, list):
            raise RuntimeError("Universe tickers must be list.")

        if not cls.ALLOW_RESEARCH_OVERRIDE:

            if len(tickers) < cls.REQUIRED_SIZE:

                raise RuntimeError(
                    f"Production universe must contain at least {cls.REQUIRED_SIZE} tickers."
                )

        if len(tickers) > cls.MAX_UNIVERSE_SIZE:

            raise RuntimeError(
                f"Universe exceeds max allowed size ({cls.MAX_UNIVERSE_SIZE})."
            )

        parsed_version = cls._parse_version(version)

        if cls._LAST_VERSION:

            if parsed_version < cls._parse_version(cls._LAST_VERSION):

                raise RuntimeError("Universe version downgrade detected.")

        normalized = []

        for t in tickers:

            if not isinstance(t, str):
                continue

            t = t.strip().upper()

            if not cls.TICKER_REGEX.match(t):
                continue

            normalized.append(t)

        if len(normalized) != len(set(normalized)):
            raise RuntimeError("Duplicate tickers detected.")

        normalized = sorted(normalized)

        cls._LAST_VERSION = version

        return {
            "version": version,
            "created_utc": created_utc,
            "description": payload["description"],
            "min_history_days": min_history_days,
            "tickers": tuple(normalized),
            "ticker_set": set(normalized),
            "file_path": cls._active_file()
        }

    # -----------------------------------------------------
    # LOAD FILE
    # -----------------------------------------------------

    @classmethod
    def _load_file(cls):

        path = Path(cls._active_file())

        if not path.exists():
            raise RuntimeError(f"Universe file missing: {path}")

        if path.stat().st_size < cls.MIN_FILE_BYTES:
            raise RuntimeError("Universe file corrupted or too small.")

        payload = json.loads(path.read_text())

        validated = cls._validate_payload(payload)

        cls._LAST_LOADED_AT = datetime.utcnow().isoformat() + "Z"

        return validated

    # -----------------------------------------------------
    # CACHE HANDLING
    # -----------------------------------------------------

    @classmethod
    def _get_cached(cls):

        current_hash = cls._hash_file()

        if cls._CACHE is None or cls._FILE_HASH != current_hash:

            with cls._LOCK:

                if cls._CACHE is None or cls._FILE_HASH != current_hash:

                    cls._CACHE = cls._load_file()

                    cls._FILE_HASH = current_hash

        return cls._CACHE

    # -----------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------

    @classmethod
    def get_universe(cls) -> Tuple[str, ...]:

        tickers = cls._get_cached()["tickers"]

        if len(tickers) > cls.RUNTIME_FETCH_LIMIT:
            tickers = tickers[: cls.RUNTIME_FETCH_LIMIT]

        if len(tickers) > cls.TRAINING_SOFT_LIMIT:
            return tickers[: cls.TRAINING_SOFT_LIMIT]

        return tickers

    @classmethod
    def contains(cls, ticker: str):

        return ticker.upper() in cls._get_cached()["ticker_set"]

    @classmethod
    def filter_valid(cls, tickers: List[str]) -> List[str]:

        universe_set = cls._get_cached()["ticker_set"]

        valid = []

        for t in tickers:

            if not isinstance(t, str):
                continue

            t = t.upper().strip()

            if t in universe_set:
                valid.append(t)

        return sorted(set(valid))

    @classmethod
    def get_min_history_days(cls):

        return cls._get_cached()["min_history_days"]

    @classmethod
    def get_version(cls):

        return cls._get_cached()["version"]

    @classmethod
    def size(cls):

        return len(cls.get_universe())

    # -----------------------------------------------------
    # FINGERPRINT
    # -----------------------------------------------------

    @classmethod
    def fingerprint(cls):

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

    # -----------------------------------------------------
    # SNAPSHOT
    # -----------------------------------------------------

    @classmethod
    def snapshot(cls):

        cache = cls._get_cached()

        return {
            "controller_version": cls.CONTROLLER_VERSION,
            "universe_version": cache["version"],
            "created_utc": cache["created_utc"],
            "description": cache["description"],
            "min_history_days": cache["min_history_days"],
            "universe_hash": cls.fingerprint(),
            "universe_size": len(cache["tickers"]),
            "file_path": cache["file_path"],
            "file_hash": cls._FILE_HASH,
            "loaded_at": cls._LAST_LOADED_AT,
            "research_mode": cls.ALLOW_RESEARCH_OVERRIDE,
            "tickers": list(cache["tickers"])
        } 