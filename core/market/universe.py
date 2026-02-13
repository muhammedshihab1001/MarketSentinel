from typing import Tuple, Dict
import hashlib
import json
import os
import threading


class MarketUniverse:
    """
    Institutional Universe Controller.

    Universe is CONFIG — not code.

    Guarantees:
    ✔ deterministic
    ✔ file-governed
    ✔ audit safe
    ✔ mutation proof
    ✔ lineage traceable
    ✔ thread safe
    """

    UNIVERSE_FILE = "config/universe.json"
    MIN_FILE_BYTES = 20

    _CACHE = None
    _LOCK = threading.Lock()

    ###################################################
    # LOAD
    ###################################################

    @classmethod
    def _load_file(cls):

        if not os.path.exists(cls.UNIVERSE_FILE):
            raise RuntimeError(
                f"Universe file missing: {cls.UNIVERSE_FILE}"
            )

        if os.path.getsize(cls.UNIVERSE_FILE) < cls.MIN_FILE_BYTES:
            raise RuntimeError(
                "Universe file corrupted or truncated."
            )

        try:
            with open(cls.UNIVERSE_FILE, "r") as f:
                payload = json.load(f)

        except Exception as exc:
            raise RuntimeError(
                "Universe config unreadable."
            ) from exc

        #################################################
        # STRICT SCHEMA
        #################################################

        if not isinstance(payload, dict):
            raise RuntimeError("Universe payload must be dict.")

        if "version" not in payload:
            raise RuntimeError("Universe missing version.")

        if "tickers" not in payload:
            raise RuntimeError("Universe missing tickers.")

        version = payload["version"]

        if not isinstance(version, str):
            raise RuntimeError(
                "Universe version must be string."
            )

        tickers = payload["tickers"]

        if not isinstance(tickers, list):
            raise RuntimeError(
                "Universe tickers must be list."
            )

        if not tickers:
            raise RuntimeError("Universe cannot be empty.")

        #################################################
        # NORMALIZE + VALIDATE
        #################################################

        normalized = []

        for t in tickers:

            if not isinstance(t, str):
                raise RuntimeError("Non-string ticker detected.")

            t = t.strip().upper()

            if not t:
                raise RuntimeError("Empty ticker detected.")

            normalized.append(t)

        if len(normalized) != len(set(normalized)):
            raise RuntimeError("Duplicate tickers detected.")

        if len(normalized) < 5:
            raise RuntimeError("Universe too small.")

        return version, tuple(sorted(normalized))

    ###################################################
    # CACHE (THREAD SAFE)
    ###################################################

    @classmethod
    def _get_cached(cls):

        if cls._CACHE is None:

            with cls._LOCK:

                if cls._CACHE is None:

                    version, tickers = cls._load_file()

                    cls._CACHE = {
                        "version": version,
                        "tickers": tickers
                    }

        return cls._CACHE

    ###################################################
    # PUBLIC
    ###################################################

    @classmethod
    def get_universe(cls) -> Tuple[str, ...]:
        """
        Immutable tuple — cannot be mutated.
        """
        return cls._get_cached()["tickers"]

    ###################################################
    # FINGERPRINT (CANONICAL)
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

        tickers = list(cls.get_universe())

        return {
            "universe_version": cls._get_cached()["version"],
            "universe_hash": cls.fingerprint(),
            "universe_size": len(tickers),
            "tickers": tickers
        }

    ###################################################
    # VALIDATION
    ###################################################

    @classmethod
    def validate_subset(cls, tickers):

        unknown = set(tickers) - set(cls.get_universe())

        if unknown:
            raise RuntimeError(
                f"Unknown tickers detected: {unknown}"
            )
