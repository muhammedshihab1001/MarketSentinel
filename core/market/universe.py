from typing import Tuple, Dict
import hashlib
import json
import os


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
    """

    UNIVERSE_FILE = "config/universe.json"

    ###################################################
    # LOAD
    ###################################################

    @classmethod
    def _load_file(cls):

        if not os.path.exists(cls.UNIVERSE_FILE):
            raise RuntimeError(
                f"Universe file missing: {cls.UNIVERSE_FILE}"
            )

        with open(cls.UNIVERSE_FILE, "r") as f:
            payload = json.load(f)

        if "version" not in payload or "tickers" not in payload:
            raise RuntimeError("Invalid universe config.")

        tickers = payload["tickers"]

        if not isinstance(tickers, list) or not tickers:
            raise RuntimeError("Universe tickers invalid.")

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

        return payload["version"], tuple(sorted(normalized))

    ###################################################
    # CACHE (immutable)
    ###################################################

    _CACHE = None

    @classmethod
    def _get_cached(cls):

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
    # FINGERPRINT
    ###################################################

    @classmethod
    def fingerprint(cls) -> str:

        contract = {
            "version": cls._get_cached()["version"],
            "tickers": cls.get_universe(),
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
