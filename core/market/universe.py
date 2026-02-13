from typing import List, Dict
import hashlib
import json


class MarketUniverse:
    """
    Institutional Universe Controller.

    Guarantees:
    deterministic asset list
    fingerprinted universe
    audit-safe
    mutation-proof
    duplicate-proof
    lineage traceable
    """

    UNIVERSE_VERSION = "2.0"

    _TRAINING_UNIVERSE = tuple(sorted([

        "AAPL","MSFT","NVDA","AMZN",
        "GOOGL","META","TSLA",

        "JPM","GS",

        "XOM",

        "AMD","AVGO",

        "SPY","QQQ"
    ]))

    ###################################################
    # INTERNAL VALIDATION
    ###################################################

    for t in _TRAINING_UNIVERSE:

        if not isinstance(t, str):
            raise RuntimeError("Non-string ticker detected.")

        if t.strip() != t:
            raise RuntimeError("Ticker contains whitespace.")

        if not t.isupper():
            raise RuntimeError("Tickers must be uppercase.")

        if not t:
            raise RuntimeError("Empty ticker detected.")

    if len(_TRAINING_UNIVERSE) != len(set(_TRAINING_UNIVERSE)):
        raise RuntimeError("Duplicate tickers detected in universe.")

    if len(_TRAINING_UNIVERSE) < 5:
        raise RuntimeError("Universe too small — unsafe for ML.")

    ###################################################
    # PUBLIC
    ###################################################

    @classmethod
    def get_universe(cls) -> List[str]:
        return list(cls._TRAINING_UNIVERSE)

    ###################################################
    # FINGERPRINT
    ###################################################

    @classmethod
    def _contract(cls) -> Dict:

        return {
            "version": cls.UNIVERSE_VERSION,
            "tickers": list(cls._TRAINING_UNIVERSE),
            "size": len(cls._TRAINING_UNIVERSE)
        }

    @classmethod
    def fingerprint(cls) -> str:

        canonical = json.dumps(
            cls._contract(),
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    ###################################################
    # SAFETY
    ###################################################

    @classmethod
    def universe_size(cls) -> int:
        return len(cls._TRAINING_UNIVERSE)

    @classmethod
    def validate_subset(cls, tickers: List[str]):

        unknown = set(tickers) - set(cls._TRAINING_UNIVERSE)

        if unknown:
            raise RuntimeError(
                f"Unknown tickers detected: {unknown}"
            )

    ###################################################
    # AUDIT SNAPSHOT
    ###################################################

    @classmethod
    def snapshot(cls) -> Dict:
        """
        Snapshot is STRUCTURE LOCKED.
        Do not modify without version bump.
        """

        contract = cls._contract()

        return {
            "universe_version": cls.UNIVERSE_VERSION,
            "universe_hash": cls.fingerprint(),
            "universe_size": contract["size"],
            "tickers": contract["tickers"]
        }
