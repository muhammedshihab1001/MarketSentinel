from typing import List
import hashlib
import json


class MarketUniverse:
    """
    Institutional Universe Controller.

    Guarantees:
    ✔ deterministic asset list
    ✔ fingerprinted universe
    ✔ audit-safe
    ✔ mutation-proof
    ✔ duplicate-proof
    ✔ lineage traceable
    """

    ###################################################
    # DO NOT MODIFY WITHOUT GOVERNANCE
    ###################################################

    _TRAINING_UNIVERSE = tuple(sorted([

        # Mega-cap tech
        "AAPL","MSFT","NVDA","AMZN",
        "GOOGL","META","TSLA",

        # Financials
        "JPM","GS",

        # Energy
        "XOM",

        # Semis
        "AMD","AVGO",

        # Market structure proxies
        "SPY","QQQ"
    ]))

    ###################################################
    # INTERNAL VALIDATION (runs at import)
    ###################################################

    if len(_TRAINING_UNIVERSE) != len(set(_TRAINING_UNIVERSE)):
        raise RuntimeError("Duplicate tickers detected in universe.")

    if len(_TRAINING_UNIVERSE) < 5:
        raise RuntimeError("Universe too small — unsafe for ML.")

    ###################################################
    # PUBLIC
    ###################################################

    @classmethod
    def get_universe(cls) -> List[str]:
        """
        Always return a COPY to prevent mutation.
        """
        return list(cls._TRAINING_UNIVERSE)

    ###################################################
    # FINGERPRINT (⭐ CRITICAL)
    ###################################################

    @classmethod
    def fingerprint(cls) -> str:
        """
        Tamper-proof universe hash.
        Must be stored in metadata.
        """

        canonical = json.dumps(
            cls._TRAINING_UNIVERSE,
            sort_keys=True
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
        """
        Prevent silent research corruption.
        """

        unknown = set(tickers) - set(cls._TRAINING_UNIVERSE)

        if unknown:
            raise RuntimeError(
                f"Unknown tickers detected: {unknown}"
            )

    ###################################################
    # AUDIT SNAPSHOT
    ###################################################

    @classmethod
    def snapshot(cls) -> dict:
        """
        Institutional lineage helper.
        """

        return {
            "universe_size": len(cls._TRAINING_UNIVERSE),
            "universe_hash": cls.fingerprint(),
            "tickers": list(cls._TRAINING_UNIVERSE)
        }
