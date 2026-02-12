from typing import List


class MarketUniverse:
    """
    Institutional universe controller.

    Guarantees:
    ✔ identical assets across models
    ✔ audit-safe research
    ✔ prevents universe drift
    ✔ deterministic experiments
    ✔ lineage traceability
    """

    # DO NOT MODIFY WITHOUT GOVERNANCE
    _TRAINING_UNIVERSE = [

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
    ]

    ###################################################
    # PUBLIC
    ###################################################

    @classmethod
    def get_universe(cls) -> List[str]:
        """
        Always return a COPY.

        Prevents runtime mutation bugs.
        """
        return list(cls._TRAINING_UNIVERSE)

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
