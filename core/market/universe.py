class MarketUniverse:
    """
    Institutional universe controller.

    Guarantees:
    ✔ identical assets across models
    ✔ audit-safe research
    ✔ prevents universe drift
    ✔ deterministic experiments
    """

    TRAINING_UNIVERSE = [

        "AAPL","MSFT","NVDA","AMZN",
        "GOOGL","META","TSLA",

        "JPM","GS",
        "XOM",

        "AMD","AVGO",

        "SPY","QQQ"
    ]

    @classmethod
    def get_universe(cls):
        return list(cls.TRAINING_UNIVERSE)
