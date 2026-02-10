from core.data.data_fetcher import StockPriceFetcher


class MarketDataService:
    """
    Institutional Market Data Boundary.

    Inference is NOT allowed to talk to providers directly.
    All market data must flow through this service.

    Future upgrades enabled:

    - batch ingestion
    - parquet lake
    - async pre-warming
    - provider routing
    - MLflow dataset lineage
    """

    def __init__(self):
        self._fetcher = StockPriceFetcher()

    # --------------------------------------------------

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):
        """
        Cache-first provider-backed retrieval.

        ZERO behavior change vs fetcher.
        Pure boundary insertion.
        """

        return self._fetcher.fetch(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
