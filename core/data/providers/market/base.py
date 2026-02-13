from abc import ABC, abstractmethod
import pandas as pd


class MarketDataProvider(ABC):

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Must return schema:

        date
        open
        high
        low
        close
        volume
        ticker
        """
        raise NotImplementedError
