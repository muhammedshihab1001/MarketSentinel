import time
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.market.universe import MarketUniverse
from app.inference.pipeline import InferencePipeline

router = APIRouter()
logger = logging.getLogger("marketsentinel.performance")


@router.get("/performance")
def compute_performance(days: int = 120):

    try:

        engine = PerformanceEngine()
        pipeline = InferencePipeline()
        market_data = MarketDataService()
        feature_store = FeatureStore()

        universe = MarketUniverse.get_universe()

        end_date = pd.Timestamp.utcnow().date()
        start_date = end_date - pd.Timedelta(days=days + 5)

        portfolio_records = []
        forward_records = []

        for offset in range(days):

            eval_date = end_date - pd.Timedelta(days=offset)

            tickers = universe

            result = pipeline.run_batch(tickers)

            for row in result:
                portfolio_records.append({
                    "date": eval_date,
                    "ticker": row["ticker"],
                    "weight": row["weight"]
                })

                # forward return proxy (next day close-to-close)
                price_df = market_data.get_price_data(
                    ticker=row["ticker"],
                    start_date=(eval_date - pd.Timedelta(days=2)).isoformat(),
                    end_date=(eval_date + pd.Timedelta(days=2)).isoformat()
                )

                if price_df is None or len(price_df) < 2:
                    continue

                price_df = price_df.sort_values("date")

                if len(price_df) >= 2:
                    r = (
                        price_df.iloc[-1]["close"]
                        / price_df.iloc[-2]["close"]
                    ) - 1

                    forward_records.append({
                        "date": eval_date,
                        "ticker": row["ticker"],
                        "forward_return": r
                    })

        portfolio_df = pd.DataFrame(portfolio_records)
        forward_df = pd.DataFrame(forward_records)

        report = engine.evaluate(portfolio_df, forward_df)

        return {
            "cumulative_return": report.cumulative_return,
            "sharpe_ratio": report.sharpe_ratio,
            "max_drawdown": report.max_drawdown,
            "hit_rate": report.hit_rate,
            "annual_volatility": report.annual_volatility,
            "annual_return": report.annual_return,
            "turnover": report.turnover
        }

    except Exception as e:
        logger.exception("Performance computation failed")
        raise HTTPException(status_code=500, detail=str(e))