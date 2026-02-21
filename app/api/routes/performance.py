import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from app.inference.pipeline import InferencePipeline

router = APIRouter()
logger = logging.getLogger("marketsentinel.performance")


MIN_HISTORY_ROWS = 50  # performance tolerance


@router.get("/performance")
def compute_performance(days: int = 120):

    try:

        engine = PerformanceEngine()
        pipeline = InferencePipeline()
        market_data = MarketDataService()

        universe = MarketUniverse.get_universe()

        end_date = pd.Timestamp.utcnow().date()
        start_date = end_date - pd.Timedelta(days=days + 10)

        # -------------------------------------------------
        # 1️⃣ FETCH HISTORY SAFELY
        # -------------------------------------------------

        all_prices = []

        for ticker in universe:

            try:
                df = market_data.get_price_data(
                    ticker=ticker,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )
            except Exception:
                logger.warning(f"Skipping {ticker} — fetch failure.")
                continue

            if df is None or df.empty:
                logger.warning(f"Skipping {ticker} — empty dataset.")
                continue

            if len(df) < MIN_HISTORY_ROWS:
                logger.warning(
                    f"Skipping {ticker} — insufficient history ({len(df)} rows)."
                )
                continue

            df = df.sort_values("date").copy()
            df["ticker"] = ticker

            df["forward_return"] = (
                df["close"].shift(-1) / df["close"] - 1
            )

            all_prices.append(df[["date", "ticker", "forward_return"]])

        if not all_prices:
            raise RuntimeError("No valid price data available.")

        forward_df = pd.concat(all_prices, ignore_index=True)
        forward_df.dropna(inplace=True)

        # -------------------------------------------------
        # 2️⃣ GENERATE HISTORICAL PORTFOLIOS
        # -------------------------------------------------

        portfolio_records = []

        eval_dates = (
            forward_df["date"]
            .drop_duplicates()
            .sort_values()
            .tail(days)
        )

        for eval_date in eval_dates:

            try:
                result = pipeline.run_batch(universe)

                for row in result:
                    portfolio_records.append({
                        "date": eval_date,
                        "ticker": row["ticker"],
                        "weight": row["weight"]
                    })

            except Exception:
                logger.warning(f"Skipping evaluation date {eval_date}")
                continue

        if not portfolio_records:
            raise RuntimeError("No portfolio history generated.")

        portfolio_df = pd.DataFrame(portfolio_records)

        # -------------------------------------------------
        # 3️⃣ EVALUATE PERFORMANCE
        # -------------------------------------------------

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