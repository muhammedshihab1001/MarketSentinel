import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from app.inference.pipeline import InferencePipeline

router = APIRouter()
logger = logging.getLogger("marketsentinel.performance")

MIN_HISTORY_ROWS = 60  # strict minimum for evaluation


@router.get("/performance")
def compute_performance(days: int = 120):

    try:

        engine = PerformanceEngine()
        pipeline = InferencePipeline()
        market_data = MarketDataService()

        universe = MarketUniverse.get_universe()

        end_date = pd.Timestamp.utcnow().normalize()
        start_date = end_date - pd.Timedelta(days=days + 365)

        ############################################################
        # 1️⃣ FETCH HISTORY ONCE (CRITICAL FIX)
        ############################################################

        price_history = {}

        for ticker in universe:

            try:

                df = market_data.get_price_data(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    interval="1d",
                    min_history=MIN_HISTORY_ROWS
                )

                if df is None or df.empty:
                    continue

                if len(df) < MIN_HISTORY_ROWS:
                    continue

                df = df.sort_values("date").reset_index(drop=True)

                df["forward_return"] = (
                    df["close"].shift(-1) / df["close"] - 1
                )

                price_history[ticker] = df

            except Exception as e:
                logger.warning(f"Skipping {ticker} — {str(e)}")
                continue

        if not price_history:
            raise RuntimeError("No valid price data available.")

        ############################################################
        # 2️⃣ BUILD EVALUATION DATES
        ############################################################

        combined_dates = sorted(
            set(
                pd.concat(price_history.values())["date"].unique()
            )
        )

        eval_dates = combined_dates[-days:]

        ############################################################
        # 3️⃣ WALK-FORWARD SIGNAL GENERATION
        ############################################################

        portfolio_records = []

        for eval_date in eval_dates:

            try:

                results = pipeline.run_historical_batch(
                    price_history=price_history,
                    evaluation_date=eval_date
                )

                for row in results:
                    portfolio_records.append(row)

            except Exception as e:
                logger.warning(
                    f"Skipping eval_date {eval_date} — {str(e)}"
                )
                continue

        if not portfolio_records:
            raise RuntimeError("No portfolio history generated.")

        portfolio_df = pd.DataFrame(portfolio_records)

        ############################################################
        # 4️⃣ BUILD FORWARD RETURN FRAME
        ############################################################

        forward_frames = []

        for ticker, df in price_history.items():

            tmp = df[["date", "forward_return"]].copy()
            tmp["ticker"] = ticker

            forward_frames.append(tmp)

        forward_df = pd.concat(forward_frames, ignore_index=True)
        forward_df.dropna(inplace=True)

        ############################################################
        # 5️⃣ EVALUATE
        ############################################################

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