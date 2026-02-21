import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from app.inference.pipeline import InferencePipeline

router = APIRouter()
logger = logging.getLogger("marketsentinel.equity")

MIN_HISTORY_ROWS = 60
BENCHMARK_TICKER = "SPY"


@router.get("/equity-curve")
def equity_curve(days: int = 120):

    try:

        engine = PerformanceEngine()
        pipeline = InferencePipeline()
        market_data = MarketDataService()

        universe = MarketUniverse.get_universe()

        end_date = pd.Timestamp.utcnow().normalize()
        start_date = end_date - pd.Timedelta(days=days + 365)

        ############################################################
        # 1️⃣ FETCH PRICE HISTORY
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

                if df is None or len(df) < MIN_HISTORY_ROWS:
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
        # 2️⃣ FETCH BENCHMARK
        ############################################################

        benchmark_df = market_data.get_price_data(
            ticker=BENCHMARK_TICKER,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            min_history=MIN_HISTORY_ROWS
        )

        benchmark_df = benchmark_df.sort_values("date")
        benchmark_df["forward_return"] = (
            benchmark_df["close"].shift(-1) / benchmark_df["close"] - 1
        )

        ############################################################
        # 3️⃣ BUILD EVALUATION DATES
        ############################################################

        combined_dates = sorted(
            set(
                pd.concat(price_history.values())["date"].unique()
            )
        )

        eval_dates = combined_dates[-days:]

        ############################################################
        # 4️⃣ WALK-FORWARD STRATEGY
        ############################################################

        portfolio_records = []

        for eval_date in eval_dates:

            try:

                results = pipeline.run_historical_batch(
                    price_history=price_history,
                    evaluation_date=eval_date
                )

                portfolio_records.extend(results)

            except Exception:
                continue

        if not portfolio_records:
            raise RuntimeError("No portfolio history generated.")

        portfolio_df = pd.DataFrame(portfolio_records)

        ############################################################
        # 5️⃣ FORWARD RETURNS
        ############################################################

        forward_frames = []

        for ticker, df in price_history.items():
            tmp = df[["date", "forward_return"]].copy()
            tmp["ticker"] = ticker
            forward_frames.append(tmp)

        forward_df = pd.concat(forward_frames, ignore_index=True)
        forward_df.dropna(inplace=True)

        ############################################################
        # 6️⃣ STRATEGY PERFORMANCE
        ############################################################

        report = engine.evaluate(portfolio_df, forward_df)

        strategy_equity = report.equity_curve

        ############################################################
        # 7️⃣ BENCHMARK EQUITY
        ############################################################

        benchmark_returns = (
            benchmark_df
            .set_index("date")["forward_return"]
            .loc[strategy_equity.index]
            .dropna()
        )

        benchmark_equity = (1 + benchmark_returns).cumprod()

        ############################################################
        # 8️⃣ ALIGN
        ############################################################

        aligned_strategy = strategy_equity.loc[benchmark_equity.index]

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in benchmark_equity.index],
            "strategy_equity": aligned_strategy.tolist(),
            "benchmark_equity": benchmark_equity.tolist()
        }

    except Exception as e:
        logger.exception("Equity curve computation failed")
        raise HTTPException(status_code=500, detail=str(e))