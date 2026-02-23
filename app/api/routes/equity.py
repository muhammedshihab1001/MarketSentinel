import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from core.features.feature_engineering import FeatureEngineer
from app.inference.pipeline import InferencePipeline

router = APIRouter()
logger = logging.getLogger("marketsentinel.equity")

MIN_HISTORY_ROWS = 60
MIN_ASSETS_PER_DAY = 4
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
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

            df["forward_return"] = (
                df["close"].shift(-1) / df["close"] - 1
            )

            price_history[ticker] = df

        if not price_history:
            raise RuntimeError("No valid price data available.")

        ############################################################
        # 2️⃣ BUILD FEATURE DATASETS (CORE ONLY)
        ############################################################

        datasets = []

        for ticker, df in price_history.items():

            features = pipeline.feature_store.get_features(
                df,
                sentiment_df=None,
                ticker=ticker,
                training=False
            )

            if features is None or features.empty:
                continue

            datasets.append(features)

        if not datasets:
            raise RuntimeError("No feature datasets built.")

        ############################################################
        # 3️⃣ CONCAT + CROSS-SECTIONAL ALIGNMENT (CRITICAL FIX)
        ############################################################

        full_df = pd.concat(datasets, ignore_index=True)
        full_df = full_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # 🔥 Ensure cross-sectional features match live inference
        full_df = FeatureEngineer.add_cross_sectional_features(full_df)
        full_df = FeatureEngineer.finalize(full_df)

        ############################################################
        # 4️⃣ EVALUATION DATES
        ############################################################

        combined_dates = sorted(full_df["date"].unique())
        eval_dates = combined_dates[-days:]

        ############################################################
        # 5️⃣ HISTORICAL SIGNAL GENERATION
        ############################################################

        portfolio_records = []

        for eval_date in eval_dates:

            snapshot = full_df[full_df["date"] == eval_date].copy()

            if snapshot.empty:
                continue

            if snapshot["ticker"].nunique() < MIN_ASSETS_PER_DAY:
                logger.warning(
                    f"Skipping {eval_date} — insufficient cross-sectional width."
                )
                continue

            try:

                results = pipeline.run_historical_with_features(
                    df=snapshot,
                    evaluation_date=eval_date
                )

                portfolio_records.extend(results)

            except Exception as e:
                logger.warning(
                    f"Historical skip {eval_date} — {str(e)}"
                )

        if not portfolio_records:
            raise RuntimeError("No portfolio history generated.")

        portfolio_df = pd.DataFrame(portfolio_records)

        ############################################################
        # 6️⃣ FORWARD RETURNS
        ############################################################

        forward_frames = []

        for ticker, df in price_history.items():
            tmp = df[["date", "forward_return"]].copy()
            tmp["ticker"] = ticker
            forward_frames.append(tmp)

        forward_df = pd.concat(forward_frames, ignore_index=True)
        forward_df.dropna(inplace=True)

        ############################################################
        # 7️⃣ STRATEGY PERFORMANCE
        ############################################################

        report = engine.evaluate(portfolio_df, forward_df)
        strategy_equity = report.equity_curve

        ############################################################
        # 8️⃣ BENCHMARK
        ############################################################

        benchmark_df = market_data.get_price_data(
            ticker=BENCHMARK_TICKER,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            min_history=MIN_HISTORY_ROWS
        )

        benchmark_df = benchmark_df.sort_values("date")
        benchmark_df["date"] = pd.to_datetime(
            benchmark_df["date"]
        ).dt.normalize()

        benchmark_df["forward_return"] = (
            benchmark_df["close"].shift(-1) /
            benchmark_df["close"] - 1
        )

        benchmark_returns = (
            benchmark_df
            .set_index("date")["forward_return"]
            .reindex(strategy_equity.index)
            .dropna()
        )

        if benchmark_returns.empty:
            raise RuntimeError("Benchmark alignment failed.")

        benchmark_equity = (1 + benchmark_returns).cumprod()

        ############################################################
        # 9️⃣ ALIGN STRATEGY
        ############################################################

        aligned_strategy = strategy_equity.loc[
            benchmark_equity.index
        ]

        return {
            "dates": [
                d.strftime("%Y-%m-%d")
                for d in benchmark_equity.index
            ],
            "strategy_equity": aligned_strategy.tolist(),
            "benchmark_equity": benchmark_equity.tolist()
        }

    except Exception as e:
        logger.exception("Equity curve computation failed")
        raise HTTPException(status_code=500, detail=str(e))