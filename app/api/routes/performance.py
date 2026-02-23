import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from app.inference.pipeline import InferencePipeline

router = APIRouter()
logger = logging.getLogger("marketsentinel.performance")

MIN_HISTORY_ROWS = 60
MIN_ASSETS_PER_DAY = 4
BENCHMARK_TICKER = "SPY"


@router.get("/performance")
def compute_performance(days: int = 120):

    try:

        ############################################################
        # 🔁 Reuse heavy components once per request
        ############################################################

        engine = PerformanceEngine()
        pipeline = InferencePipeline()
        market_data = MarketDataService()

        universe = list(set(MarketUniverse.get_universe()))

        end_date = pd.Timestamp.utcnow().normalize()
        start_date = end_date - pd.Timedelta(days=days + 365)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        ############################################################
        # 1️⃣ FETCH PRICE HISTORY (PARALLELIZED)
        ############################################################

        price_history = market_data.get_price_data_batch(
            tickers=universe,
            start_date=start_str,
            end_date=end_str,
            interval="1d",
            min_history=MIN_HISTORY_ROWS
        )

        cleaned_history = {}

        for ticker, df in price_history.items():

            if df is None or len(df) < MIN_HISTORY_ROWS:
                continue

            df = df.sort_values("date").reset_index(drop=True)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

            df["forward_return"] = (
                df["close"].shift(-1) / df["close"] - 1
            )

            cleaned_history[ticker] = df

        if not cleaned_history:
            raise RuntimeError("No valid price data available.")

        ############################################################
        # 2️⃣ PRECOMPUTE FEATURES ONCE
        ############################################################

        full_feature_cache = {}

        for ticker, df in cleaned_history.items():

            features = pipeline.feature_store.get_features(
                df,
                sentiment_df=None,
                ticker=ticker,
                training=False
            )

            if features is not None and not features.empty:
                full_feature_cache[ticker] = features

        if not full_feature_cache:
            raise RuntimeError("No feature datasets built.")

        ############################################################
        # 3️⃣ EVALUATION DATES
        ############################################################

        all_dates = pd.concat(
            [df[["date"]] for df in cleaned_history.values()],
            ignore_index=True
        )["date"].unique()

        combined_dates = sorted(all_dates)

        eval_dates = combined_dates[-days:]

        ############################################################
        # 4️⃣ HISTORICAL SIGNAL GENERATION
        ############################################################

        portfolio_records = []

        for eval_date in eval_dates:

            try:

                results = pipeline.run_historical_with_features(
                    full_feature_cache=full_feature_cache,
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
        # 5️⃣ FORWARD RETURNS
        ############################################################

        forward_frames = []

        for ticker, df in cleaned_history.items():
            tmp = df[["date", "forward_return"]].copy()
            tmp["ticker"] = ticker
            forward_frames.append(tmp)

        forward_df = pd.concat(forward_frames, ignore_index=True)
        forward_df.dropna(inplace=True)

        ############################################################
        # 6️⃣ STRATEGY PERFORMANCE
        ############################################################

        report = engine.evaluate(portfolio_df, forward_df)

        ############################################################
        # 7️⃣ BENCHMARK (PARALLEL SAFE)
        ############################################################

        benchmark_map = market_data.get_price_data_batch(
            tickers=[BENCHMARK_TICKER],
            start_date=start_str,
            end_date=end_str,
            interval="1d",
            min_history=MIN_HISTORY_ROWS
        )

        benchmark_df = benchmark_map.get(BENCHMARK_TICKER)

        if benchmark_df is None or benchmark_df.empty:
            raise RuntimeError("Benchmark fetch failed.")

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
            .reindex(report.daily_returns.index)
            .dropna()
        )

        if benchmark_returns.empty:
            raise RuntimeError("Benchmark alignment failed.")

        benchmark_equity = (1 + benchmark_returns).cumprod()
        benchmark_cumulative = benchmark_equity.iloc[-1] - 1

        aligned_strategy = report.daily_returns.loc[
            benchmark_returns.index
        ]

        excess_returns = aligned_strategy - benchmark_returns

        info_ratio = 0.0
        if excess_returns.std() > 0:
            info_ratio = (
                excess_returns.mean() /
                excess_returns.std()
            ) * (252 ** 0.5)

        years = len(benchmark_returns) / 252
        benchmark_annual = (
            (1 + benchmark_cumulative) ** (1 / years) - 1
            if years > 0 else 0.0
        )

        alpha = report.annual_return - benchmark_annual

        ############################################################
        # 8️⃣ RESPONSE
        ############################################################

        return {
            "strategy": {
                "cumulative_return": report.cumulative_return,
                "annual_return": report.annual_return,
                "annual_volatility": report.annual_volatility,
                "sharpe_ratio": report.sharpe_ratio,
                "max_drawdown": report.max_drawdown,
                "hit_rate": report.hit_rate,
                "turnover": report.turnover,
            },
            "benchmark": {
                "ticker": BENCHMARK_TICKER,
                "cumulative_return": float(benchmark_cumulative),
            },
            "relative": {
                "alpha": float(alpha),
                "information_ratio": float(info_ratio),
            }
        }

    except Exception as e:
        logger.exception("Performance computation failed")
        raise HTTPException(status_code=500, detail=str(e))