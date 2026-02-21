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
BENCHMARK_TICKER = "SPY"


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
        # 1️⃣ FETCH STRATEGY HISTORY
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
        # 2️⃣ PRECOMPUTE FULL FEATURES ONCE
        ############################################################

        full_feature_cache = {}

        for ticker, df in price_history.items():

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
        # 3️⃣ FETCH BENCHMARK ONCE
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
        # 4️⃣ BUILD EVALUATION DATES
        ############################################################

        combined_dates = sorted(
            set(
                pd.concat(price_history.values())["date"].unique()
            )
        )

        eval_dates = combined_dates[-days:]

        ############################################################
        # 5️⃣ GENERATE SIGNALS USING SLICING ONLY
        ############################################################

        portfolio_records = []

        for eval_date in eval_dates:

            sliced = []

            for ticker, feature_df in full_feature_cache.items():
                df = feature_df[feature_df["date"] == eval_date]
                if not df.empty:
                    sliced.append(df)

            if not sliced:
                continue

            latest_df = pd.concat(sliced, ignore_index=True)

            try:
                results = pipeline._run_model_and_construct(
                    latest_df,
                    use_cache=False
                )
                portfolio_records.extend(results)
            except Exception as e:
                logger.warning(
                    f"Skipping eval_date {eval_date} — {str(e)}"
                )
                continue

        if not portfolio_records:
            raise RuntimeError("No portfolio history generated.")

        portfolio_df = pd.DataFrame(portfolio_records)

        ############################################################
        # 6️⃣ BUILD FORWARD RETURN FRAME
        ############################################################

        forward_frames = []

        for ticker, df in price_history.items():
            tmp = df[["date", "forward_return"]].copy()
            tmp["ticker"] = ticker
            forward_frames.append(tmp)

        forward_df = pd.concat(forward_frames, ignore_index=True)
        forward_df.dropna(inplace=True)

        ############################################################
        # 7️⃣ STRATEGY EVALUATION
        ############################################################

        report = engine.evaluate(portfolio_df, forward_df)

        ############################################################
        # 8️⃣ BENCHMARK METRICS
        ############################################################

        benchmark_returns = (
            benchmark_df
            .set_index("date")["forward_return"]
            .loc[report.daily_returns.index]
            .dropna()
        )

        benchmark_equity = (1 + benchmark_returns).cumprod()
        benchmark_cumulative = benchmark_equity.iloc[-1] - 1

        aligned_strategy = report.daily_returns.loc[benchmark_returns.index]

        excess_returns = aligned_strategy - benchmark_returns

        info_ratio = 0.0
        if excess_returns.std() != 0:
            info_ratio = (
                excess_returns.mean() / excess_returns.std()
            ) * (252 ** 0.5)

        alpha = report.annual_return - (
            (1 + benchmark_cumulative) ** (252 / len(benchmark_returns)) - 1
        )

        ############################################################
        # 9️⃣ RESPONSE
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