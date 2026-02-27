import logging
import pandas as pd
import numpy as np
import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from core.analytics.performance_engine import PerformanceEngine
from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    DTYPE,
)

from app.inference.pipeline import InferencePipeline, get_shared_model_loader
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.equity")

MIN_HISTORY_ROWS = 60
MIN_ASSETS_PER_DAY = 4
BENCHMARK_TICKER = "SPY"
REQUEST_TIMEOUT = 60
MAX_CONCURRENT = 2
MIN_PROB_STD = 1e-6

equity_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

_pipeline: InferencePipeline | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


@router.get("/equity-curve")
async def equity_curve(days: int = 120):

    endpoint = "/equity-curve"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with equity_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_equity_curve_sync, days),
                timeout=REQUEST_TIMEOUT
            )

        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Equity curve timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Equity curve computation failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _equity_curve_sync(days: int):

    engine = PerformanceEngine()
    pipeline = get_pipeline()
    loader = get_shared_model_loader()
    market_data = MarketDataService()

    universe = list(set(MarketUniverse.get_universe()))

    end_date = pd.Timestamp.utcnow().normalize()
    start_date = end_date - pd.Timedelta(days=days + 365)

    ############################################################
    # 1️⃣ FETCH PRICE HISTORY
    ############################################################

    price_history = market_data.get_price_data_batch(
        tickers=universe,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        min_history=MIN_HISTORY_ROWS
    )

    cleaned_history = {}

    for ticker, df in price_history.items():

        if df is None or len(df) < MIN_HISTORY_ROWS:
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["forward_return"] = df["close"].shift(-1) / df["close"] - 1

        cleaned_history[ticker] = df

    if not cleaned_history:
        raise RuntimeError("No valid price data available.")

    ############################################################
    # 2️⃣ BUILD FEATURE DATASETS
    ############################################################

    datasets = []

    for ticker, df in cleaned_history.items():

        features = pipeline.feature_store.get_features(
            price_df=df,
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
    # 3️⃣ CROSS-SECTIONAL ALIGNMENT
    ############################################################

    full_df = pd.concat(datasets, ignore_index=True)
    full_df = full_df.sort_values(["date", "ticker"]).reset_index(drop=True)

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
    model = loader.xgb

    for eval_date in eval_dates:

        snapshot = full_df[full_df["date"] == eval_date].copy()

        if snapshot.empty:
            continue

        if snapshot["ticker"].nunique() < MIN_ASSETS_PER_DAY:
            continue

        try:

            feature_df = validate_feature_schema(
                snapshot.loc[:, MODEL_FEATURES],
                mode="inference"
            ).astype(DTYPE)

            probs = model.predict_proba(feature_df)[:, 1]
            probs = np.clip(probs, 1e-6, 1 - 1e-6)

            if np.std(probs) < MIN_PROB_STD:
                probs = np.full_like(probs, 0.5)

            snapshot["score"] = probs
            snapshot["rank_pct"] = snapshot["score"].rank(
                method="first", pct=True
            )

            weights = pipeline._construct_portfolio(snapshot)

            for _, row in snapshot.iterrows():
                portfolio_records.append({
                    "date": eval_date,
                    "ticker": row["ticker"],
                    "weight": weights.get(row["ticker"], 0.0)
                })

        except Exception as e:
            logger.warning(f"Historical skip {eval_date} — {str(e)}")

    if not portfolio_records:
        raise RuntimeError("No portfolio history generated.")

    portfolio_df = pd.DataFrame(portfolio_records)

    ############################################################
    # 6️⃣ FORWARD RETURNS
    ############################################################

    forward_frames = []

    for ticker, df in cleaned_history.items():
        tmp = df[["date", "forward_return"]].copy()
        tmp["ticker"] = ticker
        forward_frames.append(tmp)

    forward_df = pd.concat(forward_frames, ignore_index=True)
    forward_df.dropna(inplace=True)

    ############################################################
    # 7️⃣ BENCHMARK RETURNS
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
        .dropna()
    )

    ############################################################
    # 8️⃣ STRATEGY PERFORMANCE
    ############################################################

    report = engine.evaluate(
        portfolio_df,
        forward_df,
        benchmark_returns=benchmark_returns
    )

    ############################################################
    # 9️⃣ ALIGN STRATEGY + BENCHMARK
    ############################################################

    aligned_benchmark = benchmark_returns.reindex(
        report.equity_curve.index
    ).dropna()

    if aligned_benchmark.empty:
        raise RuntimeError("Benchmark alignment failed.")

    aligned_strategy = report.equity_curve.loc[
        aligned_benchmark.index
    ]

    benchmark_equity = (1 + aligned_benchmark).cumprod()

    ############################################################
    # 🔟 STRUCTURED OUTPUT
    ############################################################

    return {
        "summary": report.to_dict(),
        "series": {
            "dates": [
                d.strftime("%Y-%m-%d")
                for d in benchmark_equity.index
            ],
            "strategy_equity": aligned_strategy.tolist(),
            "benchmark_equity": benchmark_equity.tolist(),
            "drawdown": report.drawdown_series.loc[
                benchmark_equity.index
            ].tolist(),
        }
    }