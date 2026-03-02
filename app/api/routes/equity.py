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
BENCHMARK_TICKER = "SPY"
REQUEST_TIMEOUT = 180
MAX_CONCURRENT = 2
MIN_SCORE_STD = 1e-6

equity_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_pipeline: InferencePipeline | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


# =========================================================
# ASYNC ENTRYPOINT
# =========================================================

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


# =========================================================
# SYNC LOGIC
# =========================================================

def _equity_curve_sync(days: int):

    start_time = time.time()

    engine = PerformanceEngine()
    pipeline = get_pipeline()
    loader = get_shared_model_loader()
    market_data = MarketDataService()

    universe = list(set(MarketUniverse.get_universe()))

    end_date = pd.Timestamp.utcnow().normalize()
    start_date = end_date - pd.Timedelta(days=days + 365)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # =========================================================
    # FETCH PRICE HISTORY
    # =========================================================

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
        ).replace([np.inf, -np.inf], np.nan)

        cleaned_history[ticker] = df

    if not cleaned_history:
        raise RuntimeError("No valid price data available.")

    # =========================================================
    # BUILD FEATURES
    # =========================================================

    datasets = []

    for ticker, df in cleaned_history.items():

        try:
            features = FeatureEngineer.build_feature_pipeline(
                price_df=df,
                sentiment_df=None,
                training=False
            )

            if features is None or features.empty:
                continue

            datasets.append(features)

        except Exception:
            logger.warning("Feature build failed for %s", ticker)

    if not datasets:
        raise RuntimeError("No feature datasets built.")

    full_df = pd.concat(datasets, ignore_index=True)
    full_df = full_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    full_df = FeatureEngineer.add_cross_sectional_features(full_df)
    full_df = FeatureEngineer.finalize(full_df)

    eval_dates = sorted(full_df["date"].unique())[-days:]

    portfolio_records = []
    model = loader.xgb

    # =========================================================
    # HISTORICAL SIGNAL GENERATION
    # =========================================================

    for eval_date in eval_dates:

        snapshot = full_df[full_df["date"] == eval_date].copy()

        if snapshot["ticker"].nunique() < pipeline.MIN_UNIVERSE_WIDTH:
            continue

        feature_df = validate_feature_schema(
            snapshot.loc[:, MODEL_FEATURES],
            mode="inference"
        ).astype(DTYPE)

        scores = model.predict(feature_df)

        if np.std(scores) < MIN_SCORE_STD:
            continue

        scores = (scores - scores.mean()) / (scores.std() + 1e-12)
        snapshot["score"] = scores

        ranked = snapshot.sort_values("score")

        longs = ranked.tail(pipeline.TOP_K)
        shorts = ranked.head(pipeline.BOTTOM_K)

        if longs.empty or shorts.empty:
            continue

        # Flexible portfolio construction
        weights = pipeline._construct_portfolio(longs, shorts)

        # Use safe drift wrapper
        drift_result = pipeline._safe_drift(feature_df)
        exposure_scale = drift_result.get("exposure_scale", 1.0)

        for ticker in weights:
            weights[ticker] *= exposure_scale

        for _, row in snapshot.iterrows():
            portfolio_records.append({
                "date": eval_date,
                "ticker": row["ticker"],
                "weight": float(weights.get(row["ticker"], 0.0))
            })

    if not portfolio_records:
        raise RuntimeError("No portfolio history generated.")

    portfolio_df = pd.DataFrame(portfolio_records)

    # =========================================================
    # FORWARD RETURNS
    # =========================================================

    forward_frames = []

    for ticker, df in cleaned_history.items():
        tmp = df[["date", "forward_return"]].copy()
        tmp["ticker"] = ticker
        forward_frames.append(tmp)

    forward_df = pd.concat(forward_frames, ignore_index=True)
    forward_df.dropna(inplace=True)

    report = engine.evaluate(portfolio_df, forward_df)

    # =========================================================
    # BENCHMARK
    # =========================================================

    benchmark_equity = []

    benchmark_data = market_data.get_price_data_batch(
        tickers=[BENCHMARK_TICKER],
        start_date=start_str,
        end_date=end_str,
        interval="1d",
        min_history=MIN_HISTORY_ROWS
    )

    benchmark_df = benchmark_data.get(BENCHMARK_TICKER)

    if benchmark_df is not None and not benchmark_df.empty:

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
            .reindex(report.equity_curve.index)
            .dropna()
        )

        if len(benchmark_returns) > 1:
            benchmark_equity = (
                (1 + benchmark_returns).cumprod().tolist()
            )

    # =========================================================
    # OUTPUT
    # =========================================================

    return {
        "summary": report.to_dict(),
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in report.equity_curve.index],
            "strategy_equity": report.equity_curve.tolist(),
            "drawdown": report.drawdown_series.tolist(),
        },
        "benchmark": {
            "ticker": BENCHMARK_TICKER,
            "equity": benchmark_equity
        },
        "governance": {
            "model_version": loader.xgb_version,
            "schema_signature": loader.schema_signature,
            "artifact_hash": loader.artifact_hash,
        },
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time())
    }