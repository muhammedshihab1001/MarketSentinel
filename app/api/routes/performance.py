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
logger = logging.getLogger("marketsentinel.performance")

MIN_HISTORY_ROWS = 60
BENCHMARK_TICKER = "SPY"
MIN_SCORE_STD = 1e-6
REQUEST_TIMEOUT = 60
MAX_CONCURRENT = 2

performance_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_pipeline: InferencePipeline | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


# =========================================================
# ASYNC ENTRYPOINT
# =========================================================

@router.get("/performance")
async def compute_performance(days: int = 120):

    endpoint = "/performance"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with performance_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_compute_performance_sync, days),
                timeout=REQUEST_TIMEOUT
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Performance computation timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Performance computation failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# SYNC LOGIC
# =========================================================

def _compute_performance_sync(days: int):

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
    # FETCH PRICE DATA
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

        df["forward_return"] = df["close"].shift(-1) / df["close"] - 1
        df["forward_return"] = df["forward_return"].replace(
            [np.inf, -np.inf], np.nan
        )

        cleaned_history[ticker] = df

    if not cleaned_history:
        raise RuntimeError("No valid price data available.")

    # =========================================================
    # BUILD FEATURES (UPDATED — NO feature_store)
    # =========================================================

    feature_frames = []

    for ticker, df in cleaned_history.items():

        try:
            features = FeatureEngineer.build_feature_pipeline(
                price_df=df,
                sentiment_df=None,
                training=False
            )

            if features is None or features.empty:
                continue

            feature_frames.append(features)

        except Exception:
            logger.warning("Feature build failed for %s", ticker)

    if not feature_frames:
        raise RuntimeError("No feature datasets built.")

    full_df = pd.concat(feature_frames, ignore_index=True)
    full_df = full_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    full_df = FeatureEngineer.add_cross_sectional_features(full_df)
    full_df = FeatureEngineer.finalize(full_df)

    portfolio_records = []
    eval_dates = sorted(full_df["date"].unique())[-days:]
    model = loader.xgb

    # =========================================================
    # DAILY PORTFOLIO SIMULATION
    # =========================================================

    for eval_date in eval_dates:

        daily_slice = full_df[full_df["date"] == eval_date].copy()

        if daily_slice["ticker"].nunique() < 5:
            continue

        feature_df = validate_feature_schema(
            daily_slice.loc[:, MODEL_FEATURES],
            mode="inference"
        ).astype(DTYPE)

        scores = model.predict(feature_df)

        if np.std(scores) < MIN_SCORE_STD:
            continue

        scores = (scores - scores.mean()) / (scores.std() + 1e-12)
        daily_slice["score"] = scores

        ranked = daily_slice.sort_values("score")

        longs = ranked.tail(pipeline.TOP_K)
        shorts = ranked.head(pipeline.BOTTOM_K)

        if longs.empty or shorts.empty:
            continue

        weights = pipeline._construct_portfolio(longs, shorts)

        for _, row in daily_slice.iterrows():
            portfolio_records.append({
                "date": row["date"],
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
    forward_df = forward_df.dropna(subset=["forward_return"])

    report = engine.evaluate(portfolio_df, forward_df)

    # =========================================================
    # BENCHMARK (SPY)
    # =========================================================

    benchmark_data = market_data.get_price_data_batch(
        tickers=[BENCHMARK_TICKER],
        start_date=start_str,
        end_date=end_str,
        interval="1d",
        min_history=MIN_HISTORY_ROWS
    )

    benchmark_df = benchmark_data.get(BENCHMARK_TICKER)

    benchmark_cumulative = 0.0
    alpha = 0.0
    info_ratio = 0.0

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
            .reindex(report.daily_returns.index)
            .dropna()
        )

        if len(benchmark_returns) > 2:

            benchmark_equity = (1 + benchmark_returns).cumprod()
            benchmark_cumulative = float(
                benchmark_equity.iloc[-1] - 1
            )

            aligned_strategy = report.daily_returns.loc[
                benchmark_returns.index
            ]

            excess_returns = aligned_strategy - benchmark_returns

            if excess_returns.std() > 0:
                info_ratio = float(
                    (excess_returns.mean() /
                     excess_returns.std()) * np.sqrt(252)
                )

            years = len(benchmark_returns) / 252
            benchmark_annual = (
                (1 + benchmark_cumulative) ** (1 / years) - 1
                if years > 0 else 0.0
            )

            alpha = float(report.annual_return - benchmark_annual)

    return {
        "strategy": {
            "cumulative_return": float(report.cumulative_return),
            "annual_return": float(report.annual_return),
            "annual_volatility": float(report.annual_volatility),
            "sharpe_ratio": float(report.sharpe_ratio),
            "max_drawdown": float(report.max_drawdown),
            "hit_rate": float(report.hit_rate),
            "turnover": float(report.turnover),
        },
        "benchmark": {
            "ticker": BENCHMARK_TICKER,
            "cumulative_return": float(benchmark_cumulative),
        },
        "relative": {
            "alpha": float(alpha),
            "information_ratio": float(info_ratio),
        },
        "governance": {
            "model_version": loader.xgb_version,
            "schema_signature": loader.schema_signature,
            "artifact_hash": loader.artifact_hash,
        },
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time())
    }