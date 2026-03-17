# =========================================================
# PERFORMANCE ROUTE v2.2
# DB-Backed | CV-Optimized
# =========================================================

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
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.performance")

router = APIRouter()


# =========================================================
# PERFORMANCE SUMMARY ENDPOINT
# =========================================================

@router.get("/performance")
async def performance_summary(tickers: str = "", days: int = 252):
    """
    Returns portfolio-level performance metrics.
    tickers: comma-separated list (default: full universe)
    days: lookback window (default 252 = 1 year, max 500)
    """

    endpoint = "/performance"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    days = max(30, min(days, 500))

    try:
        universe = MarketUniverse.snapshot()
        all_tickers = universe.get("tickers", [])

        if tickers:
            requested = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            valid = set(all_tickers)
            ticker_list = [t for t in requested if t in valid]
            if not ticker_list:
                raise HTTPException(
                    status_code=400,
                    detail="No valid tickers in request"
                )
        else:
            ticker_list = all_tickers

        def _fetch_and_compute():
            svc = MarketDataService()
            price_map, errors = svc.get_price_data_batch(
                ticker_list,
                interval="1d",
                min_history=days,
            )

            if not price_map:
                return None, errors

            # Build close price matrix
            close_frames = {}
            for t, df in price_map.items():
                if df is not None and not df.empty and "Close" in df.columns:
                    close_frames[t] = df["Close"].tail(days)

            if not close_frames:
                return None, errors

            close_matrix = pd.DataFrame(close_frames).dropna(how="all")

            engine = PerformanceEngine()
            report = engine.compute(close_matrix)
            return report, errors

        report, errors = await run_in_threadpool(_fetch_and_compute)

        if report is None:
            raise HTTPException(
                status_code=503,
                detail="No price data available for performance computation"
            )

        result = {
            "tickers_requested": len(ticker_list),
            "tickers_computed": len(ticker_list) - len(errors),
            "lookback_days": days,
            "data_source": "postgresql",
            "metrics": {
                "cumulative_return": round(float(report.cumulative_return), 6),
                "sharpe_ratio": round(float(report.sharpe_ratio), 4),
                "sortino_ratio": round(float(report.sortino_ratio), 4),
                "calmar_ratio": round(float(report.calmar_ratio), 4),
                "max_drawdown": round(float(report.max_drawdown), 6),
                "volatility_ann": round(float(report.volatility_ann), 6),
                "win_rate": round(float(report.win_rate), 4),
            },
        }

        if errors:
            result["fetch_errors"] = errors

        return result

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Performance endpoint failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# PER-TICKER PERFORMANCE ENDPOINT
# =========================================================

@router.get("/performance/{ticker}")
async def ticker_performance(ticker: str, days: int = 252):
    """
    Returns performance metrics for a single ticker.
    """

    endpoint = "/performance/ticker"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    ticker = ticker.upper().strip()
    days = max(30, min(days, 500))

    try:
        universe = MarketUniverse.snapshot()
        valid_tickers = set(universe.get("tickers", []))

        if ticker not in valid_tickers:
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker}' not in universe"
            )

        def _fetch_and_compute():
            svc = MarketDataService()
            df = svc.get_price_data(ticker, interval="1d", min_history=days)
            if df is None or df.empty:
                return None
            close = df["Close"].tail(days).to_frame()
            engine = PerformanceEngine()
            return engine.compute(close)

        report = await run_in_threadpool(_fetch_and_compute)

        if report is None:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}"
            )

        return {
            "ticker": ticker,
            "lookback_days": days,
            "data_source": "postgresql",
            "metrics": {
                "cumulative_return": round(float(report.cumulative_return), 6),
                "sharpe_ratio": round(float(report.sharpe_ratio), 4),
                "sortino_ratio": round(float(report.sortino_ratio), 4),
                "calmar_ratio": round(float(report.calmar_ratio), 4),
                "max_drawdown": round(float(report.max_drawdown), 6),
                "volatility_ann": round(float(report.volatility_ann), 6),
                "win_rate": round(float(report.win_rate), 4),
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Ticker performance failure | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)