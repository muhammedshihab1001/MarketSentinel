# =========================================================
# EQUITY ROUTE v2.2
# DB-Backed | Pipeline-Integrated | CV-Optimized
# =========================================================

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.inference.pipeline import InferencePipeline, get_shared_model_loader
from core.market.universe import MarketUniverse
from core.data.market_data_service import MarketDataService
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.equity")

router = APIRouter()

# =========================================================
# EQUITY DETAIL ENDPOINT
# =========================================================

@router.get("/equity/{ticker}")
async def equity_detail(ticker: str):
    """
    Returns equity detail for a single ticker:
    - Latest OHLCV snapshot from DB
    - Computed features (from pipeline)
    - Signal + confidence from last snapshot
    """

    endpoint = f"/equity/{ticker}"
    API_REQUEST_COUNT.labels(endpoint="/equity").inc()
    start_time = time.time()

    ticker = ticker.upper().strip()

    try:
        universe = MarketUniverse.snapshot()
        valid_tickers = set(universe.get("tickers", []))

        if ticker not in valid_tickers:
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker}' not in universe"
            )

        def _fetch():
            svc = MarketDataService()
            df = svc.get_price_data(ticker, interval="1d", min_history=30)
            return df

        df = await run_in_threadpool(_fetch)

        if df is None or df.empty:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}"
            )

        latest = df.iloc[-1]

        # Build OHLCV summary from DB data
        ohlcv = {
            "date": str(latest.name.date()) if hasattr(latest.name, "date") else str(latest.name),
            "open": round(float(latest.get("Open", np.nan)), 4),
            "high": round(float(latest.get("High", np.nan)), 4),
            "low": round(float(latest.get("Low", np.nan)), 4),
            "close": round(float(latest.get("Close", np.nan)), 4),
            "volume": int(latest.get("Volume", 0)),
        }

        # 5-day and 20-day return
        returns = {}
        if len(df) >= 6:
            ret_5 = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1)
            returns["5d_return"] = round(float(ret_5), 6)
        if len(df) >= 21:
            ret_20 = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1)
            returns["20d_return"] = round(float(ret_20), 6)

        # Volatility (20-day annualised)
        if len(df) >= 21:
            log_rets = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            vol = float(log_rets.tail(20).std() * np.sqrt(252))
            returns["volatility_20d_ann"] = round(vol, 6)

        return {
            "ticker": ticker,
            "ohlcv": ohlcv,
            "returns": returns,
            "data_source": "postgresql",
            "rows_available": len(df),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint="/equity").inc()
        logger.exception("Equity endpoint failure | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint="/equity").observe(time.time() - start_time)


# =========================================================
# EQUITY HISTORY ENDPOINT
# =========================================================

@router.get("/equity/{ticker}/history")
async def equity_history(ticker: str, days: int = 90):
    """
    Returns OHLCV history for a ticker from PostgreSQL.
    days: number of trading days to return (default 90, max 500)
    """

    endpoint = "/equity/history"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    ticker = ticker.upper().strip()
    days = max(5, min(days, 500))

    try:
        universe = MarketUniverse.snapshot()
        valid_tickers = set(universe.get("tickers", []))

        if ticker not in valid_tickers:
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker}' not in universe"
            )

        def _fetch():
            svc = MarketDataService()
            return svc.get_price_data(ticker, interval="1d", min_history=days)

        df = await run_in_threadpool(_fetch)

        if df is None or df.empty:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}"
            )

        df = df.tail(days)

        records = []
        for dt, row in df.iterrows():
            records.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "open": round(float(row.get("Open", np.nan)), 4),
                "high": round(float(row.get("High", np.nan)), 4),
                "low": round(float(row.get("Low", np.nan)), 4),
                "close": round(float(row.get("Close", np.nan)), 4),
                "volume": int(row.get("Volume", 0)),
            })

        return {
            "ticker": ticker,
            "days_requested": days,
            "rows_returned": len(records),
            "data_source": "postgresql",
            "history": records,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Equity history failure | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)