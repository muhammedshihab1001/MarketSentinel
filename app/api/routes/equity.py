# =========================================================
# EQUITY ROUTE v2.3
# FIX: pass start_date/end_date to get_price_data
#      MarketDataService requires these as positional args
# =========================================================

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

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


def _date_window(days: int):
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days + 30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# =========================================================
# EQUITY DETAIL ENDPOINT
# =========================================================

@router.get("/equity/{ticker}")
async def equity_detail(ticker: str):

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
                detail=f"Ticker '{ticker}' not in universe",
            )

        def _fetch():
            svc = MarketDataService()
            start_date, end_date = _date_window(30)
            return svc.get_price_data(
                ticker,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                min_history=30,
            )

        df = await run_in_threadpool(_fetch)

        if df is None or df.empty:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}",
            )

        latest = df.iloc[-1]

        def _get(row, *keys):
            for k in keys:
                if k in row.index:
                    return row[k]
            return np.nan

        ohlcv = {
            "date": str(latest.name.date()) if hasattr(latest.name, "date") else str(latest.name),
            "open": round(float(_get(latest, "open", "Open")), 4),
            "high": round(float(_get(latest, "high", "High")), 4),
            "low": round(float(_get(latest, "low", "Low")), 4),
            "close": round(float(_get(latest, "close", "Close")), 4),
            "volume": int(_get(latest, "volume", "Volume") or 0),
        }

        close_col = "close" if "close" in df.columns else "Close"
        returns = {}

        if len(df) >= 6:
            returns["5d_return"] = round(float(df[close_col].iloc[-1] / df[close_col].iloc[-6] - 1), 6)
        if len(df) >= 21:
            returns["20d_return"] = round(float(df[close_col].iloc[-1] / df[close_col].iloc[-21] - 1), 6)
            log_rets = np.log(df[close_col] / df[close_col].shift(1)).dropna()
            returns["volatility_20d_ann"] = round(float(log_rets.tail(20).std() * np.sqrt(252)), 6)

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
                detail=f"Ticker '{ticker}' not in universe",
            )

        def _fetch():
            svc = MarketDataService()
            start_date, end_date = _date_window(days)
            return svc.get_price_data(
                ticker,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                min_history=days,
            )

        df = await run_in_threadpool(_fetch)

        if df is None or df.empty:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}",
            )

        df = df.tail(days)

        records = []
        for dt, row in df.iterrows():
            records.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "open": round(float(row.get("open", row.get("Open", np.nan))), 4),
                "high": round(float(row.get("high", row.get("High", np.nan))), 4),
                "low": round(float(row.get("low", row.get("Low", np.nan))), 4),
                "close": round(float(row.get("close", row.get("Close", np.nan))), 4),
                "volume": int(row.get("volume", row.get("Volume", 0))),
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