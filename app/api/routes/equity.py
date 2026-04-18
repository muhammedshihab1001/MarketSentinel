# =========================================================
# EQUITY ROUTE v2.6
# FIX #21: Removed session_factory=get_session from both
# MarketDataService() calls. MarketDataService.__init__()
# does not accept this kwarg — caused 500 on all price
# chart requests from the Agent page.
# =========================================================

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

from core.data.market_data_service import MarketDataService
from core.db.engine import get_session  # noqa: F401 — kept for other imports
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.equity")

router = APIRouter(tags=["equity"])

DEFAULT_HISTORY_DAYS = 90
MAX_HISTORY_DAYS = 730


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return default if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return default


def _safe_str(val, default="") -> str:
    if val is None:
        return default
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    s = str(val)
    return s[:10] if len(s) >= 10 else s


@router.get(
    "/equity/{ticker}",
    summary="Latest Equity Data",
    description="""
Returns the latest OHLCV row plus calculated returns and volatility
for a single ticker from the PostgreSQL database.

**Requires:** Owner or Demo authentication (counts against `signals` quota).
""",
    response_description="Latest OHLCV + returns + volatility for the ticker.",
)
def get_equity(ticker: str):
    API_REQUEST_COUNT.labels(endpoint="/equity/ticker").inc()
    start_time = time.time()
    ticker = ticker.upper().strip()

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        svc = MarketDataService()  # FIX #21: no session_factory kwarg
        df = svc.get_price_data(ticker, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        latest = df.iloc[-1]
        date_val = latest["date"] if "date" in df.columns else df.index[-1]

        ret_5d = 0.0
        if len(df) >= 6:
            p_now = _safe_float(latest.get("close", latest.get("adj_close")))
            p_5d = _safe_float(df.iloc[-6].get("close", df.iloc[-6].get("adj_close")))
            if p_5d > 0:
                ret_5d = (p_now - p_5d) / p_5d

        ret_20d = 0.0
        if len(df) >= 21:
            p_now = _safe_float(latest.get("close", latest.get("adj_close")))
            p_20d = _safe_float(
                df.iloc[-21].get("close", df.iloc[-21].get("adj_close"))
            )
            if p_20d > 0:
                ret_20d = (p_now - p_20d) / p_20d

        vol_20d = 0.0
        if len(df) >= 21:
            close_col = "close" if "close" in df.columns else "adj_close"
            closes = df[close_col].iloc[-21:].astype(float)
            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) > 1:
                vol_20d = float(daily_returns.std() * np.sqrt(252))
                if not np.isfinite(vol_20d):
                    vol_20d = 0.0

        return {
            "ticker": ticker,
            "ohlcv": {
                "date": _safe_str(date_val),
                "open": _safe_float(latest.get("open")),
                "high": _safe_float(latest.get("high")),
                "low": _safe_float(latest.get("low")),
                "close": _safe_float(latest.get("close", latest.get("adj_close"))),
                "volume": int(_safe_float(latest.get("volume"), 0)),
            },
            "returns": {
                "5d_return": round(ret_5d, 6),
                "20d_return": round(ret_20d, 6),
                "volatility_20d_ann": round(vol_20d, 6),
            },
            "data_source": "postgresql",
            "rows_available": len(df),
        }

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint="/equity/ticker").inc()
        logger.exception("Equity detail failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint="/equity/ticker").observe(time.time() - start_time)


@router.get(
    "/equity/{ticker}/history",
    summary="OHLCV History for Ticker",
    description="""
Returns historical daily OHLCV rows for a ticker from the PostgreSQL database.
Used by the Agent page price chart.

**days:** Number of trading days to return (1–730). Default: 90.
""",
    response_description="Array of daily OHLCV rows oldest to newest.",
)
def get_equity_history(
    ticker: str,
    days: int = Query(
        default=DEFAULT_HISTORY_DAYS,
        ge=1,
        le=MAX_HISTORY_DAYS,
        description="Number of calendar days of history to return (1–730)",
        example=90,
    ),
):
    endpoint = "/equity/ticker/history"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()
    ticker = ticker.upper().strip()

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

        svc = MarketDataService()  # FIX #21: no session_factory kwarg
        df = svc.get_price_data(ticker, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        df = df.tail(days)

        history = []
        for _, row in df.iterrows():
            date_val = row["date"] if "date" in df.columns else row.name
            history.append(
                {
                    "date": _safe_str(date_val),
                    "open": _safe_float(row.get("open")),
                    "high": _safe_float(row.get("high")),
                    "low": _safe_float(row.get("low")),
                    "close": _safe_float(row.get("close", row.get("adj_close"))),
                    "volume": int(_safe_float(row.get("volume"), 0)),
                }
            )

        return {
            "ticker": ticker,
            "days_requested": days,
            "rows_returned": len(history),
            "data_source": "postgresql",
            "history": history,
        }

    except HTTPException:
        raise
    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Equity history failed | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
