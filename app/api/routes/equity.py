# =========================================================
# EQUITY ROUTE v2.4
#
# Changes from v2.3:
# FIX 1: get_price_data() result is a DataFrame with a
#         reset integer index after _validate_dataset.
#         Reading latest.name gives RangeIndex (integer),
#         not a date string. Fix: read row["date"] column.
# FIX 2: /equity/{ticker}/history response aligns to the
#         EquityHistoryResponse type in frontend types/index.ts
# FIX 3: Safe NaN handling before JSON serialisation —
#         np.nan is not JSON-serialisable.
# =========================================================

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from core.data.market_data_service import MarketDataService
from core.db.engine import get_session
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.equity")

router = APIRouter()

DEFAULT_HISTORY_DAYS = 90
MAX_HISTORY_DAYS = 730


def _safe_float(val, default=0.0) -> float:
    """Convert to float, replacing NaN/Inf with default."""
    try:
        v = float(val)
        return default if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return default


def _safe_str(val, default="") -> str:
    """Convert date-like value to YYYY-MM-DD string."""
    if val is None:
        return default
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    s = str(val)
    return s[:10] if len(s) >= 10 else s


# =========================================================
# GET /equity/{ticker}
# Latest OHLCV + returns + volatility
# =========================================================

@router.get("/equity/{ticker}")
def get_equity(ticker: str):
    """
    Returns the latest OHLCV row plus calculated returns and
    volatility for a single ticker.
    """
    endpoint = f"/equity/{ticker}"
    API_REQUEST_COUNT.labels(endpoint="/equity/ticker").inc()
    start_time = time.time()

    ticker = ticker.upper().strip()

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        svc = MarketDataService(session_factory=get_session)

        df = svc.get_price_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        # FIX: After reset_index(drop=True) the index is RangeIndex.
        # Read date from the "date" column, not from index.
        latest = df.iloc[-1]

        # Safe date extraction
        if "date" in df.columns:
            date_val = latest["date"]
        else:
            date_val = df.index[-1]

        date_str = _safe_str(date_val)

        # 5-day return
        ret_5d = 0.0
        if len(df) >= 6:
            price_now = _safe_float(latest.get("close", latest.get("adj_close")))
            price_5d = _safe_float(
                df.iloc[-6].get("close", df.iloc[-6].get("adj_close"))
            )
            if price_5d > 0:
                ret_5d = (price_now - price_5d) / price_5d

        # 20-day return
        ret_20d = 0.0
        if len(df) >= 21:
            price_now = _safe_float(latest.get("close", latest.get("adj_close")))
            price_20d = _safe_float(
                df.iloc[-21].get("close", df.iloc[-21].get("adj_close"))
            )
            if price_20d > 0:
                ret_20d = (price_now - price_20d) / price_20d

        # 20-day annualised volatility
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
                "date": date_str,
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
        API_LATENCY.labels(endpoint="/equity/ticker").observe(
            time.time() - start_time
        )


# =========================================================
# GET /equity/{ticker}/history
# OHLCV history array
# =========================================================

@router.get("/equity/{ticker}/history")
def get_equity_history(
    ticker: str,
    days: int = Query(default=DEFAULT_HISTORY_DAYS, ge=1, le=MAX_HISTORY_DAYS),
):
    """
    Returns historical OHLCV rows for a ticker.
    Default: 90 days. Max: 730 days.
    """
    endpoint = "/equity/ticker/history"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    ticker = ticker.upper().strip()

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

        svc = MarketDataService(session_factory=get_session)

        df = svc.get_price_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        # Trim to requested days
        df = df.tail(days)

        history = []
        for _, row in df.iterrows():
            # FIX: Read date from column, not from index
            if "date" in df.columns:
                date_val = row["date"]
            else:
                date_val = row.name

            history.append({
                "date": _safe_str(date_val),
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "close": _safe_float(row.get("close", row.get("adj_close"))),
                "volume": int(_safe_float(row.get("volume"), 0)),
            })

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