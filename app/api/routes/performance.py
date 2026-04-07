# =========================================================
# PERFORMANCE ROUTE v2.6
#
# SWAGGER FIX v2.6:
# - tickers and days now use Query() with description,
#   example, ge/le constraints — visible in Swagger UI
# - Added summary and description to both endpoints
# - per-ticker endpoint documents path param
# =========================================================

import pandas as pd
import numpy as np
import asyncio
import time
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from core.data.market_data_service import MarketDataService
from core.market.universe import MarketUniverse
from core.analytics.performance_engine import PerformanceEngine
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.performance")

router = APIRouter(tags=["performance"])

TRADING_DAYS = 252


def _date_window(days: int):
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days + 60)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _build_inputs(price_map: dict, days: int):
    close_frames = []

    for ticker, df in price_map.items():
        if df is None or df.empty:
            continue
        col = "close" if "close" in df.columns else "Close"
        if col not in df.columns:
            continue
        s = df[["date", col]].copy().rename(columns={col: "close"})
        s["ticker"] = ticker
        s = s.tail(days + 10)
        close_frames.append(s)

    if not close_frames:
        return None, None

    prices = pd.concat(close_frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.normalize()
    prices = prices.sort_values(["ticker", "date"])

    prices["forward_return"] = (
        prices.groupby("ticker")["close"]
        .pct_change()
        .shift(-1)
        .clip(-0.5, 0.5)
    )

    prices = prices.dropna(subset=["forward_return"])

    all_dates = prices["date"].drop_duplicates().sort_values()
    if len(all_dates) > days:
        cutoff = all_dates.iloc[-days]
        prices = prices[prices["date"] >= cutoff]

    n_per_date = prices.groupby("date")["ticker"].transform("count")
    prices["weight"] = 1.0 / n_per_date

    portfolio_df = prices[["date", "ticker", "weight"]].copy()
    forward_returns = prices[["date", "ticker", "forward_return"]].copy()

    return portfolio_df, forward_returns


# =========================================================
# GET /performance
# =========================================================

@router.get(
    "/performance",
    summary="Portfolio Performance Metrics",
    description="""
Returns institutional-grade performance metrics for the universe portfolio
over the requested lookback window.

**Metrics include:**
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown and drawdown duration
- Rolling beta, tracking error, information ratio
- Compound annual growth rate (CAGR)

**tickers** (optional): Comma-separated list to filter, e.g. `AAPL,NVDA,MSFT`.
Leave blank to compute for the full 100-ticker universe.

**days**: Trading days lookback. `252` = 1 year, `126` = 6 months.

**Requires:** Owner or Demo authentication.
""",
    response_description="Performance report with institutional metrics.",
)
async def performance_summary(
    tickers: str = Query(
        default="",
        description="Comma-separated tickers to include. Leave blank for full universe.",
        example="AAPL,NVDA,MSFT,GOOGL,JPM",
    ),
    days: int = Query(
        default=252,
        ge=30,
        le=500,
        description="Lookback window in trading days. 252 = 1 year.",
        example=252,
    ),
):
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
                    detail="No valid tickers in request. Check /universe for valid tickers.",
                )
        else:
            ticker_list = all_tickers

        def _fetch_and_evaluate():
            svc = MarketDataService()
            start_date, end_date = _date_window(days)

            price_map, errors = svc.get_price_data_batch(
                ticker_list,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                min_history=days,
            )

            if not price_map:
                return None, errors

            portfolio_df, forward_returns = _build_inputs(price_map, days)

            if portfolio_df is None or portfolio_df.empty:
                return None, errors

            engine = PerformanceEngine()
            report = engine.evaluate(
                portfolio_df=portfolio_df,
                forward_returns=forward_returns,
                benchmark_returns=None,
            )

            return report.to_dict(), errors

        result_dict, errors = await run_in_threadpool(_fetch_and_evaluate)

        if result_dict is None:
            raise HTTPException(
                status_code=503,
                detail="No price data available for performance computation",
            )

        response = {
            "tickers_requested": len(ticker_list),
            "tickers_computed": len(ticker_list) - len(errors),
            "lookback_days": days,
            "data_source": "postgresql",
            "metrics": result_dict,
        }

        if errors:
            response["fetch_errors"] = errors

        return response

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Performance endpoint failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# =========================================================
# GET /performance/{ticker}
# =========================================================

@router.get(
    "/performance/{ticker}",
    summary="Per-Ticker Performance Metrics",
    description="""
Returns performance metrics for a single ticker over the lookback window.

**Example tickers:** AAPL, NVDA, MSFT, GOOGL, JPM, AMZN, TSLA

Same metrics as the portfolio endpoint but computed for a single stock.
Useful for comparing individual ticker performance against the portfolio.

**Requires:** Owner or Demo authentication.
""",
    response_description="Performance metrics for the requested ticker.",
)
async def ticker_performance(
    ticker: str,
    days: int = Query(
        default=252,
        ge=30,
        le=500,
        description="Lookback window in trading days. 252 = 1 year.",
        example=252,
    ),
):
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
                detail=f"Ticker '{ticker}' not in universe. Check GET /universe.",
            )

        def _fetch_and_evaluate():
            svc = MarketDataService()
            start_date, end_date = _date_window(days)

            df = svc.get_price_data(
                ticker,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                min_history=days,
            )

            if df is None or df.empty:
                return None

            col = "close" if "close" in df.columns else "Close"
            prices = df[["date", col]].copy().rename(columns={col: "close"})
            prices["ticker"] = ticker
            prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.normalize()
            prices = prices.sort_values("date").tail(days + 10)

            prices["forward_return"] = (
                prices["close"].pct_change().shift(-1).clip(-0.5, 0.5)
            )
            prices = prices.dropna(subset=["forward_return"])
            prices["weight"] = 1.0

            portfolio_df = prices[["date", "ticker", "weight"]].copy()
            forward_returns = prices[["date", "ticker", "forward_return"]].copy()

            engine = PerformanceEngine()
            report = engine.evaluate(
                portfolio_df=portfolio_df,
                forward_returns=forward_returns,
                benchmark_returns=None,
            )

            return report.to_dict()

        result_dict = await run_in_threadpool(_fetch_and_evaluate)

        if result_dict is None:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}",
            )

        return {
            "ticker": ticker,
            "lookback_days": days,
            "data_source": "postgresql",
            "metrics": result_dict,
        }

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Ticker performance failure | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
