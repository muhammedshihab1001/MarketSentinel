# =========================================================
# PERFORMANCE ROUTE v2.4
# FIX: PerformanceEngine uses evaluate(), not compute()
#      evaluate() needs portfolio_df + forward_returns format.
#      We compute simple daily returns from close prices instead.
# =========================================================

import pandas as pd
import numpy as np
import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

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

TRADING_DAYS = 252
EPSILON = 1e-12


def _date_window(days: int):
    """Compute start/end date strings from lookback days."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days + 30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _compute_metrics(close_series: pd.Series) -> dict:
    """
    Compute performance metrics directly from a close price series.
    PerformanceEngine.evaluate() requires portfolio_df + forward_returns
    which we don't have here — so we compute metrics directly.
    """

    close_series = close_series.dropna()

    if len(close_series) < 5:
        raise RuntimeError("Insufficient price data for metrics computation")

    daily_returns = close_series.pct_change().dropna()
    daily_returns = daily_returns.clip(-0.5, 0.5)

    # Cumulative return
    equity = (1 + daily_returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)

    # Sharpe ratio
    std = daily_returns.std(ddof=1)
    sharpe = float((daily_returns.mean() / (std + EPSILON)) * np.sqrt(TRADING_DAYS))

    # Sortino ratio
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float((daily_returns.mean() / (downside_std + EPSILON)) * np.sqrt(TRADING_DAYS))

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / (rolling_max + EPSILON)
    max_drawdown = float(drawdown.min())

    # Annualised volatility
    volatility_ann = float(std * np.sqrt(TRADING_DAYS))

    # Annualised return
    years = len(daily_returns) / TRADING_DAYS
    ann_return = float((1 + cumulative_return) ** (1.0 / max(years, 0.01)) - 1.0) if cumulative_return > -1 else -1.0

    # Calmar ratio
    calmar = float(ann_return / abs(max_drawdown)) if abs(max_drawdown) > EPSILON else 0.0

    # Win rate (hit rate)
    win_rate = float((daily_returns > 0).mean())

    return {
        "cumulative_return": round(cumulative_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_drawdown, 6),
        "volatility_ann": round(volatility_ann, 6),
        "win_rate": round(win_rate, 4),
    }


# =========================================================
# PERFORMANCE SUMMARY ENDPOINT
# =========================================================

@router.get("/performance")
async def performance_summary(tickers: str = "", days: int = 252):

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
                raise HTTPException(status_code=400, detail="No valid tickers in request")
        else:
            ticker_list = all_tickers

        def _fetch_and_compute():
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

            # Build equal-weight portfolio daily returns
            close_series_list = []
            for t, df in price_map.items():
                if df is not None and not df.empty:
                    col = "close" if "close" in df.columns else "Close"
                    if col in df.columns:
                        s = df[col].tail(days)
                        if len(s) > 5:
                            close_series_list.append(s.pct_change().dropna())

            if not close_series_list:
                return None, errors

            # Equal-weight average daily return across tickers
            close_matrix = pd.concat(close_series_list, axis=1).dropna(how="all")
            portfolio_returns = close_matrix.mean(axis=1)

            # Rebuild close series from returns for metric computation
            synthetic_close = (1 + portfolio_returns).cumprod()

            metrics = _compute_metrics(synthetic_close)
            return metrics, errors

        result_metrics, errors = await run_in_threadpool(_fetch_and_compute)

        if result_metrics is None:
            raise HTTPException(
                status_code=503,
                detail="No price data available for performance computation",
            )

        result = {
            "tickers_requested": len(ticker_list),
            "tickers_computed": len(ticker_list) - len(errors),
            "lookback_days": days,
            "data_source": "postgresql",
            "metrics": result_metrics,
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
                detail=f"Ticker '{ticker}' not in universe",
            )

        def _fetch_and_compute():
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
            close = df[col].tail(days)
            return _compute_metrics(close)

        metrics = await run_in_threadpool(_fetch_and_compute)

        if metrics is None:
            raise HTTPException(
                status_code=503,
                detail=f"No price data available for {ticker}",
            )

        return {
            "ticker": ticker,
            "lookback_days": days,
            "data_source": "postgresql",
            "metrics": metrics,
        }

    except HTTPException:
        raise

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Ticker performance failure | ticker=%s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)