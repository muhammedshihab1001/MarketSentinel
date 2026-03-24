# =========================================================
# MODEL INFO ROUTE v2.5
#
# FIX: All imports corrected (get_model_loader, loader.version,
#      loader.model, loader.metadata)
# NEW (item 60): GET /model/ic-stats — Information Coefficient
#   monitoring endpoint. Computes Spearman IC between model's
#   raw_model_score predictions and actual 1-day forward returns.
#   IC > 0.05 is meaningful. IC < 0 means the model is predicting
#   against actual moves. Exposed so you can track signal decay.
# =========================================================

import asyncio
import time
import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any, Optional

from app.inference.model_loader import get_model_loader
from core.schema.feature_schema import MODEL_FEATURES
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)

router = APIRouter()
logger = logging.getLogger("marketsentinel.model_info")

REQUEST_TIMEOUT = 20
MAX_CONCURRENT = 4

model_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# =========================================================
# MODEL INFO  →  GET /model/info
# =========================================================

@router.get("/info")
async def model_info():

    endpoint = "/model/info"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_model_info_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Model info timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Model info retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _model_info_sync():

    start_time = time.time()
    loader = get_model_loader()
    meta = loader.metadata or {}

    return {
        "model_version": loader.version or "unknown",
        "schema_signature": loader.schema_signature or "unknown",
        "dataset_hash": meta.get("dataset_hash", "unknown"),
        "training_code_hash": meta.get("training_code_hash", "unknown"),
        "artifact_hash": loader.artifact_hash or "unknown",
        "feature_checksum": meta.get("feature_checksum", "unknown"),
        "feature_count": len(MODEL_FEATURES),
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# FEATURE IMPORTANCE  →  GET /model/feature-importance
# =========================================================

@router.get("/feature-importance")
async def feature_importance():

    endpoint = "/model/feature-importance"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_feature_importance_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Feature importance timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Feature importance retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _feature_importance_sync():

    start_time = time.time()
    loader = get_model_loader()
    model = loader.model
    meta = loader.metadata or {}

    importance = []
    try:
        if model is not None and hasattr(model, "export_feature_importance"):
            raw = model.export_feature_importance()
            importance = [
                {"feature": item["feature"], "importance": float(item["importance"])}
                for item in raw.get("feature_importance", [])
            ]
        elif model is not None and hasattr(model, "model") and model.model is not None:
            scores = model.model.get_score(importance_type="gain")
            total = sum(scores.values()) or 1.0
            importance = sorted(
                [{"feature": f, "importance": round(v / total, 6)} for f, v in scores.items()],
                key=lambda x: x["importance"],
                reverse=True,
            )
    except Exception as e:
        logger.warning("Feature importance extraction failed: %s", e)

    return {
        "model_version": loader.version or "unknown",
        "feature_checksum": meta.get("feature_checksum", "unknown"),
        "best_iteration": getattr(model, "best_iteration", None) if model else None,
        "training_fingerprint": getattr(model, "training_fingerprint", None) if model else None,
        "importance": importance,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# MODEL DIAGNOSTICS  →  GET /model/diagnostics
# =========================================================

@router.get("/diagnostics")
async def model_diagnostics() -> Dict[str, Any]:

    endpoint = "/model/diagnostics"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_model_diagnostics_sync),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="Model diagnostics timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Model diagnostics failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _model_diagnostics_sync():

    start_time = time.time()
    loader = get_model_loader()
    model = loader.model
    meta = loader.metadata or {}

    return {
        "model_version": loader.version or "unknown",
        "artifact_hash": loader.artifact_hash or "unknown",
        "schema_signature": loader.schema_signature or "unknown",
        "dataset_hash": meta.get("dataset_hash", "unknown"),
        "training_code_hash": meta.get("training_code_hash", "unknown"),
        "feature_checksum": meta.get("feature_checksum", "unknown"),
        "feature_count": len(MODEL_FEATURES),
        "training_fingerprint": getattr(model, "training_fingerprint", None) if model else None,
        "training_cols": getattr(model, "training_cols", None) if model else None,
        "param_checksum": getattr(model, "param_checksum", None) if model else None,
        "booster_checksum": getattr(model, "booster_checksum", None) if model else None,
        "best_iteration": getattr(model, "best_iteration", None) if model else None,
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }


# =========================================================
# IC STATS  →  GET /model/ic-stats  (item 60)
#
# Information Coefficient monitoring.
# Computes Spearman rank correlation between model's
# raw_model_score predictions and actual 1-day forward returns.
#
# Interpretation:
#   IC > 0.05  → meaningful predictive signal
#   IC ≈ 0     → random (model not predicting direction)
#   IC < 0     → model predicting against actual moves (decay)
#
# Reads stored predictions from PredictionRepository and
# matches them against actual price returns from DB.
# =========================================================

@router.get("/ic-stats")
async def ic_stats(
    days: int = Query(30, ge=5, le=252, description="Lookback days for IC calculation"),
):

    endpoint = "/model/ic-stats"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        async with model_semaphore:
            result = await asyncio.wait_for(
                run_in_threadpool(_ic_stats_sync, days),
                timeout=REQUEST_TIMEOUT,
            )
        return result

    except asyncio.TimeoutError:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=504, detail="IC stats timeout")

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("IC stats failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def _ic_stats_sync(days: int) -> Dict[str, Any]:
    """
    Compute Information Coefficient from stored predictions + actual returns.

    Steps:
    1. Load stored predictions from PredictionRepository
    2. Load actual price data for the same tickers + dates
    3. Compute 1-day forward return for each ticker-date
    4. Spearman rank correlation between raw_model_score and forward_return
    5. Return per-day IC and rolling stats
    """

    import pandas as pd
    import numpy as np
    from scipy import stats as scipy_stats

    start_time = time.time()

    # ── Step 1: Load stored predictions ───────────────────
    try:
        from core.db.repository import PredictionRepository
        import datetime

        ic_rows = []
        end_date = datetime.date.today()

        for i in range(days):
            date = (end_date - datetime.timedelta(days=i)).isoformat()
            preds_df = PredictionRepository.get_predictions(date=date)
            if preds_df is not None and not preds_df.empty:
                ic_rows.append(preds_df)

        if not ic_rows:
            return {
                "status": "no_predictions",
                "message": "No stored predictions found. Predictions are stored when STORE_PREDICTIONS=1.",
                "ic_mean": None,
                "ic_std": None,
                "ic_t_stat": None,
                "daily_ic": [],
                "lookback_days": days,
                "latency_ms": int((time.time() - start_time) * 1000),
            }

        predictions = pd.concat(ic_rows, ignore_index=True)

    except Exception as e:
        logger.warning("Could not load predictions for IC: %s", e)
        return {
            "status": "error",
            "message": f"Prediction load failed: {e}",
            "ic_mean": None,
            "ic_std": None,
            "daily_ic": [],
            "lookback_days": days,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    # ── Step 2: Load actual returns ────────────────────────
    try:
        from core.data.market_data_service import MarketDataService

        tickers = predictions["ticker"].unique().tolist()
        dates = pd.to_datetime(predictions["date"])
        start_date = (dates.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_date_str = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

        svc = MarketDataService()
        price_map, _ = svc.get_price_data_batch(
            tickers,
            start_date=start_date,
            end_date=end_date_str,
            interval="1d",
            min_history=5,
        )

        # Build forward returns DataFrame
        return_frames = []
        for ticker, df in (price_map or {}).items():
            if df is None or df.empty:
                continue
            col = "close" if "close" in df.columns else "Close"
            s = df[["date", col]].copy().rename(columns={col: "close"})
            s["ticker"] = ticker
            s["date"] = pd.to_datetime(s["date"]).dt.date.astype(str)
            s["forward_return"] = s["close"].pct_change().shift(-1).clip(-0.5, 0.5)
            return_frames.append(s[["date", "ticker", "forward_return"]].dropna())

        if not return_frames:
            raise RuntimeError("No price data for IC computation")

        actual_returns = pd.concat(return_frames, ignore_index=True)

    except Exception as e:
        logger.warning("Could not load actual returns for IC: %s", e)
        return {
            "status": "error",
            "message": f"Return data load failed: {e}",
            "ic_mean": None,
            "ic_std": None,
            "daily_ic": [],
            "lookback_days": days,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    # ── Step 3: Merge and compute IC per date ──────────────
    predictions["date"] = predictions["date"].astype(str)
    merged = predictions.merge(actual_returns, on=["date", "ticker"], how="inner")

    if len(merged) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(merged)} matched prediction-return pairs. Need >= 10.",
            "ic_mean": None,
            "ic_std": None,
            "daily_ic": [],
            "lookback_days": days,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    # ── Step 4: Spearman IC per date ──────────────────────
    daily_ic = []

    for date, group in merged.groupby("date"):
        if len(group) < 5:
            continue
        try:
            ic, pval = scipy_stats.spearmanr(
                group["raw_model_score"],
                group["forward_return"],
            )
            daily_ic.append({
                "date": date,
                "ic": round(float(ic), 6),
                "p_value": round(float(pval), 6),
                "n_stocks": len(group),
            })
        except Exception:
            pass

    if not daily_ic:
        return {
            "status": "computation_failed",
            "message": "IC computation returned no results.",
            "ic_mean": None,
            "ic_std": None,
            "daily_ic": [],
            "lookback_days": days,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    ic_values = [row["ic"] for row in daily_ic]
    ic_mean = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0

    # t-stat: tests if IC is significantly different from 0
    ic_t_stat = (
        float(ic_mean / (ic_std / np.sqrt(len(ic_values)) + 1e-9))
        if ic_std > 0
        else 0.0
    )

    # Signal quality label
    if abs(ic_mean) >= 0.08:
        signal_quality = "strong"
    elif abs(ic_mean) >= 0.04:
        signal_quality = "moderate"
    elif abs(ic_mean) >= 0.02:
        signal_quality = "weak"
    else:
        signal_quality = "noise"

    return {
        "status": "ok",
        "model_version": get_model_loader().version or "unknown",
        "lookback_days": days,
        "n_days": len(daily_ic),
        "n_predictions_total": len(merged),
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "ic_t_stat": round(ic_t_stat, 4),
        "signal_quality": signal_quality,
        "interpretation": {
            "strong":   "IC > 0.08: strong predictive signal",
            "moderate": "IC 0.04-0.08: meaningful signal",
            "weak":     "IC 0.02-0.04: weak signal",
            "noise":    "IC < 0.02: near-random, investigate model",
        }[signal_quality],
        "daily_ic": sorted(daily_ic, key=lambda x: x["date"], reverse=True),
        "latency_ms": int((time.time() - start_time) * 1000),
        "timestamp": int(time.time()),
    }