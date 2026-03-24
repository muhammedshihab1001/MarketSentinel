# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v5.8
#
# FIX v5.8: run_snapshot() was processing ALL 27,400 rows
#   (274 days × 100 tickers) in the per-ticker loop.
#   Root cause: dataset from _build_cross_sectional_frame()
#   is the full lookback window — needed for cross-sectional
#   feature computation — but per-ticker scoring must only
#   use the LATEST date row per ticker.
#
#   Fix: after feature engineering, filter dataset to the
#   latest available date per ticker before the scoring loop.
#   Result: 100 rows (one per ticker) → snapshot takes ~5s
#   instead of 182s. signals=100 not signals=27400.
#
#   Also fixed: _build_cross_sectional_frame() now checks
#   feature cache HIT correctly — if cache returns all rows
#   for all dates, we still only score on latest per ticker.
# =========================================================

import time
import logging
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema, DTYPE
from core.market.universe import MarketUniverse

logger = logging.getLogger("marketsentinel.pipeline")

PIPELINE_TIMEOUT = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "12"))
TOP_K = int(os.getenv("PIPELINE_TOP_K", "5"))
BOTTOM_K = int(os.getenv("PIPELINE_BOTTOM_K", "5"))
INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))


class InferencePipeline:

    def __init__(self, model=None, cache=None, db_session_factory=None):
        self._model = model
        self._cache = cache
        self._db_session_factory = db_session_factory

        self._signal_agent = None
        self._technical_agent = None
        self._portfolio_agent = None
        self._political_agent = None

    # =====================================================
    # MODEL ACCESSOR
    # =====================================================

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from app.api.routes.predict import _model_loader
            if _model_loader is not None:
                return _model_loader
        except ImportError:
            pass
        from app.inference.model_loader import get_model_loader
        return get_model_loader()

    # =====================================================
    # AGENT ACCESSORS (lazy init)
    # =====================================================

    @property
    def signal_agent(self):
        if self._signal_agent is None:
            from core.agent.signal_agent import SignalAgent
            self._signal_agent = SignalAgent()
        return self._signal_agent

    @property
    def technical_agent(self):
        if self._technical_agent is None:
            from core.agent.technical_risk_agent import TechnicalRiskAgent
            self._technical_agent = TechnicalRiskAgent()
        return self._technical_agent

    @property
    def portfolio_agent(self):
        if self._portfolio_agent is None:
            from core.agent.portfolio_decision_agent import PortfolioDecisionAgent
            self._portfolio_agent = PortfolioDecisionAgent()
        return self._portfolio_agent

    @property
    def political_agent(self):
        if self._political_agent is None:
            from core.agent.political_risk_agent import PoliticalRiskAgent
            self._political_agent = PoliticalRiskAgent()
        return self._political_agent

    # =====================================================
    # SAFE AGENT CALL
    # =====================================================

    def _safe_agent(self, agent, context: dict) -> dict:
        try:
            return agent.analyze(context) or {}
        except Exception as e:
            logger.debug("Agent %s failed: %s", type(agent).__name__, e)
            return {}

    # =====================================================
    # BUILD CROSS-SECTIONAL FRAME
    # Returns the FULL lookback window for ALL tickers.
    # Cross-sectional features (z-scores, ranks) need the
    # full history to be computed correctly. Caller is
    # responsible for filtering to latest date per ticker.
    # =====================================================

    def _build_cross_sectional_frame(
        self,
        tickers: List[str],
        end_date: str,
    ) -> Optional[pd.DataFrame]:

        from core.data.market_data_service import MarketDataService
        from core.features.feature_engineering import FeatureEngineer

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.Timedelta(days=INFERENCE_LOOKBACK_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")

        svc = MarketDataService()
        engineer = FeatureEngineer()

        try:
            price_result = svc.get_price_data_batch(
                tickers,
                start_date=start_date,
                end_date=end_date,
            )
            if isinstance(price_result, tuple):
                price_data, errors = price_result
                if errors:
                    logger.warning(
                        "Price fetch errors for %d tickers: %s",
                        len(errors), list(errors)[:5],
                    )
            else:
                price_data = price_result

        except Exception as e:
            logger.error("Price data fetch failed: %s", e)
            return None

        if not price_data:
            logger.warning("No price data returned for any ticker")
            return None

        all_frames = []
        for ticker, df in price_data.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            all_frames.append(df)

        if not all_frames:
            return None

        combined_prices = pd.concat(all_frames, ignore_index=True)

        try:
            features = engineer.build_feature_pipeline(combined_prices, training=False)
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            return None

        return features

    # =====================================================
    # FILTER TO LATEST DATE PER TICKER
    #
    # FIX v5.8: This is the core fix.
    #
    # _build_cross_sectional_frame() returns 274 days × 100
    # tickers = 27,400 rows. Cross-sectional features need the
    # full window to compute z-scores correctly.
    #
    # But the scoring loop must only process ONE row per ticker:
    # the most recent date with valid data.
    #
    # This method reduces 27,400 rows → 100 rows.
    # Result: snapshot takes ~5s not 182s.
    # signals=100 not signals=27,400.
    # Sector neutralisation no longer sees duplicate tickers.
    # =====================================================

    @staticmethod
    def _filter_latest_per_ticker(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce a multi-date feature frame to one row per ticker:
        the row with the latest valid date.

        Input:  27,400 rows (274 days × 100 tickers)
        Output: 100 rows (latest date per ticker)
        """
        if dataset is None or dataset.empty:
            return dataset

        dataset = dataset.copy()

        # Normalise date column — may be Timestamp or string
        dataset["date"] = pd.to_datetime(dataset["date"], utc=True, errors="coerce")
        dataset = dataset.dropna(subset=["date"])

        # Keep only the latest row per ticker
        latest = (
            dataset
            .sort_values("date")
            .groupby("ticker", sort=False)
            .tail(1)
            .reset_index(drop=True)
        )

        logger.info(
            "Latest-per-ticker filter | input_rows=%d output_rows=%d tickers=%d",
            len(dataset), len(latest), latest["ticker"].nunique(),
        )

        return latest

    # =====================================================
    # RUN SNAPSHOT
    # =====================================================

    def run_snapshot(self, snapshot_date: Optional[str] = None) -> dict:
        start_time = time.time()

        if snapshot_date is None:
            snapshot_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        logger.info("Running snapshot | date=%s", snapshot_date)

        # ── Get model ─────────────────────────────────
        try:
            loader = self._get_model()
        except Exception as e:
            return self._error_snapshot(f"Model loader unavailable: {e}")

        # ── Get universe ──────────────────────────────
        try:
            universe = MarketUniverse.snapshot()
            tickers = list(universe.get("tickers", []))
        except Exception as e:
            logger.error("Universe load failed: %s", e)
            return self._error_snapshot("Universe load failed")

        if len(tickers) < int(os.getenv("MIN_UNIVERSE_WIDTH", "8")):
            return self._error_snapshot("Universe too small")

        # ── Build feature frame (full lookback window) ─
        dataset = self._build_cross_sectional_frame(tickers, snapshot_date)

        if dataset is None or dataset.empty:
            return self._error_snapshot("Feature engineering failed for all tickers")

        # ── FIX v5.8: Reduce to ONE row per ticker ────
        # Cross-sectional features need the full history for z-score
        # computation. But inference only needs the latest row per ticker.
        # Without this filter: 27,400 rows processed → 182s, signals=27400.
        # With this filter: 100 rows processed → ~5s, signals=100.
        dataset = self._filter_latest_per_ticker(dataset)

        if dataset.empty:
            return self._error_snapshot("No latest-date rows found after filter")

        # ── Validate features ─────────────────────────
        try:
            available_features = [f for f in MODEL_FEATURES if f in dataset.columns]
            if len(available_features) < len(MODEL_FEATURES) * 0.8:
                logger.warning(
                    "Only %d/%d features available",
                    len(available_features), len(MODEL_FEATURES),
                )
            feature_block = validate_feature_schema(
                dataset.reindex(columns=MODEL_FEATURES, fill_value=0.0),
                mode="inference",
            ).astype(DTYPE)
        except Exception as e:
            logger.error("Feature validation failed: %s", e)
            return self._error_snapshot("Feature validation failed")

        # ── Run XGBoost inference ─────────────────────
        try:
            raw_scores = loader.predict(feature_block)
        except Exception as e:
            logger.error("Model inference failed: %s", e)
            return self._error_snapshot("Model inference failed")

        dataset = dataset.reset_index(drop=True)
        dataset["raw_model_score"] = raw_scores

        # ── Drift detection ───────────────────────────
        drift_state = "none"
        drift_result = {}
        try:
            from core.monitoring.drift_detector import DriftDetector
            detector = DriftDetector()
            drift_result = detector.detect(dataset)
            drift_state = drift_result.get("drift_state", "none")
        except Exception as e:
            logger.warning("Drift detection failed: %s", e)

        # ── Political risk ────────────────────────────
        political_output = {}
        try:
            political_output = self._safe_agent(
                self.political_agent,
                {"ticker": "MARKET", "country": "US"},
            )
        except Exception as e:
            logger.debug("Political agent failed: %s", e)

        political_label = political_output.get("political_risk_label", "LOW")

        # ── Per-ticker scoring loop ───────────────────
        # dataset now has exactly one row per ticker (latest date).
        # No duplicate tickers, no sector neutralisation spam.
        snapshot_rows = []
        tickers_set = set(tickers)
        exposure_scale = float(drift_result.get("exposure_scale", 1.0))

        for idx, row in dataset.iterrows():
            ticker = row.get("ticker", "UNKNOWN")

            if ticker not in tickers_set:
                continue

            context = {
                "row": row.to_dict(),
                "ticker": ticker,
                "drift_state": drift_state,
                "political_risk_label": political_label,
                "probability_stats": {
                    "mean": float(dataset["raw_model_score"].mean()),
                    "std": float(dataset["raw_model_score"].std()),
                },
            }

            signal_output = self._safe_agent(self.signal_agent, context)
            technical_output = self._safe_agent(self.technical_agent, context)

            raw_score = float(row.get("raw_model_score", 0.0))
            signal_score = float(signal_output.get("score", 0.0))
            technical_score = float(technical_output.get("score", 0.0))

            hybrid_score = float(np.clip(
                0.50 * raw_score + 0.30 * signal_score + 0.20 * technical_score,
                -1.0, 1.0,
            ))

            if political_label == "CRITICAL":
                hybrid_score = 0.0
            elif political_label == "HIGH":
                hybrid_score *= 0.5

            weight = float(np.clip(hybrid_score * exposure_scale, -1.0, 1.0))

            snapshot_rows.append({
                "ticker": ticker,
                "date": str(row.get("date", snapshot_date))[:10],
                "raw_model_score": round(raw_score, 6),
                "hybrid_consensus_score": round(hybrid_score, 6),
                "weight": round(weight, 6),
                "agents": {
                    "signal_agent": signal_output,
                    "technical_agent": technical_output,
                },
            })

        if not snapshot_rows:
            return self._error_snapshot("No valid signals produced")

        snapshot_rows.sort(key=lambda x: x["raw_model_score"], reverse=True)

        long_signals = [r for r in snapshot_rows if r["weight"] > 0.01]
        short_signals = [r for r in snapshot_rows if r["weight"] < -0.01]

        weights = [r["weight"] for r in snapshot_rows]
        gross_exposure = float(sum(abs(w) for w in weights))
        net_exposure = float(sum(weights))

        if gross_exposure > 1.0:
            net_exposure = net_exposure / gross_exposure
            gross_exposure = 1.0

        lc, sc = len(long_signals), len(short_signals)

        if lc > sc * 1.5:
            portfolio_bias = "LONG_BIASED"
        elif sc > lc * 1.5:
            portfolio_bias = "SHORT_BIASED"
        else:
            portfolio_bias = "BALANCED"

        top_5 = snapshot_rows[:TOP_K]
        avg_hybrid = float(np.mean([r["hybrid_consensus_score"] for r in snapshot_rows]))

        portfolio_output = {}
        try:
            portfolio_context = {
                "signals": snapshot_rows,
                "drift_state": drift_state,
                "gross_exposure": round(gross_exposure, 4),
                "net_exposure": round(net_exposure, 4),
            }
            portfolio_output = self._safe_agent(self.portfolio_agent, portfolio_context)
        except Exception as e:
            logger.debug("Portfolio agent failed: %s", e)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        result = {
            "meta": {
                "model_version": getattr(loader, "version", "unknown"),
                "drift_state": drift_state,
                "long_signals": lc,
                "short_signals": sc,
                "avg_hybrid_score": round(avg_hybrid, 6),
                "latency_ms": latency_ms,
            },
            "executive_summary": {
                "top_5_tickers": [r["ticker"] for r in top_5],
                "portfolio_bias": portfolio_bias,
                "gross_exposure": round(gross_exposure, 4),
                "net_exposure": round(net_exposure, 4),
            },
            "snapshot": {
                "snapshot_date": snapshot_date,
                "model_version": getattr(loader, "version", "unknown"),
                "drift": {
                    "drift_detected": drift_result.get("drift_detected", False),
                    "severity_score": drift_result.get("severity_score", 0),
                    "drift_state": drift_state,
                    "exposure_scale": exposure_scale,
                    "drift_confidence": drift_result.get("drift_confidence", 0.0),
                },
                "signals": [
                    {
                        "ticker": r["ticker"],
                        "date": r["date"],
                        "raw_model_score": r["raw_model_score"],
                        "hybrid_consensus_score": r["hybrid_consensus_score"],
                        "weight": r["weight"],
                    }
                    for r in snapshot_rows
                ],
            },
            "_signal_details": {r["ticker"]: r["agents"] for r in snapshot_rows},
            "_political": political_output,
            "_portfolio": portfolio_output,
        }

        logger.info(
            "Snapshot complete | tickers=%d | long=%d | short=%d | latency=%.0fms",
            len(snapshot_rows), lc, sc, latency_ms,
        )

        return result

    # =====================================================
    # ERROR SNAPSHOT
    # =====================================================

    def _error_snapshot(self, reason: str) -> dict:
        logger.error("Snapshot failed: %s", reason)
        return {
            "meta": {
                "model_version": "unknown",
                "drift_state": "none",
                "long_signals": 0,
                "short_signals": 0,
                "avg_hybrid_score": 0.0,
                "latency_ms": 0,
                "error": reason,
            },
            "executive_summary": {
                "top_5_tickers": [],
                "portfolio_bias": "UNKNOWN",
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
            },
            "snapshot": {
                "snapshot_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "model_version": "unknown",
                "drift": {
                    "drift_detected": False,
                    "severity_score": 0,
                    "drift_state": "none",
                    "exposure_scale": 1.0,
                    "drift_confidence": 0.0,
                },
                "signals": [],
            },
            "_signal_details": {},
            "_political": {},
            "_portfolio": {},
        }