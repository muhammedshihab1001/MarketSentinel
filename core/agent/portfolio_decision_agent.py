"""
MarketSentinel v4.3.0

Portfolio Decision Agent — aggregates per-ticker agent outputs into a
ranked portfolio selection with portfolio-level risk analysis.

Changes from v4.2.0:
  FIX (item 59): Sector neutralisation — caps at MAX_PER_SECTOR=2
    positions per GICS sector. Prevents tech concentration bias
    where NVDA, AMD, INTC, QCOM, AVGO all rank top-5 simultaneously.
    Sector map covers all 100 tickers in universe v7.0.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from core.agent.base_agent import BaseAgent


# ============================================================
# SECTOR MAP  (item 59)
# GICS sector assignments for all 100 universe v7.0 tickers.
# Used to enforce MAX_PER_SECTOR cap in portfolio construction.
# ============================================================

TICKER_SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD":  "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "CRM":  "Technology", "INTC": "Technology", "QCOM": "Technology",
    "TXN":  "Technology", "MU":   "Technology", "AMAT": "Technology",
    "LRCX": "Technology", "ADI":  "Technology", "KLAC": "Technology",
    "NOW":  "Technology", "ADBE": "Technology", "INTU": "Technology",
    "CDNS": "Technology",

    # Communication Services
    "GOOGL": "Communication", "META": "Communication",
    "NFLX":  "Communication", "DIS":  "Communication",
    "CMCSA": "Communication", "T":    "Communication",
    "VZ":    "Communication", "TMUS": "Communication",
    "AMZN":  "Communication",  # AWS + Prime (mixed but classified Comm)

    # Financials
    "JPM":  "Financials", "BAC":  "Financials", "GS":   "Financials",
    "SPGI": "Financials", "V":    "Financials", "MA":   "Financials",
    "MS":   "Financials", "WFC":  "Financials", "BLK":  "Financials",
    "AXP":  "Financials", "CB":   "Financials", "ICE":  "Financials",
    "CME":  "Financials", "AON":  "Financials", "TRV":  "Financials",
    "PGR":  "Financials", "MCO":  "Financials", "MSCI": "Financials",

    # Health Care
    "UNH":  "Healthcare", "JNJ":  "Healthcare", "LLY":  "Healthcare",
    "ABBV": "Healthcare", "MRK":  "Healthcare", "TMO":  "Healthcare",
    "ABT":  "Healthcare", "DHR":  "Healthcare", "MDT":  "Healthcare",
    "SYK":  "Healthcare", "EW":   "Healthcare", "ISRG": "Healthcare",
    "BMY":  "Healthcare", "AMGN": "Healthcare",

    # Consumer Discretionary
    "HD":   "Consumer Disc", "COST": "Consumer Disc", "WMT": "Consumer Disc",
    "MCD":  "Consumer Disc", "SBUX": "Consumer Disc", "NKE": "Consumer Disc",
    "LOW":  "Consumer Disc", "TGT":  "Consumer Disc",

    # Consumer Staples
    "PG":  "Consumer Staples", "KO":  "Consumer Staples",
    "PEP": "Consumer Staples", "CL":  "Consumer Staples",
    "PM":  "Consumer Staples",

    # Energy
    "XOM": "Energy", "CVX":  "Energy", "SLB": "Energy",
    "EOG": "Energy", "OXY":  "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO":  "Energy",

    # Industrials
    "CAT": "Industrials", "RTX":  "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "DE":   "Industrials", "GE":  "Industrials",
    "LMT": "Industrials", "NOC":  "Industrials", "EMR": "Industrials",
    "ITW": "Industrials",

    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials",

    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate",

    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "AEP": "Utilities",
}


class PortfolioDecisionAgent(BaseAgent):

    name = "PortfolioDecisionAgent"
    weight = 1.0
    description = (
        "Ranks stocks by hybrid consensus score, applies sector "
        "neutralisation (max 2 per sector), and produces portfolio-level "
        "risk analysis and executive summary."
    )

    TOP_K = int(os.getenv("TOP_N_STOCKS", "5"))
    # FIX (item 59): max positions per GICS sector to reduce concentration
    MAX_PER_SECTOR = int(os.getenv("MAX_POSITIONS_PER_SECTOR", "2"))

    # ============================================================
    # SAFE FLOAT
    # ============================================================

    def _safe_float(self, value, default=0.0):
        try:
            v = float(value)
            if not np.isfinite(v):
                return default
            return v
        except Exception:
            return default

    # ============================================================
    # BASE AGENT ENTRYPOINT
    # ============================================================

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context.get("snapshot", context)
        return self.analyze_snapshot(snapshot)

    # ============================================================
    # SECTOR-NEUTRAL SELECTION  (item 59)
    #
    # Iterates ranked signals in score order. Adds a ticker to the
    # selection only if its sector count is below MAX_PER_SECTOR.
    # Continues until TOP_K positions filled or signals exhausted.
    #
    # Example: if NVDA, AMD, INTC, QCOM all rank 1-4, only the top
    # two (NVDA + AMD) are selected; INTC and QCOM are skipped.
    # The next non-Technology ticker fills slot 3.
    # ============================================================

    def _sector_neutral_select(
        self, ranked: List[Dict], top_k: int, max_per_sector: int
    ) -> List[Dict]:
        """
        Select top_k signals with sector cap of max_per_sector.

        Args:
            ranked:         signals sorted by hybrid_consensus_score desc
            top_k:          max total positions
            max_per_sector: max positions per GICS sector

        Returns:
            selected signals (len <= top_k)
        """

        selected = []
        sector_counts: Dict[str, int] = {}
        skipped: List[str] = []

        for signal in ranked:
            if len(selected) >= top_k:
                break

            ticker = signal.get("ticker", "")
            sector = TICKER_SECTOR_MAP.get(ticker, "Unknown")
            current = sector_counts.get(sector, 0)

            if current < max_per_sector:
                selected.append(signal)
                sector_counts[sector] = current + 1
            else:
                skipped.append(f"{ticker}({sector})")

        if skipped:
            import logging
            logging.getLogger(__name__).info(
                "Sector neutralisation skipped %d tickers: %s",
                len(skipped),
                ", ".join(skipped[:10]),
            )

        return selected

    # ============================================================
    # MAIN PORTFOLIO ANALYSIS
    # ============================================================

    def analyze_snapshot(self, snapshot: Dict[str, Any]):

        signals = snapshot.get("signals", [])
        drift_info = snapshot.get("drift", {}) or {}
        generated_at = datetime.now(timezone.utc).isoformat()

        if not signals:
            return {
                "generated_at": generated_at,
                "snapshot_date": snapshot.get("snapshot_date"),
                "top_selections": [],
                "portfolio_findings": {"status": "no_signals"},
                "executive_summary": "No valid trading signals generated.",
            }

        # Rank by hybrid_consensus_score descending
        ranked = sorted(
            signals,
            key=lambda x: self._safe_float(x.get("hybrid_consensus_score"), 0.0),
            reverse=True,
        )

        # FIX (item 59): Apply sector neutralisation before taking top-K
        top_k = self._sector_neutral_select(
            ranked, self.TOP_K, self.MAX_PER_SECTOR
        )

        # Build detailed selection
        detailed_selection = []
        confidence_values = []
        hybrid_scores = []
        weights = []
        risk_distribution = {}
        liquidity_warnings = 0
        long_count = 0
        short_count = 0
        sectors_used: Dict[str, List[str]] = {}

        for stock in top_k:
            agents = stock.get("agents", {}) or {}
            signal_agent = agents.get("signal_agent", {}) or {}
            tech_agent = agents.get("technical_agent", {}) or {}

            confidence = self._safe_float(signal_agent.get("confidence_numeric"), 0.0)
            hybrid_score = self._safe_float(stock.get("hybrid_consensus_score"), 0.0)
            weight = self._safe_float(stock.get("weight"), 0.0)

            confidence_values.append(confidence)
            hybrid_scores.append(hybrid_score)
            weights.append(abs(weight))

            risk_level = signal_agent.get("risk_level", "unknown")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

            sa_warnings = signal_agent.get("warnings", []) or []
            tech_warnings = tech_agent.get("warnings", []) or []
            combined_warnings = list(dict.fromkeys(sa_warnings + tech_warnings))

            if any("liquidity" in w.lower() for w in combined_warnings):
                liquidity_warnings += 1

            direction = (
                signal_agent.get("signals", {}).get("direction")
                or signal_agent.get("signal")
                or "NEUTRAL"
            )

            if direction == "LONG":
                long_count += 1
            elif direction == "SHORT":
                short_count += 1

            ticker = stock.get("ticker", "")
            sector = TICKER_SECTOR_MAP.get(ticker, "Unknown")
            sectors_used.setdefault(sector, []).append(ticker)

            explanation = (
                f"{ticker} selected with hybrid score {hybrid_score:.2f}. "
                f"Sector: {sector}. Direction: {direction}. "
                f"Confidence: {confidence:.2f}. Risk level: {risk_level}."
            )

            detailed_selection.append({
                "ticker": ticker,
                "sector": sector,
                "weight": round(weight, 6),
                "direction": direction,
                "hybrid_score": round(hybrid_score, 4),
                "model_score": round(self._safe_float(stock.get("raw_model_score")), 4),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "volatility_regime": signal_agent.get("volatility_regime"),
                "technical_bias": tech_agent.get("bias"),
                "warnings": combined_warnings,
                "explanation": explanation,
            })

        # Portfolio metrics
        avg_confidence = float(np.mean(confidence_values)) if confidence_values else 0
        confidence_std = float(np.std(confidence_values)) if confidence_values else 0
        hybrid_dispersion = float(np.std(hybrid_scores)) if hybrid_scores else 0

        gross_exposure = self._safe_float(snapshot.get("gross_exposure"), 0)
        net_exposure = self._safe_float(snapshot.get("net_exposure"), 0)

        portfolio_bias = (
            "long_bias" if net_exposure > 0.05
            else "short_bias" if net_exposure < -0.05
            else "market_neutral"
        )

        max_weight = max(weights) if weights else 0
        concentration_risk = (
            "high" if max_weight > 0.18
            else "moderate" if max_weight > 0.12
            else "low"
        )

        drift_state = drift_info.get("drift_state", "unknown")
        severity = self._safe_float(drift_info.get("severity_score"), 0)
        drift_risk = (
            "high" if severity > 7
            else "moderate" if severity > 3
            else "low"
        )

        findings = {
            "average_confidence": round(avg_confidence, 4),
            "confidence_dispersion": round(confidence_std, 4),
            "hybrid_score_dispersion_topk": round(hybrid_dispersion, 4),
            "gross_exposure": round(gross_exposure, 4),
            "net_exposure": round(net_exposure, 4),
            "portfolio_bias": portfolio_bias,
            "concentration_risk": concentration_risk,
            "risk_distribution": risk_distribution,
            "liquidity_warnings_in_topk": liquidity_warnings,
            "long_positions_in_topk": long_count,
            "short_positions_in_topk": short_count,
            "drift_state": drift_state,
            "drift_severity": round(severity, 4),
            "drift_risk_level": drift_risk,
            # FIX (item 59): sector breakdown for transparency
            "sector_allocation": {k: v for k, v in sectors_used.items()},
            "sector_neutralisation": f"max {self.MAX_PER_SECTOR} per sector",
        }

        executive_summary = (
            f"Hybrid AI selected {len(top_k)} equities across "
            f"{len(sectors_used)} sectors with average confidence "
            f"{avg_confidence:.2f}. Portfolio bias: "
            f"{portfolio_bias.replace('_', ' ')}. "
            f"Drift: {drift_state} (risk {drift_risk}). "
            f"Gross exposure {gross_exposure:.2f}, net {net_exposure:.2f}. "
            f"Concentration risk: {concentration_risk}. "
            f"Sector cap: {self.MAX_PER_SECTOR}/sector."
        )

        return {
            "generated_at": generated_at,
            "model_version": snapshot.get("model_version"),
            "snapshot_date": snapshot.get("snapshot_date"),
            "top_selections": detailed_selection,
            "portfolio_findings": findings,
            "executive_summary": executive_summary,
        }
