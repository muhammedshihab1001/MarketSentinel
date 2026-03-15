"""
MarketSentinel v4.1.0

Base class for all intelligence agents in the Hybrid
MarketSentinel multi-agent architecture.

Agent hierarchy:
    BaseAgent
    ├── SignalAgent           (ML model signal interpretation)
    ├── TechnicalRiskAgent   (technical indicator risk scoring)
    ├── PortfolioDecisionAgent (final buy/hold/sell decision)
    └── PoliticalRiskAgent   (macro / geopolitical risk overlay)

Design goals:
    - Stateless  — no shared mutable state between calls
    - Deterministic — same context always produces same output
    - Safe for noisy yfinance data (NaN/inf tolerant)
    - Compatible with hybrid consensus scoring (weighted average)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base for all MarketSentinel agents.

    Each agent:
        - Receives a structured context dict
        - Returns a structured output dict via _format_output()
        - Never mutates shared state
        - Is safe to call with partial or noisy data
    """

    name:        str   = "BaseAgent"
    weight:      float = 1.0    # contribution weight in hybrid consensus scoring
    description: str   = ""     # human-readable summary for dashboards

    # ─────────────────────────────────────────────────────
    # MAIN CONTRACT
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis using the provided context snapshot.

        Parameters
        ----------
        context : Dict[str, Any]
            Structured snapshot containing any of:
                - feature_row      : dict of engineered features
                - model_output     : XGBoost prediction output
                - drift_info       : drift detector results
                - probability      : predicted probability stats
                - market_data      : recent OHLCV rows
                - additional keys  : agent-specific inputs

        Returns
        -------
        Dict[str, Any]
            Structured output — use _format_output() to build this.
            Required keys:
                agent_name  : str
                weight      : float
                score       : float  in [0, 1]
                confidence  : float  in [0, 1]
                signals     : Dict[str, Any]
                warnings    : List[str]
                reasoning   : List[str]  (in logical order — not sorted)
        """
        raise NotImplementedError

    # ─────────────────────────────────────────────────────
    # STANDARD OUTPUT FORMATTER
    # ─────────────────────────────────────────────────────

    def _format_output(
        self,
        score:      float,
        confidence: float,
        signals:    Optional[Dict[str, Any]] = None,
        warnings:   Optional[List[str]]      = None,
        reasoning:  Optional[List[str]]      = None,
    ) -> Dict[str, Any]:
        """
        Build a standardised agent output dict.

        Guarantees:
            - score and confidence are clipped to [0, 1]
            - NaN / inf values are replaced with 0.0
            - warnings are deduplicated (order preserved via dict trick)
            - reasoning order is preserved — NOT sorted (logical flow matters)
        """
        score      = self._clip(score)
        confidence = self._clip(confidence)

        # Deduplicate warnings while preserving insertion order
        raw_warnings  = warnings  or []
        raw_reasoning = reasoning or []

        deduped_warnings  = list(dict.fromkeys(raw_warnings))
        deduped_reasoning = list(dict.fromkeys(raw_reasoning))

        return {
            "agent_name": self.name,
            "weight":     self.weight,
            "score":      score,
            "confidence": confidence,
            "signals":    signals or {},
            "warnings":   deduped_warnings,
            "reasoning":  deduped_reasoning,
        }

    # ─────────────────────────────────────────────────────
    # SAFE NUMERIC HELPERS
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _clip(value: float) -> float:
        """
        Safely clip a value to [0.0, 1.0].
        Returns 0.0 for NaN, inf, or any non-numeric input.
        """
        try:
            v = float(value)
            if not np.isfinite(v):
                return 0.0
            return float(np.clip(v, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Convert any value to float safely.
        Returns default for None, NaN, inf, or unconvertible values.
        """
        try:
            v = float(value)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    # ─────────────────────────────────────────────────────
    # METADATA  (for dashboards and health endpoints)
    # ─────────────────────────────────────────────────────

    def metadata(self) -> Dict[str, Any]:
        """
        Expose agent identity and configuration.
        Used by monitoring dashboards and the /agent/info endpoint.
        """
        return {
            "agent_name":  self.name,
            "weight":      self.weight,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"<Agent: {self.name} | weight={self.weight}>"