# =========================================================
# BASE AGENT INTERFACE v2.0
# Hybrid Multi-Agent Architecture Foundation
# =========================================================

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseAgent(ABC):
    """
    Base class for all intelligence agents in the Hybrid
    MarketSentinel architecture.

    Design Goals:
    - Stateless
    - Deterministic
    - Safe for noisy financial data (yfinance)
    - Compatible with hybrid consensus scoring
    - CV-polished architecture

    Each agent:
    - Receives structured context
    - Returns structured output
    - Does NOT mutate shared state
    """

    name: str = "BaseAgent"
    weight: float = 1.0  # Used for hybrid consensus scoring

    # -----------------------------------------------------
    # MAIN CONTRACT
    # -----------------------------------------------------

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis using provided context.

        Parameters
        ----------
        context : Dict[str, Any]
            Structured snapshot context including:
            - feature row
            - model outputs
            - drift info
            - probability stats
            - any additional agent inputs

        Returns
        -------
        Dict[str, Any]
            Must return structured output containing at minimum:
            {
                "agent_name": str,
                "weight": float,
                "score": float (0–1),
                "confidence": float (0–1),
                "signals": Dict[str, Any],
                "warnings": list,
                "reasoning": list
            }
        """
        raise NotImplementedError

    # -----------------------------------------------------
    # STANDARD OUTPUT FORMATTER (SAFE)
    # -----------------------------------------------------

    def _format_output(
        self,
        score: float,
        confidence: float,
        signals: Dict[str, Any] | None = None,
        warnings: list | None = None,
        reasoning: list | None = None
    ) -> Dict[str, Any]:
        """
        Standardized agent output builder.
        Ensures:
        - Score clipped to [0,1]
        - Confidence clipped to [0,1]
        - Safe defaults
        """

        score = self._clip(score)
        confidence = self._clip(confidence)

        return {
            "agent_name": self.name,
            "weight": self.weight,
            "score": score,
            "confidence": confidence,
            "signals": signals or {},
            "warnings": sorted(set(warnings or [])),
            "reasoning": sorted(set(reasoning or [])),
        }

    # -----------------------------------------------------
    # SAFE NUMERIC CLIP
    # -----------------------------------------------------

    @staticmethod
    def _clip(value: float) -> float:
        try:
            v = float(value)
            if not np.isfinite(v):
                return 0.0
            return float(np.clip(v, 0.0, 1.0))
        except Exception:
            return 0.0

    # -----------------------------------------------------
    # OPTIONAL METADATA
    # -----------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        """
        Optional metadata exposure for dashboard/debugging.
        """
        return {
            "agent_name": self.name,
            "weight": self.weight
        }