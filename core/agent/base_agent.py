# =========================================================
# BASE AGENT INTERFACE
# Hybrid Multi-Agent Architecture Foundation
# =========================================================

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """
    Base class for all intelligence agents in the Hybrid
    MarketSentinel architecture.

    Each agent:
    - Receives structured context
    - Returns structured output
    - Does NOT mutate shared state
    """

    name: str = "BaseAgent"
    weight: float = 1.0  # Used later for consensus scoring

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
                "score": float (0–1),
                "confidence": float (0–1),
                "signals": Dict[str, Any],
                "warnings": list,
                "reasoning": list
            }
        """
        raise NotImplementedError

    # -----------------------------------------------------
    # OPTIONAL HOOK
    # -----------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        """
        Optional metadata exposure for dashboard/debugging.
        """
        return {
            "agent_name": self.name,
            "weight": self.weight
        }