# =========================================================
# RETRAIN TRIGGER v1.0
# Simple governance layer for MarketSentinel
# CV-optimized (no heavy orchestration)
# =========================================================

import os
import json
import time
import logging
from typing import Dict, List

logger = logging.getLogger("marketsentinel.retrain")


class RetrainTrigger:

    EVENTS_FILE = os.getenv(
        "RETRAIN_EVENTS_FILE",
        "artifacts/drift/retrain_events.json"
    )

    HARD_THRESHOLD = int(
        os.getenv("RETRAIN_HARD_THRESHOLD", "10")
    )

    SOFT_THRESHOLD = int(
        os.getenv("RETRAIN_SOFT_THRESHOLD", "6")
    )

    MAX_EVENT_HISTORY = 100

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self):

        os.makedirs(os.path.dirname(self.EVENTS_FILE), exist_ok=True)

        self._events = self._load_events()

        self.retrain_required = any(
            e.get("retrain_required", False) for e in self._events
        )

    # =====================================================
    # LOAD EVENTS
    # =====================================================

    def _load_events(self) -> List[Dict]:

        if not os.path.exists(self.EVENTS_FILE):
            return []

        try:

            with open(self.EVENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

        except Exception as e:
            logger.warning("Failed to load retrain events: %s", str(e))

        return []

    # =====================================================
    # SAVE EVENTS
    # =====================================================

    def _save_events(self):

        try:

            trimmed = self._events[-self.MAX_EVENT_HISTORY:]

            with open(self.EVENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, indent=2)

        except Exception as e:
            logger.warning("Failed to persist retrain events: %s", str(e))

    # =====================================================
    # LOG EVENT
    # =====================================================

    def _log_event(self, severity: int, drift_state: str):

        event = {
            "timestamp": int(time.time()),
            "severity_score": severity,
            "drift_state": drift_state,
            "retrain_required": severity >= self.HARD_THRESHOLD
        }

        self._events.append(event)

        self._save_events()

        return event

    # =====================================================
    # EVALUATE DRIFT
    # =====================================================

    def evaluate(self, drift_result: Dict):

        severity = int(drift_result.get("severity_score", 0))
        drift_state = drift_result.get("drift_state", "unknown")

        event = None

        if severity >= self.SOFT_THRESHOLD:

            event = self._log_event(severity, drift_state)

            if severity >= self.HARD_THRESHOLD:

                self.retrain_required = True

                logger.warning(
                    "Retrain trigger activated | severity=%d state=%s",
                    severity,
                    drift_state
                )

            else:

                logger.info(
                    "Drift warning recorded | severity=%d",
                    severity
                )

        return {
            "retrain_required": self.retrain_required,
            "events": self._events[-10:],  # recent history
            "last_event": event
        }

    # =====================================================
    # MANUAL RESET
    # =====================================================

    def clear_retrain_flag(self):

        self.retrain_required = False

        logger.info("Retrain flag manually cleared.")

        return {
            "retrain_required": False
        }