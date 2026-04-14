# =========================================================
# RETRAIN TRIGGER v1.3
# FIX: Added cooldown lock — prevents repeated retrain triggers
# New: write_lock(), is_on_cooldown(), cooldown_remaining()
#
# FIX v1.3: evaluate() now accepts both int and Dict input.
#   drift.py calls trigger.evaluate(severity_score) with int.
#   Old signature only accepted Dict and called .get() on it,
#   which raised AttributeError when passed an int.
#   Result: retrain_required was always False — retrain
#   was never triggered even when drift was critical.
#
#   Fix: evaluate() now accepts Union[int, Dict].
#   If int passed → wraps in dict automatically.
#   If dict passed → reads severity_score key as before.
# =========================================================

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger("marketsentinel.retrain")

RETRAIN_COOLDOWN_SECONDS = int(os.getenv("RETRAIN_COOLDOWN_SECONDS", "3600"))
DRIFT_RETRAIN_THRESHOLD = int(os.getenv("DRIFT_RETRAIN_THRESHOLD", "8"))


class RetrainTrigger:

    EVENTS_FILE = os.getenv(
        "RETRAIN_EVENTS_FILE",
        "artifacts/drift/retrain_events.json",
    )

    LOCK_FILE = os.getenv(
        "RETRAIN_LOCK_FILE",
        "artifacts/drift/retrain.lock",
    )

    HARD_THRESHOLD = int(os.getenv("RETRAIN_HARD_THRESHOLD", "10"))
    SOFT_THRESHOLD = int(os.getenv("RETRAIN_SOFT_THRESHOLD", "6"))

    MAX_EVENT_HISTORY = 100
    MAX_SEVERITY = 100

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
    # COOLDOWN LOCK
    # =====================================================

    def write_lock(self):
        """
        Write the cooldown lock file.
        Call this BEFORE triggering the training container.
        """
        os.makedirs(os.path.dirname(self.LOCK_FILE), exist_ok=True)

        lock_data = {
            "triggered_at": int(time.time()),
            "triggered_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cooldown_seconds": RETRAIN_COOLDOWN_SECONDS,
            "expires_at": int(time.time()) + RETRAIN_COOLDOWN_SECONDS,
        }

        try:
            with open(self.LOCK_FILE, "w", encoding="utf-8") as f:
                json.dump(lock_data, f, indent=2)
            logger.info(
                "Retrain lock written | expires_in=%ds",
                RETRAIN_COOLDOWN_SECONDS,
            )
        except Exception as e:
            logger.warning("Failed to write retrain lock | error=%s", e)

    def _read_lock(self) -> Optional[dict]:
        if not os.path.exists(self.LOCK_FILE):
            return None
        try:
            with open(self.LOCK_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def is_on_cooldown(self) -> bool:
        lock = self._read_lock()
        if not lock:
            return False
        triggered_at = lock.get("triggered_at", 0)
        cooldown = lock.get("cooldown_seconds", RETRAIN_COOLDOWN_SECONDS)
        return (time.time() - triggered_at) < cooldown

    def cooldown_remaining(self) -> int:
        lock = self._read_lock()
        if not lock:
            return 0
        triggered_at = lock.get("triggered_at", 0)
        cooldown = lock.get("cooldown_seconds", RETRAIN_COOLDOWN_SECONDS)
        remaining = cooldown - (time.time() - triggered_at)
        return max(0, int(remaining))

    # =====================================================
    # LOAD / SAVE EVENTS
    # =====================================================

    def _load_events(self) -> List[Dict]:
        if not os.path.exists(self.EVENTS_FILE):
            try:
                with open(self.EVENTS_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f)
            except Exception:
                pass
            return []

        try:
            with open(self.EVENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning("Failed to load retrain events: %s", str(e))
            try:
                backup = self.EVENTS_FILE + ".corrupt"
                os.rename(self.EVENTS_FILE, backup)
                logger.warning("Corrupted retrain log backed up → %s", backup)
            except Exception:
                pass

        return []

    def _save_events(self):
        try:
            trimmed = self._events[-self.MAX_EVENT_HISTORY:]
            with open(self.EVENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist retrain events: %s", str(e))

    def _log_event(self, severity: int, drift_state: str, suppressed: bool = False):
        severity = max(0, min(severity, self.MAX_SEVERITY))
        event = {
            "timestamp": int(time.time()),
            "severity_score": severity,
            "drift_state": drift_state,
            "retrain_required": severity >= self.HARD_THRESHOLD,
            "suppressed_by_cooldown": suppressed,
        }
        self._events.append(event)
        self._save_events()
        return event

    # =====================================================
    # EVALUATE DRIFT
    #
    # FIX v1.3: Accepts Union[int, Dict] input.
    # drift.py calls trigger.evaluate(severity_score) with int.
    # Old code called drift_result.get() on int → AttributeError.
    # Now normalises input to dict before processing.
    # =====================================================

    def evaluate(self, drift_result: Union[int, Dict]) -> Dict:
        """
        Evaluate drift and decide if retrain is needed.

        Args:
            drift_result: Either:
                - int: severity_score directly (from drift.py)
                - dict: full drift result with "severity_score" key

        Returns:
            dict with retrain_required, cooldown_active, etc.
        """
        # FIX v1.3: normalise int input to dict
        if isinstance(drift_result, (int, float)):
            drift_result = {
                "severity_score": int(drift_result),
                "drift_state": "unknown",
            }

        severity = int(drift_result.get("severity_score", 0))
        drift_state = drift_result.get("drift_state", "unknown")

        on_cooldown = self.is_on_cooldown()
        remaining = self.cooldown_remaining()

        event = None
        retrain_required = False

        if severity >= self.SOFT_THRESHOLD:

            if severity >= self.HARD_THRESHOLD or severity >= DRIFT_RETRAIN_THRESHOLD:

                if on_cooldown:
                    event = self._log_event(severity, drift_state, suppressed=True)
                    logger.info(
                        "Retrain suppressed by cooldown | severity=%d | remaining=%ds",
                        severity, remaining,
                    )
                else:
                    retrain_required = True
                    self.retrain_required = True
                    event = self._log_event(severity, drift_state, suppressed=False)
                    logger.warning(
                        "Retrain trigger activated | severity=%d state=%s",
                        severity, drift_state,
                    )
            else:
                event = self._log_event(severity, drift_state, suppressed=False)
                logger.info("Drift warning recorded | severity=%d", severity)

        return {
            "retrain_required": retrain_required,
            "cooldown_active": on_cooldown,
            "cooldown_remaining_seconds": remaining,
            "events": self._events[-10:],
            "last_event": event,
        }

    # =====================================================
    # MANUAL RESET
    # =====================================================

    def clear_retrain_flag(self):
        self.retrain_required = False
        logger.info("Retrain flag manually cleared.")
        return {"retrain_required": False}