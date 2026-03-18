# =========================================================
# RETRAIN TRIGGER v1.2
# FIX: Added cooldown lock — prevents repeated retrain triggers
# New: write_lock(), is_on_cooldown(), cooldown_remaining()
# =========================================================

import os
import json
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("marketsentinel.retrain")

# Cooldown defaults — override via .env
RETRAIN_COOLDOWN_SECONDS = int(os.getenv("RETRAIN_COOLDOWN_SECONDS", "3600"))   # 1 hour
DRIFT_RETRAIN_THRESHOLD = int(os.getenv("DRIFT_RETRAIN_THRESHOLD", "8"))         # severity 0-15


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
        The lock expires after RETRAIN_COOLDOWN_SECONDS (default 1 hour).
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
        """Read the lock file. Returns None if file does not exist or is corrupt."""

        if not os.path.exists(self.LOCK_FILE):
            return None

        try:
            with open(self.LOCK_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def is_on_cooldown(self) -> bool:
        """
        Returns True if a retrain was triggered recently and the
        cooldown period has not expired.
        """

        lock = self._read_lock()

        if not lock:
            return False

        triggered_at = lock.get("triggered_at", 0)
        cooldown = lock.get("cooldown_seconds", RETRAIN_COOLDOWN_SECONDS)

        return (time.time() - triggered_at) < cooldown

    def cooldown_remaining(self) -> int:
        """
        Returns seconds remaining in the cooldown period.
        Returns 0 if not on cooldown.
        """

        lock = self._read_lock()

        if not lock:
            return 0

        triggered_at = lock.get("triggered_at", 0)
        cooldown = lock.get("cooldown_seconds", RETRAIN_COOLDOWN_SECONDS)
        remaining = cooldown - (time.time() - triggered_at)

        return max(0, int(remaining))

    # =====================================================
    # LOAD EVENTS
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
    # FIX: Now checks cooldown before setting retrain_required.
    # If cooldown is active, retrain_required stays False even
    # if drift is above threshold — prevents repeated triggers.
    # =====================================================

    def evaluate(self, drift_result: Dict) -> Dict:

        severity = int(drift_result.get("severity_score", 0))
        drift_state = drift_result.get("drift_state", "unknown")

        on_cooldown = self.is_on_cooldown()
        remaining = self.cooldown_remaining()

        event = None
        retrain_required = False

        if severity >= self.SOFT_THRESHOLD:

            if severity >= self.HARD_THRESHOLD or severity >= DRIFT_RETRAIN_THRESHOLD:

                if on_cooldown:
                    # Drift is bad but cooldown is active — suppress trigger
                    event = self._log_event(severity, drift_state, suppressed=True)
                    logger.info(
                        "Retrain suppressed by cooldown | severity=%d | remaining=%ds",
                        severity,
                        remaining,
                    )
                else:
                    # Drift is bad and cooldown is clear — trigger allowed
                    retrain_required = True
                    self.retrain_required = True
                    event = self._log_event(severity, drift_state, suppressed=False)
                    logger.warning(
                        "Retrain trigger activated | severity=%d state=%s",
                        severity,
                        drift_state,
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