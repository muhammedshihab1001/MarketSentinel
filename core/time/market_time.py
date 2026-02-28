import datetime
import json
import os
import hashlib


class MarketTime:
    """
    Institutional Time Governor (Deterministic).

    Guarantees:
    ✔ reproducible training
    ✔ tamper-detected freeze file
    ✔ atomic persistence
    ✔ lineage-safe time hashing
    ✔ walk-forward protection
    ✔ configurable institutional training windows
    """

    TIME_GOVERNANCE_VERSION = "3.1"

    ########################################################
    # UPDATED TRAINING WINDOWS (INCREASED DATA DEPTH)
    ########################################################

    MODEL_WINDOWS = {
        # Increased from 3 → 8 years for signal stability
        "xgboost": 8,
    }

    WALK_FORWARD_MONTHS = 3

    FREEZE_FILE = os.path.abspath(
        os.path.join("artifacts", "time_freeze.json")
    )

    LOCK_FILE = FREEZE_FILE + ".lock"

    _frozen_today = None

    ########################################################
    # UTC
    ########################################################

    @staticmethod
    def _utc_today():
        return datetime.datetime.utcnow().date()

    ########################################################
    # HASH
    ########################################################

    @staticmethod
    def _hash_payload(payload: dict):

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    ########################################################
    # ATOMIC WRITE
    ########################################################

    @classmethod
    def _atomic_write(cls, path, payload):

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        payload["integrity_hash"] = cls._hash_payload(payload)

        tmp = path + ".tmp"

        try:

            with open(tmp, "w") as f:
                json.dump(payload, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, path)

            if os.name != "nt":
                fd = os.open(directory, os.O_DIRECTORY)
                os.fsync(fd)
                os.close(fd)

        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    ########################################################
    # LOCK
    ########################################################

    @classmethod
    def _acquire_lock(cls):

        try:
            fd = os.open(
                cls.LOCK_FILE,
                os.O_CREAT | os.O_EXCL | os.O_RDWR
            )
            os.close(fd)

        except FileExistsError:
            raise RuntimeError(
                "Time freeze lock detected — another process may be freezing time."
            )

    @classmethod
    def _release_lock(cls):

        if os.path.exists(cls.LOCK_FILE):
            try:
                os.remove(cls.LOCK_FILE)
            except Exception:
                pass

    ########################################################
    # FREEZE
    ########################################################

    @classmethod
    def freeze_today(cls, date_str: str):

        frozen = datetime.date.fromisoformat(date_str)

        if frozen > cls._utc_today():
            raise RuntimeError(
                "Cannot freeze time in the future."
            )

        cls._acquire_lock()

        try:

            cls._frozen_today = frozen

            payload = {
                "frozen_today": date_str,
                "governance_version":
                    cls.TIME_GOVERNANCE_VERSION
            }

            cls._atomic_write(
                cls.FREEZE_FILE,
                payload
            )

        finally:
            cls._release_lock()

    ########################################################
    # LOAD FREEZE (TAMPER SAFE)
    ########################################################

    @classmethod
    def _load_freeze(cls):

        if not os.path.exists(cls.FREEZE_FILE):
            return None

        try:
            with open(cls.FREEZE_FILE) as f:
                payload = json.load(f)

            integrity = payload.pop("integrity_hash", None)

            if integrity != cls._hash_payload(payload):
                raise RuntimeError(
                    "Freeze file integrity failure — possible tampering."
                )

            frozen = datetime.date.fromisoformat(
                payload["frozen_today"]
            )

            if frozen > cls._utc_today():
                raise RuntimeError(
                    "Freeze file contains future date."
                )

            return frozen

        except Exception:
            raise RuntimeError(
                "Time freeze file corrupted — refusing to run."
            )

    ########################################################
    # TODAY
    ########################################################

    @classmethod
    def today(cls):

        if cls._frozen_today:
            return cls._frozen_today

        persisted = cls._load_freeze()

        if persisted:
            cls._frozen_today = persisted
            return persisted

        return cls._utc_today()

    ########################################################
    # VALIDATE YEARS
    ########################################################

    @classmethod
    def _validate_years(cls, years: int):

        if not isinstance(years, int):
            raise RuntimeError("Training window must be integer years.")

        if years <= 0 or years > 15:
            raise RuntimeError(
                "Training window outside institutional bounds (1–15 years allowed)."
            )

    ########################################################

    @classmethod
    def training_window(cls, years: int):

        cls._validate_years(years)

        end = cls.today()

        start = end - datetime.timedelta(
            days=int(365.25 * years)
        )

        if start >= end:
            raise RuntimeError(
                "Invalid training window generated."
            )

        return start.isoformat(), end.isoformat()

    ########################################################

    @classmethod
    def window_for(cls, model_name: str):

        if model_name not in cls.MODEL_WINDOWS:
            raise RuntimeError(
                f"No training window configured for model: {model_name}"
            )

        years = cls.MODEL_WINDOWS[model_name]

        return cls.training_window(years)

    ########################################################

    @classmethod
    def walk_forward_anchor(cls):

        today = cls.today()

        anchor = today - datetime.timedelta(
            days=int(30.437 * cls.WALK_FORWARD_MONTHS)
        )

        return anchor

    ########################################################
    # SNAPSHOT
    ########################################################

    @classmethod
    def snapshot_for(cls, model_name: str):

        start, end = cls.window_for(model_name)
        anchor = cls.walk_forward_anchor()

        if anchor.isoformat() <= start:
            raise RuntimeError(
                "Walk-forward anchor overlaps training window."
            )

        contract = {
            "governance_version": cls.TIME_GOVERNANCE_VERSION,
            "model": model_name,
            "today": cls.today().isoformat(),
            "training_start": start,
            "training_end": end,
            "walk_forward_anchor": anchor.isoformat()
        }

        canonical = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        contract["time_hash"] = hashlib.sha256(
            canonical
        ).hexdigest()

        return contract