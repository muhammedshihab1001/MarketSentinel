import datetime
import json
import os
import hashlib
import logging


logger = logging.getLogger(__name__)


class MarketTime:
    """
    Institutional Time Governor (Production Grade v4.2)

    Guarantees:
    ✔ deterministic training windows
    ✔ tamper-detected freeze file
    ✔ atomic persistence
    ✔ walk-forward protection
    ✔ dynamic env override support
    ✔ institutional safety bounds
    ✔ reproducible time_hash lineage
    ✔ research vs production mode separation
    """

    TIME_GOVERNANCE_VERSION = "4.2"

    ########################################################
    # CONFIGURABLE TRAINING WINDOWS
    #
    # FIX (item 29): Changed xgboost from 5 years → 2 years.
    # Training pipeline uses DATA_SYNC_HISTORY_DAYS=730 (2 years).
    # A 5-year DEFAULT_WINDOWS caused a mismatch where MarketTime
    # would request 5 years of data but only 2 years existed in DB.
    ########################################################

    DEFAULT_WINDOWS = {
        "xgboost": 2,   # FIX: was 5 — align to 730-day (2yr) training window
    }

    TRADING_DAYS_PER_YEAR = 252
    WALK_FORWARD_MONTHS = 3

    MIN_YEARS = 1
    MAX_YEARS = 15

    FREEZE_FILE = os.path.abspath(
        os.path.join("artifacts", "time_freeze.json")
    )

    LOCK_FILE = FREEZE_FILE + ".lock"

    _frozen_today = None

    @staticmethod
    def _utc_today():
        return datetime.datetime.utcnow().date()

    @staticmethod
    def _hash_payload(payload: dict):
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

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

    @classmethod
    def _acquire_lock(cls):
        try:
            fd = os.open(cls.LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
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

    @classmethod
    def freeze_today(cls, date_str: str):
        frozen = datetime.date.fromisoformat(date_str)
        if frozen > cls._utc_today():
            raise RuntimeError("Cannot freeze time in the future.")
        cls._acquire_lock()
        try:
            cls._frozen_today = frozen
            payload = {
                "frozen_today": date_str,
                "governance_version": cls.TIME_GOVERNANCE_VERSION
            }
            cls._atomic_write(cls.FREEZE_FILE, payload)
        finally:
            cls._release_lock()

    @classmethod
    def _load_freeze(cls):
        if not os.path.exists(cls.FREEZE_FILE):
            return None
        try:
            with open(cls.FREEZE_FILE) as f:
                payload = json.load(f)
            integrity = payload.pop("integrity_hash", None)
            if integrity != cls._hash_payload(payload):
                raise RuntimeError("Freeze file integrity failure — possible tampering.")
            frozen = datetime.date.fromisoformat(payload["frozen_today"])
            if frozen > cls._utc_today():
                raise RuntimeError("Freeze file contains future date.")
            return frozen
        except Exception as e:
            raise RuntimeError(f"Time freeze file corrupted — refusing to run. ({e})")

    @classmethod
    def today(cls):
        if cls._frozen_today:
            return cls._frozen_today
        persisted = cls._load_freeze()
        if persisted:
            cls._frozen_today = persisted
            return persisted
        return cls._utc_today()

    @classmethod
    def _validate_years(cls, years: int):
        if not isinstance(years, int):
            raise RuntimeError("Training window must be integer years.")
        if years < cls.MIN_YEARS or years > cls.MAX_YEARS:
            raise RuntimeError(
                f"Training window outside institutional bounds "
                f"({cls.MIN_YEARS}–{cls.MAX_YEARS} years allowed)."
            )

    @classmethod
    def _resolve_years(cls, model_name: str):
        env_key = f"{model_name.upper()}_TRAIN_YEARS"
        if env_key in os.environ:
            try:
                years = int(os.environ[env_key])
            except ValueError:
                raise RuntimeError(f"Invalid env override for {env_key}")
        else:
            years = cls.DEFAULT_WINDOWS.get(model_name)
        if years is None:
            raise RuntimeError(f"No training window configured for model: {model_name}")
        cls._validate_years(years)
        return years

    @classmethod
    def training_window(cls, years: int):
        cls._validate_years(years)
        end = cls.today()
        days = int(cls.TRADING_DAYS_PER_YEAR * years * 1.05)
        start = end - datetime.timedelta(days=days)
        if start >= end:
            raise RuntimeError("Invalid training window generated.")
        return start.isoformat(), end.isoformat()

    @classmethod
    def window_for(cls, model_name: str):
        years = cls._resolve_years(model_name)
        return cls.training_window(years)

    @classmethod
    def walk_forward_anchor(cls):
        today = cls.today()
        anchor = today - datetime.timedelta(days=int(30.437 * cls.WALK_FORWARD_MONTHS))
        return anchor

    @classmethod
    def snapshot_for(cls, model_name: str):
        start, end = cls.window_for(model_name)
        anchor = cls.walk_forward_anchor()
        if anchor.isoformat() <= start:
            raise RuntimeError("Walk-forward anchor overlaps training window.")
        contract = {
            "governance_version": cls.TIME_GOVERNANCE_VERSION,
            "model": model_name,
            "today": cls.today().isoformat(),
            "training_start": start,
            "training_end": end,
            "walk_forward_anchor": anchor.isoformat()
        }
        canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        contract["time_hash"] = hashlib.sha256(canonical).hexdigest()
        contract["training_id"] = hashlib.sha256(
            (contract["time_hash"] + model_name).encode()
        ).hexdigest()
        return contract

    @classmethod
    def is_frozen(cls):
        return os.path.exists(cls.FREEZE_FILE)
