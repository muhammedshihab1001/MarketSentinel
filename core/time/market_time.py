import datetime
import json
import os


class MarketTime:
    """
    Institutional time governor.

    Guarantees:
    ✔ deterministic windows
    ✔ restart-safe freeze
    ✔ audit lineage
    ✔ timezone neutrality
    ✔ prevents future leakage
    ✔ model-governed horizons
    """

    ########################################################
    # MODEL WINDOWS
    ########################################################

    MODEL_WINDOWS = {
        "xgboost": 3,
        "lstm": 5,
        "sarimax": 7
    }

    WALK_FORWARD_MONTHS = 3

    FREEZE_FILE = "artifacts/time_freeze.json"

    _frozen_today = None

    ########################################################
    # TRUE UTC TODAY
    ########################################################

    @staticmethod
    def _utc_today():
        return datetime.datetime.utcnow().date()

    ########################################################
    # FREEZE (PERSISTENT)
    ########################################################

    @classmethod
    def freeze_today(cls, date_str: str):
        """
        Freeze globally and persist.

        NEVER silently changes across restarts.
        """

        frozen = datetime.date.fromisoformat(date_str)

        if frozen > cls._utc_today():
            raise RuntimeError(
                "Cannot freeze time in the future."
            )

        cls._frozen_today = frozen

        os.makedirs("artifacts", exist_ok=True)

        with open(cls.FREEZE_FILE, "w") as f:
            json.dump({"frozen_today": date_str}, f)

    ########################################################
    # LOAD FREEZE (AUTO)
    ########################################################

    @classmethod
    def _load_freeze(cls):

        if not os.path.exists(cls.FREEZE_FILE):
            return None

        try:
            with open(cls.FREEZE_FILE) as f:
                payload = json.load(f)

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
    # SAFE TODAY
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
    # GENERIC WINDOW
    ########################################################

    @classmethod
    def training_window(cls, years: int):

        if years <= 0:
            raise RuntimeError("Training years must be > 0.")

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
    # MODEL WINDOW
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
    # WALK FORWARD
    ########################################################

    @classmethod
    def walk_forward_anchor(cls):

        today = cls.today()

        anchor = today - datetime.timedelta(
            days=int(30.437 * cls.WALK_FORWARD_MONTHS)
        )

        return anchor.isoformat()

    ########################################################
    # SNAPSHOT (EXTREMELY IMPORTANT)
    ########################################################

    @classmethod
    def snapshot_for(cls, model_name: str):

        start, end = cls.window_for(model_name)

        return {
            "model": model_name,
            "today": cls.today().isoformat(),
            "training_start": start,
            "training_end": end,
            "walk_forward_anchor": cls.walk_forward_anchor()
        }
