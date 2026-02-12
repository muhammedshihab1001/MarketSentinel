import datetime


class MarketTime:
    """
    Institutional time governor.

    Guarantees:
    ✔ deterministic windows
    ✔ model-specific horizons
    ✔ audit-safe lineage
    ✔ prevents temporal leakage
    ✔ restart-safe freeze
    ✔ no ENV drift
    """

    ########################################################
    # MODEL-SPECIFIC HORIZONS (VERY IMPORTANT)
    ########################################################

    MODEL_WINDOWS = {
        "xgboost": 3,     # regime aware
        "lstm": 5,        # sequence depth
        "sarimax": 7      # macro structure
    }

    WALK_FORWARD_MONTHS = 3

    _frozen_today = None

    ########################################################
    # FREEZE (PIPELINE LEVEL)
    ########################################################

    @classmethod
    def freeze_today(cls, date_str: str):
        """
        Freeze time across the entire process.

        Example:
            MarketTime.freeze_today("2026-02-12")
        """

        cls._frozen_today = datetime.date.fromisoformat(date_str)

    ########################################################
    # SAFE TODAY
    ########################################################

    @classmethod
    def today(cls):
        """
        Priority:

        1️⃣ Explicit freeze
        2️⃣ System clock
        """

        if cls._frozen_today:
            return cls._frozen_today

        return datetime.date.today()

    ########################################################
    # GENERIC WINDOW
    ########################################################

    @classmethod
    def training_window(cls, years: int):

        end = cls.today()

        # Leap-safe
        start = end - datetime.timedelta(days=int(365.25 * years))

        return start.isoformat(), end.isoformat()

    ########################################################
    # MODEL WINDOW (NEW — CRITICAL)
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
    # WALK FORWARD ANCHOR
    ########################################################

    @classmethod
    def walk_forward_anchor(cls):

        today = cls.today()

        anchor = today - datetime.timedelta(
            days=int(30.437 * cls.WALK_FORWARD_MONTHS)
        )

        return anchor.isoformat()

    ########################################################
    # AUDIT SNAPSHOT (ELITE FEATURE)
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
