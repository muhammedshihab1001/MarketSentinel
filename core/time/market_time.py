import datetime
import os


class MarketTime:
    """
    Institutional time controller.

    Guarantees:
    ✔ deterministic training windows
    ✔ pipeline-level time freeze
    ✔ audit-safe reproducibility
    ✔ override support for backfills
    ✔ prevents temporal drift across models
    """

    DEFAULT_TRAINING_YEARS = 6
    WALK_FORWARD_MONTHS = 3

    # Pipeline can freeze this
    _frozen_today = None

    ########################################################
    # TIME FREEZE (CRITICAL)
    ########################################################

    @classmethod
    def freeze_today(cls, date_str: str):
        """
        Freeze time across the entire training pipeline.

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

        1️ Frozen pipeline date
        2️ ENV override (for backfills)
        3️ System clock
        """

        if cls._frozen_today:
            return cls._frozen_today

        env_override = os.getenv("MARKETSENTINEL_TODAY")

        if env_override:
            return datetime.date.fromisoformat(env_override)

        return datetime.date.today()

    ########################################################
    # TRAINING WINDOW
    ########################################################

    @classmethod
    def training_window(cls, years: int | None = None):

        years = years or cls.DEFAULT_TRAINING_YEARS

        end = cls.today()

        # Leap-safe
        start = end - datetime.timedelta(days=int(365.25 * years))

        return start.isoformat(), end.isoformat()

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
