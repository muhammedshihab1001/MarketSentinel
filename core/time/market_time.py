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
    ✔ restart-safe freeze via ENV
    """

    DEFAULT_TRAINING_YEARS = 6
    WALK_FORWARD_MONTHS = 3

    _frozen_today = None

    ########################################################
    # FREEZE (PIPELINE LEVEL)
    ########################################################

    @classmethod
    def freeze_today(cls, date_str: str):
        """
        Freeze time across the entire process.
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
        2️⃣ ENV freeze
        3️⃣ System clock
        """

        # Highest priority
        if cls._frozen_today:
            return cls._frozen_today

        # Restart-safe freeze
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

    ########################################################
    # AUDIT HELPER (VERY POWERFUL)
    ########################################################

    @classmethod
    def snapshot(cls):
        """
        Returns time lineage for metadata.
        """

        start, end = cls.training_window()

        return {
            "today": cls.today().isoformat(),
            "training_start": start,
            "training_end": end,
            "walk_forward_anchor": cls.walk_forward_anchor()
        }
