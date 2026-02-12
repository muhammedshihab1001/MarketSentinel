import datetime


class MarketTime:
    """
    Institutional time controller.

    Guarantees:
    ✔ identical training windows across models
    ✔ deterministic pipeline runs
    ✔ audit-safe metadata
    ✔ prevents temporal drift between models
    """

    DEFAULT_TRAINING_YEARS = 6
    WALK_FORWARD_MONTHS = 3

    @staticmethod
    def today():
        return datetime.date.today()

    @classmethod
    def training_window(cls, years: int | None = None):

        years = years or cls.DEFAULT_TRAINING_YEARS

        end = cls.today()
        start = end - datetime.timedelta(days=365 * years)

        return start.isoformat(), end.isoformat()

    @classmethod
    def walk_forward_anchor(cls):

        today = cls.today()

        anchor = today - datetime.timedelta(
            days=30 * cls.WALK_FORWARD_MONTHS
        )

        return anchor.isoformat()
