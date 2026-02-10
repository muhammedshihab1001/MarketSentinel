import pandas as pd
from dataclasses import dataclass


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

@dataclass
class RegimeConfig:
    trend_window: int = 200
    volatility_window: int = 50

    bull_vol_threshold: float = 0.02
    bear_vol_threshold: float = 0.025


# ---------------------------------------------------
# DETECTOR
# ---------------------------------------------------

class MarketRegimeDetector:
    """
    Institutional market regime classifier.

    Upgrades:
    ✅ Fully vectorized (NO loops)
    ✅ Configurable thresholds
    ✅ Research scalable
    ✅ Numerically safer
    """

    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.config = config

    # ---------------------------------------------------

    def detect(self, df: pd.DataFrame):

        df = df.copy()

        # ---------------------------------------
        # CORE FEATURES
        # ---------------------------------------

        df["ma_long"] = df["close"].rolling(
            self.config.trend_window
        ).mean()

        df["returns"] = df["close"].pct_change()

        df["volatility"] = df["returns"].rolling(
            self.config.volatility_window
        ).std()

        # ---------------------------------------
        # VECTOR REGIME CLASSIFICATION
        # ---------------------------------------

        df["regime"] = "SIDEWAYS"

        bull_mask = (
            (df["close"] > df["ma_long"]) &
            (df["volatility"] < self.config.bull_vol_threshold)
        )

        bear_mask = (
            (df["close"] < df["ma_long"]) &
            (df["volatility"] > self.config.bear_vol_threshold)
        )

        df.loc[bull_mask, "regime"] = "BULL"
        df.loc[bear_mask, "regime"] = "BEAR"

        # Unknown early rows
        df.loc[df["ma_long"].isna(), "regime"] = "UNKNOWN"

        return df
