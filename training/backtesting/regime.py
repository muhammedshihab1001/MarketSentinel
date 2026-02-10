import pandas as pd
import numpy as np
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

    trend_buffer: float = 0.01      # 🔥 1% deviation required
    crash_vol_threshold: float = 0.04

    persistence_days: int = 5       # 🔥 prevents regime flip noise


# ---------------------------------------------------
# DETECTOR
# ---------------------------------------------------

class MarketRegimeDetector:
    """
    Institutional market regime classifier.

    Guarantees:
    - no forward leakage
    - regime stability
    - crash detection
    - persistence filtering
    """

    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.config = config

    # ---------------------------------------------------

    def detect(self, df: pd.DataFrame):

        df = df.copy()

        # ---------------------------------------
        # CORE FEATURES (STRICTLY TRAILING)
        # ---------------------------------------

        df["ma_long"] = df["close"].rolling(
            window=self.config.trend_window,
            min_periods=self.config.trend_window
        ).mean()

        df["returns"] = df["close"].pct_change()

        df["volatility"] = df["returns"].rolling(
            window=self.config.volatility_window,
            min_periods=self.config.volatility_window
        ).std()

        # ---------------------------------------
        # TREND DEVIATION
        # ---------------------------------------

        df["trend_dev"] = (
            (df["close"] - df["ma_long"]) / df["ma_long"]
        )

        # ---------------------------------------
        # INITIAL REGIME
        # ---------------------------------------

        df["regime"] = "SIDEWAYS"

        bull_mask = (
            (df["trend_dev"] > self.config.trend_buffer) &
            (df["volatility"] < self.config.bull_vol_threshold)
        )

        bear_mask = (
            (df["trend_dev"] < -self.config.trend_buffer) &
            (df["volatility"] > self.config.bear_vol_threshold)
        )

        crash_mask = (
            df["volatility"] > self.config.crash_vol_threshold
        )

        df.loc[bull_mask, "regime"] = "BULL"
        df.loc[bear_mask, "regime"] = "BEAR"
        df.loc[crash_mask, "regime"] = "CRISIS"

        # ---------------------------------------
        # REMOVE EARLY ROWS
        # ---------------------------------------

        df = df.dropna(subset=["ma_long", "volatility"])

        # ---------------------------------------
        # PERSISTENCE FILTER
        # ---------------------------------------

        regime = df["regime"].values
        filtered = regime.copy()

        min_days = self.config.persistence_days

        for i in range(min_days, len(regime)):

            window = regime[i-min_days:i]

            if len(set(window)) == 1:
                filtered[i] = window[0]
            else:
                filtered[i] = filtered[i-1]

        df["regime"] = filtered

        return df
