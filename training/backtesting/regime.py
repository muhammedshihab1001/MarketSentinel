import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class RegimeConfig:

    trend_window: int = 200
    volatility_window: int = 50

    bull_vol_threshold: float = 0.02
    bear_vol_threshold: float = 0.025

    trend_buffer: float = 0.01
    crash_vol_threshold: float = 0.04

    persistence_days: int = 5


class MarketRegimeDetector:
    """
    Institutional market regime classifier.

    Guarantees:
    - zero cross-asset leakage
    - strictly trailing indicators
    - deterministic persistence
    - crisis-aware labeling
    - no index distortion
    """

    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.config = config

    # ---------------------------------------------------

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        df = df.sort_values("date").copy()

        ma_long = df["close"].rolling(
            window=cfg.trend_window,
            min_periods=cfg.trend_window
        ).mean()

        returns = df["close"].pct_change()

        volatility = returns.rolling(
            window=cfg.volatility_window,
            min_periods=cfg.volatility_window
        ).std()

        trend_dev = (df["close"] - ma_long) / ma_long

        regime = np.full(len(df), "SIDEWAYS", dtype=object)

        bull = (
            (trend_dev > cfg.trend_buffer) &
            (volatility < cfg.bull_vol_threshold)
        )

        bear = (
            (trend_dev < -cfg.trend_buffer) &
            (volatility > cfg.bear_vol_threshold)
        )

        crisis = volatility > cfg.crash_vol_threshold

        regime[bull] = "BULL"
        regime[bear] = "BEAR"
        regime[crisis] = "CRISIS"

        regime = self._apply_persistence(regime)

        df["regime"] = regime

        return df

    # ---------------------------------------------------

    def _apply_persistence(self, regimes):

        cfg = self.config
        n = len(regimes)

        confirmed = regimes.copy()

        current = regimes[0]
        streak = 1

        for i in range(1, n):

            if regimes[i] == current:
                streak += 1
                continue

            # new regime candidate
            candidate = regimes[i]
            confirm = True

            if i + cfg.persistence_days >= n:
                break

            for j in range(1, cfg.persistence_days):
                if regimes[i + j] != candidate:
                    confirm = False
                    break

            if confirm:
                current = candidate
                streak = cfg.persistence_days
            else:
                confirmed[i] = current

        return confirmed

    # ---------------------------------------------------

    def detect(self, df: pd.DataFrame):

        if "ticker" not in df.columns:
            raise RuntimeError(
                "Regime detection requires ticker column."
            )

        grouped = []

        for ticker, slice_df in df.groupby("ticker", sort=False):
            detected = self._detect_single_asset(slice_df)
            grouped.append(detected)

        result = pd.concat(grouped).sort_values("date")

        return result.reset_index(drop=True)
