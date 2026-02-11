import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
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
    - zero lookahead bias
    - strictly trailing indicators
    - cross-asset isolation
    - deterministic output
    """

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()

    # ---------------------------------------------------

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        df = df.sort_values("date").copy()

        # SHIFTED — prevents same-bar leakage
        ma_long = (
            df["close"]
            .rolling(cfg.trend_window, min_periods=cfg.trend_window)
            .mean()
            .shift(1)
        )

        returns = df["close"].pct_change()

        volatility = (
            returns
            .rolling(cfg.volatility_window,
                     min_periods=cfg.volatility_window)
            .std()
            .shift(1)
        )

        trend_dev = (df["close"] - ma_long) / ma_long

        regime = np.full(len(df), "SIDEWAYS", dtype=object)

        crisis = volatility > cfg.crash_vol_threshold

        bull = (
            (trend_dev > cfg.trend_buffer) &
            (volatility < cfg.bull_vol_threshold)
        )

        bear = (
            (trend_dev < -cfg.trend_buffer) &
            (volatility > cfg.bear_vol_threshold)
        )

        regime[crisis] = "CRISIS"
        regime[bull & ~crisis] = "BULL"
        regime[bear & ~crisis] = "BEAR"

        regime = self._apply_persistence_trailing(regime)

        df["regime"] = regime

        return df

    # ---------------------------------------------------
    # TRAILING PERSISTENCE (NO FUTURE)
    # ---------------------------------------------------

    def _apply_persistence_trailing(self, regimes):

        cfg = self.config

        confirmed = regimes.copy()

        current = regimes[0]
        streak = 1

        for i in range(1, len(regimes)):

            if regimes[i] == current:
                streak += 1
                continue

            # new candidate begins
            if regimes[i] != regimes[i-1]:
                streak = 1

            if streak >= cfg.persistence_days:
                current = regimes[i]

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

        result = (
            pd.concat(grouped)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        return result
