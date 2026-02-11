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

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        df = df.sort_values("date").copy()

        shifted_close = df["close"].shift(1)

        ma_long = (
            shifted_close
            .rolling(cfg.trend_window, min_periods=cfg.trend_window)
            .mean()
        )

        returns = shifted_close.pct_change()

        volatility = (
            returns
            .rolling(cfg.volatility_window,
                     min_periods=cfg.volatility_window)
            .std()
        )

        trend_dev = (shifted_close - ma_long) / ma_long

        regime = np.full(len(df), "UNKNOWN", dtype=object)

        ready_mask = (
            ma_long.notna() &
            volatility.notna()
        )

        bull = (
            ready_mask &
            (trend_dev > cfg.trend_buffer) &
            (volatility < cfg.bull_vol_threshold)
        )

        bear = (
            ready_mask &
            (trend_dev < -cfg.trend_buffer) &
            (volatility > cfg.bear_vol_threshold)
        )

        sideways = (
            ready_mask &
            ~bull &
            ~bear
        )

        regime[sideways] = "SIDEWAYS"
        regime[bull] = "BULL"
        regime[bear] = "BEAR"

        crisis = ready_mask & (volatility > cfg.crash_vol_threshold)
        regime[crisis] = "CRISIS"

        regime = self._apply_persistence_trailing(regime)

        df["regime"] = regime

        df = df[df["regime"] != "UNKNOWN"].copy()

        return df

    def _apply_persistence_trailing(self, regimes):

        cfg = self.config

        confirmed = regimes.copy()

        current = regimes[0]
        streak = 1

        for i in range(1, len(regimes)):

            if regimes[i] == current:
                streak += 1
                continue

            streak = 1

            if streak >= cfg.persistence_days:
                current = regimes[i]

            confirmed[i] = current

        return confirmed

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
