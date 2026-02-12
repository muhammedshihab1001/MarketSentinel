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

    EPSILON: float = 1e-8


class MarketRegimeDetector:

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()

    def _apply_persistence(self, regimes):

        cfg = self.config

        confirmed = regimes.copy()

        current = regimes[0]
        streak = 1
        candidate = None
        candidate_streak = 0

        for i in range(1, len(regimes)):

            new = regimes[i]

            # Crisis always overrides immediately
            if new == "CRISIS":
                current = "CRISIS"
                candidate = None
                candidate_streak = 0
                confirmed[i] = current
                continue

            if new == current:
                candidate = None
                candidate_streak = 0
                confirmed[i] = current
                continue

            # Start candidate regime
            if candidate is None:
                candidate = new
                candidate_streak = 1
            elif candidate == new:
                candidate_streak += 1
            else:
                candidate = new
                candidate_streak = 1

            if candidate_streak >= cfg.persistence_days:
                current = candidate
                candidate = None
                candidate_streak = 0

            confirmed[i] = current

        return confirmed

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
            .rolling(
                cfg.volatility_window,
                min_periods=cfg.volatility_window
            )
            .std()
        )

        safe_ma = ma_long.replace(0, np.nan)

        trend_dev = (
            (shifted_close - safe_ma) /
            (safe_ma + cfg.EPSILON)
        )

        regime = np.full(len(df), "SIDEWAYS", dtype=object)

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

        crisis = (
            ready_mask &
            (volatility > cfg.crash_vol_threshold)
        )

        regime[bull] = "BULL"
        regime[bear] = "BEAR"
        regime[crisis] = "CRISIS"

        regime = self._apply_persistence(regime)

        df["regime"] = regime

        return df

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
