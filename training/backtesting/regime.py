import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging


logger = logging.getLogger("marketsentinel.regime")


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
    MAX_DAILY_RETURN: float = 0.60

    MIN_SURVIVAL_RATIO: float = 0.35
    MAX_CRISIS_RATIO: float = 0.65

    HYSTERESIS_BUFFER: float = 0.003


class MarketRegimeDetector:

    VALID_REGIMES = ("BULL", "BEAR", "SIDEWAYS", "CRISIS")

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()

    ########################################################

    def _assert_monotonic(self, df):

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError(
                "Non-monotonic timestamps detected."
            )

    ########################################################
    # SAFE FALLBACK — DO NOT DESTROY STRUCTURE
    ########################################################

    def _neutral_regime(self, df: pd.DataFrame):

        logger.warning(
            "Regime detector degraded — assigning neutral SIDEWAYS without overwriting dataset."
        )

        df = df.copy()

        if "regime" not in df.columns:
            df["regime"] = "SIDEWAYS"

        df["regime"] = df["regime"].fillna("SIDEWAYS")

        df["regime"] = pd.Categorical(
            df["regime"],
            categories=self.VALID_REGIMES
        )

        df["market_regime"] = df["regime"]

        return df

    ########################################################
    # PERSISTENCE
    ########################################################

    def _apply_persistence(self, regimes):

        cfg = self.config
        regimes = np.asarray(regimes, dtype=object)

        confirmed = regimes.copy()

        current = None
        candidate = None
        streak = 0

        for i in range(len(regimes)):

            new = regimes[i]

            if new not in self.VALID_REGIMES:
                new = "SIDEWAYS"

            if new == "CRISIS":
                current = "CRISIS"
                candidate = None
                streak = 0
                confirmed[i] = current
                continue

            if current is None:
                current = new
                confirmed[i] = current
                continue

            if new != current:

                if candidate == new:
                    streak += 1
                else:
                    candidate = new
                    streak = 1

                if streak >= cfg.persistence_days:
                    current = candidate
                    candidate = None
                    streak = 0

                confirmed[i] = current
                continue

            confirmed[i] = current

        return confirmed

    ########################################################
    # SINGLE ASSET — NEVER DROP TICKER
    ########################################################

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        try:

            df = df.sort_values("date").copy()
            self._assert_monotonic(df)

            close = pd.to_numeric(df["close"], errors="raise")

            shifted_close = close.shift(1)
            returns = shifted_close.pct_change()

            ma_long = (
                shifted_close
                .rolling(cfg.trend_window,
                         min_periods=cfg.trend_window)
                .mean()
                .shift(1)
            )

            volatility = (
                returns
                .rolling(cfg.volatility_window,
                         min_periods=cfg.volatility_window)
                .std()
                .shift(1)
            ).clip(lower=cfg.EPSILON)

            trend_dev = (
                (shifted_close - ma_long) /
                (ma_long + cfg.EPSILON)
            )

            regime = np.full(len(df), "SIDEWAYS", dtype=object)

            ready = ma_long.notna() & volatility.notna()

            crisis = ready & (volatility > cfg.crash_vol_threshold)

            bull = (
                ready &
                ~crisis &
                (trend_dev > cfg.trend_buffer) &
                (volatility < cfg.bull_vol_threshold)
            )

            bear = (
                ready &
                ~crisis &
                (trend_dev < -cfg.trend_buffer) &
                (volatility > cfg.bear_vol_threshold)
            )

            regime[crisis] = "CRISIS"
            regime[bull] = "BULL"
            regime[bear] = "BEAR"

            regime = self._apply_persistence(regime)

            df["regime"] = pd.Categorical(
                regime,
                categories=self.VALID_REGIMES
            )

            # ⭐ DO NOT DROP WARMUP
            df["regime"] = df["regime"].fillna("SIDEWAYS")

            return df

        except Exception as e:

            logger.warning(
                "Regime detection degraded for ticker — %s",
                str(e)
            )

            df["regime"] = "SIDEWAYS"
            return df

    ########################################################
    # MULTI ASSET
    ########################################################

    def detect(self, df: pd.DataFrame):

        grouped = []

        for _, slice_df in df.sort_values(
            ["ticker", "date"]
        ).groupby("ticker", sort=False):

            detected = self._detect_single_asset(slice_df)
            grouped.append(detected)

        result = (
            pd.concat(grouped)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        counts = result.groupby("date")["ticker"].nunique()

        if counts.min() < 2:
            return self._neutral_regime(result)

        return result
