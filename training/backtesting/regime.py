import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger("marketsentinel.regime")


@dataclass(frozen=True)
class RegimeConfig:

    trend_window: int = 150
    volatility_window: int = 40

    bull_vol_threshold: float = 0.03
    bear_vol_threshold: float = 0.03

    trend_buffer: float = 0.007
    crash_vol_threshold: float = 0.05

    persistence_days: int = 4

    EPSILON: float = 1e-8
    MAX_DAILY_RETURN: float = 0.60
    HYSTERESIS_BUFFER: float = 0.003

    MIN_VOL_FLOOR: float = 0.005

    RETURN_SMOOTH_SPAN: int = 5
    TREND_SMOOTH_SPAN: int = 8


class MarketRegimeDetector:

    VALID_REGIMES = ("BULL", "BEAR", "SIDEWAYS", "CRISIS")

    REGIME_MULTIPLIER = {
        "BULL": 1.2,
        "SIDEWAYS": 1.0,
        "BEAR": 0.8,
        "CRISIS": 0.4
    }

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()

    ########################################################
    # SAFETY
    ########################################################

    def _neutral_regime(self, df: pd.DataFrame):

        logger.warning("Regime fallback → assigning SIDEWAYS")

        df = df.copy()

        df["regime"] = "SIDEWAYS"
        df["market_regime"] = "SIDEWAYS"
        df["regime_multiplier"] = 1.0

        return df

    ########################################################
    # PERSISTENCE
    ########################################################

    def _apply_persistence(self, regimes):

        cfg = self.config

        regimes = np.asarray(regimes, dtype=object)

        if len(regimes) == 0:
            return regimes

        confirmed = regimes.copy()

        current = regimes[0]

        streak = 0

        for i in range(1, len(regimes)):

            if regimes[i] == current:
                streak = 0
                confirmed[i] = current
                continue

            streak += 1

            if streak >= cfg.persistence_days:
                current = regimes[i]
                streak = 0

            confirmed[i] = current

        return confirmed

    ########################################################
    # SINGLE ASSET (STRICTLY CAUSAL)
    ########################################################

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        df = df.sort_values("date").copy()

        if "close" not in df.columns:
            return self._neutral_regime(df)

        try:

            close = pd.to_numeric(df["close"], errors="coerce")

            if close.isna().any():
                return self._neutral_regime(df)

            raw_returns = close.pct_change()

            if raw_returns.abs().max() > cfg.MAX_DAILY_RETURN:
                logger.warning("Extreme return detected — fallback.")
                return self._neutral_regime(df)

            ########################################################
            # RETURN SMOOTHING
            ########################################################

            returns = raw_returns.ewm(
                span=cfg.RETURN_SMOOTH_SPAN,
                adjust=False
            ).mean()

            ########################################################
            # TREND
            ########################################################

            ma_long = close.rolling(
                cfg.trend_window,
                min_periods=cfg.trend_window
            ).mean()

            trend_dev = (
                (close - ma_long) /
                (ma_long + cfg.EPSILON)
            )

            trend_dev = trend_dev.ewm(
                span=cfg.TREND_SMOOTH_SPAN,
                adjust=False
            ).mean()

            ########################################################
            # VOLATILITY
            ########################################################

            raw_vol = returns.rolling(
                cfg.volatility_window,
                min_periods=cfg.volatility_window
            ).std()

            volatility = raw_vol.ewm(
                span=10,
                adjust=False
            ).mean()

            volatility = volatility.clip(lower=cfg.MIN_VOL_FLOOR)

            ########################################################
            # REGIME CLASSIFICATION
            ########################################################

            regime = np.full(len(df), "SIDEWAYS", dtype=object)

            ready = ma_long.notna() & volatility.notna()

            crisis = ready & (
                volatility > (cfg.crash_vol_threshold + cfg.HYSTERESIS_BUFFER)
            )

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

            ########################################################
            # APPLY PERSISTENCE
            ########################################################

            regime = self._apply_persistence(regime)

            df["regime"] = regime
            df["market_regime"] = regime

            ########################################################
            # MULTIPLIER FEATURE
            ########################################################

            df["regime_multiplier"] = [
                self.REGIME_MULTIPLIER.get(r, 1.0) for r in regime
            ]

            return df

        except Exception as e:

            logger.warning("Regime degraded → %s", str(e))

            return self._neutral_regime(df)

    ########################################################
    # MULTI ASSET
    ########################################################

    def detect(self, df: pd.DataFrame):

        if df.empty:
            return df

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

        return result