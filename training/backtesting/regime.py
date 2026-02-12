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


class MarketRegimeDetector:

    VALID_REGIMES = {"BULL", "BEAR", "SIDEWAYS", "CRISIS"}

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()

    ########################################################
    # PERSISTENCE FILTER
    ########################################################

    def _apply_persistence(self, regimes):

        cfg = self.config
        regimes = np.asarray(regimes, dtype=object)

        confirmed = regimes.copy()
        current = regimes[0]

        candidate = None
        streak = 0

        for i in range(1, len(regimes)):

            new = regimes[i]

            if new not in self.VALID_REGIMES:
                logger.warning("Invalid regime detected: %s", new)
                new = "SIDEWAYS"

            # Crisis has ABSOLUTE PRIORITY
            if new == "CRISIS":
                current = "CRISIS"
                candidate = None
                streak = 0
                confirmed[i] = current
                continue

            if new == current:
                candidate = None
                streak = 0
                confirmed[i] = current
                continue

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

        return confirmed

    ########################################################
    # SINGLE ASSET
    ########################################################

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        try:

            df = df.sort_values("date").copy()

            if df["close"].isna().any():
                raise RuntimeError("NaN close prices")

            if not np.isfinite(df["close"]).all():
                raise RuntimeError("Non-finite prices")

            if (df["close"] <= 0).any():
                raise RuntimeError("Non-positive prices")

            shifted = df["close"].shift(1)

            returns = shifted.pct_change()

            # SOFT GUARD — drop ticker instead of crashing
            if returns.abs().max() > cfg.MAX_DAILY_RETURN:
                raise RuntimeError("Unrealistic returns")

            ma_long = shifted.rolling(
                cfg.trend_window,
                min_periods=cfg.trend_window
            ).mean()

            volatility = (
                returns
                .rolling(cfg.volatility_window,
                         min_periods=cfg.volatility_window)
                .std()
                .clip(lower=cfg.EPSILON)
            )

            safe_ma = ma_long.replace(0, np.nan)

            trend_dev = (
                (shifted - safe_ma) /
                (safe_ma + cfg.EPSILON)
            )

            regime = np.full(len(df), "SIDEWAYS", dtype=object)

            ready = ma_long.notna() & volatility.notna()

            # CRISIS FIRST (priority)
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

            df["regime"] = regime

            # DO NOT DELETE WARMUP ROWS
            df["regime"] = df["regime"].fillna("SIDEWAYS")

            df["date"] = pd.to_datetime(df["date"], utc=True)

            return df

        except Exception as e:

            logger.warning(
                "Regime detection rejected ticker — %s",
                str(e)
            )

            return None

    ########################################################
    # MULTI ASSET
    ########################################################

    def detect(self, df: pd.DataFrame):

        if "ticker" not in df.columns:
            raise RuntimeError(
                "Regime detection requires ticker column."
            )

        if df.empty:
            raise RuntimeError("Empty dataframe passed to regime detector.")

        grouped = []

        for ticker, slice_df in df.sort_values(
            ["ticker", "date"]
        ).groupby("ticker"):

            if len(slice_df) < 300:
                logger.info("Skipping small asset: %s", ticker)
                continue

            detected = self._detect_single_asset(slice_df)

            if detected is not None and not detected.empty:
                grouped.append(detected)

        if len(grouped) < 3:
            raise RuntimeError(
                "Too few assets survived regime detection."
            )

        result = (
            pd.concat(grouped)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        return result
