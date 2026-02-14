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

    MIN_SURVIVAL_RATIO: float = 0.6
    MAX_CRISIS_RATIO: float = 0.65

    # NEW — prevents regime flip noise
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
    # INSTITUTIONAL PERSISTENCE + HYSTERESIS
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

            # crisis overrides everything
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

            # hysteresis — prevent flip noise
            if (
                new != current
                and new != "CRISIS"
                and current != "CRISIS"
            ):
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
    # SINGLE ASSET
    ########################################################

    def _detect_single_asset(self, df: pd.DataFrame):

        cfg = self.config

        try:

            df = df.sort_values("date").copy()
            self._assert_monotonic(df)

            close = pd.to_numeric(df["close"], errors="raise")

            if not np.isfinite(close).all():
                raise RuntimeError("Non-finite prices")

            if (close <= 0).any():
                raise RuntimeError("Non-positive prices")

            ###################################################
            # TRUE NO-LEAK RETURNS
            ###################################################

            shifted_close = close.shift(1)
            returns = shifted_close.pct_change()

            if returns.abs().max() > cfg.MAX_DAILY_RETURN:
                raise RuntimeError("Unrealistic returns")

            ###################################################
            # DOUBLE LAG ROLLING
            ###################################################

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

            safe_ma = ma_long.replace(0, np.nan)

            trend_dev = (
                (shifted_close - safe_ma) /
                (safe_ma + cfg.EPSILON)
            )

            regime = np.full(len(df), "SIDEWAYS", dtype=object)

            ready = ma_long.notna() & volatility.notna()

            crisis = ready & (volatility > cfg.crash_vol_threshold)

            bull = (
                ready &
                ~crisis &
                (trend_dev > (cfg.trend_buffer + cfg.HYSTERESIS_BUFFER)) &
                (volatility < cfg.bull_vol_threshold)
            )

            bear = (
                ready &
                ~crisis &
                (trend_dev < -(cfg.trend_buffer + cfg.HYSTERESIS_BUFFER)) &
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

            df["date"] = pd.to_datetime(df["date"], utc=True)

            warmup = max(
                cfg.trend_window,
                cfg.volatility_window
            ) + cfg.persistence_days + 5

            if len(df) <= warmup:
                return None

            return df.iloc[warmup:].copy()

        except Exception as e:

            logger.warning(
                "Regime detection rejected ticker — %s",
                str(e)
            )

            return None

    ########################################################
    # MARKET REGIME
    ########################################################

    def _build_market_regime(self, df):

        regime_counts = (
            df.groupby(["date", "regime"])
            .size()
            .unstack(fill_value=0)
        )

        crisis_ratio = (
            regime_counts.get("CRISIS", 0) /
            regime_counts.sum(axis=1)
        )

        ###################################################
        # INSTITUTIONAL FIX — NEVER CRASH PIPELINE
        ###################################################

        market_regime = regime_counts.idxmax(axis=1)

        market_regime[crisis_ratio > self.config.MAX_CRISIS_RATIO] = "CRISIS"

        # shift to prevent lookahead
        market_regime = market_regime.shift(1)

        return market_regime.rename("market_regime")

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

        tickers_total = df["ticker"].nunique()

        grouped = []

        for ticker, slice_df in df.sort_values(
            ["ticker", "date"]
        ).groupby("ticker", sort=False):

            min_required = (
                self.config.trend_window +
                self.config.volatility_window +
                self.config.persistence_days + 10
            )

            if len(slice_df) < min_required:
                continue

            detected = self._detect_single_asset(slice_df)

            if detected is not None and not detected.empty:
                grouped.append(detected)

        survivors = len(grouped)

        if survivors < 3:
            raise RuntimeError(
                "Too few assets survived regime detection."
            )

        if survivors / tickers_total < self.config.MIN_SURVIVAL_RATIO:
            raise RuntimeError(
                "Universe collapse detected during regime filtering."
            )

        result = (
            pd.concat(grouped)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        ###################################################
        # ADD MARKET REGIME
        ###################################################

        market_regime = self._build_market_regime(result)

        result = result.merge(
            market_regime,
            on="date",
            how="left"
        )

        counts = result.groupby("date")["ticker"].nunique()

        if counts.min() < 2:
            raise RuntimeError(
                "Calendar misalignment detected across assets."
            )

        return result
