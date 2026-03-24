"""
MarketSentinel — Market Regime Detector v1.0

Detects the current market regime (BULL / BEAR / CRISIS / SIDEWAYS)
from cross-sectional price data and returns a regime_multiplier
that scales position sizes accordingly.

Used as an XGBoost feature (item 61) — gives the model market context
it cannot derive from individual ticker features alone.

Regime logic:
  BULL    → breadth > 0.60, market return > 0        → multiplier = 1.2
  BEAR    → breadth < 0.40, market return < 0        → multiplier = 0.6
  CRISIS  → volatility > 2× median, breadth < 0.30  → multiplier = 0.3
  SIDEWAYS→ otherwise                                → multiplier = 1.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detects market regime from cross-sectional OHLCV data.

    Usage (inference):
        detector = MarketRegimeDetector()
        regime = detector.detect(price_df)
        multiplier = regime["regime_multiplier"]   # 0.3 / 0.6 / 1.0 / 1.2

    Usage (feature pipeline):
        detector = MarketRegimeDetector()
        df = detector.add_regime_feature(df)
        # adds columns: regime, market_regime, regime_multiplier
    """

    # Regime thresholds
    BULL_BREADTH_MIN = 0.60
    BEAR_BREADTH_MAX = 0.40
    CRISIS_BREADTH_MAX = 0.30
    CRISIS_VOL_MULTIPLIER = 2.0      # vol must be > 2× median to trigger crisis

    REGIME_MULTIPLIERS = {
        "BULL":     1.2,
        "SIDEWAYS": 1.0,
        "BEAR":     0.6,
        "CRISIS":   0.3,
    }

    WINDOW = 20   # rolling window for cross-sectional stats

    def detect(self, price_df: pd.DataFrame) -> Dict:
        """
        Detect current regime from the most recent cross-section.

        Args:
            price_df: DataFrame with columns [date, ticker, close, return]
                      May be multi-ticker (cross-sectional) or single-ticker.

        Returns:
            {
                "regime": "BULL" | "BEAR" | "CRISIS" | "SIDEWAYS",
                "regime_multiplier": float,
                "breadth": float,
                "market_return": float,
                "volatility": float,
            }
        """

        try:
            df = price_df.copy()
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.sort_values("date")

            if "return" not in df.columns:
                df["return"] = df.groupby("ticker")["close"].pct_change().clip(-0.5, 0.5)

            # Use the most recent N days
            recent_dates = df["date"].drop_duplicates().sort_values().tail(self.WINDOW)
            recent = df[df["date"].isin(recent_dates)]

            if recent.empty:
                return self._default_regime()

            # Cross-sectional breadth: fraction of tickers with positive return
            breadth = float((recent["return"] > 0).mean())

            # Market return: equal-weight mean daily return
            market_return = float(recent["return"].mean())

            # Volatility: cross-sectional std of returns
            volatility = float(recent["return"].std())

            # Median historical volatility for crisis threshold
            all_vol = float(df["return"].std()) if len(df) > self.WINDOW else volatility

            # Regime classification
            if (
                volatility > self.CRISIS_VOL_MULTIPLIER * all_vol
                and breadth < self.CRISIS_BREADTH_MAX
            ):
                regime = "CRISIS"

            elif breadth > self.BULL_BREADTH_MIN and market_return > 0:
                regime = "BULL"

            elif breadth < self.BEAR_BREADTH_MAX and market_return < 0:
                regime = "BEAR"

            else:
                regime = "SIDEWAYS"

            multiplier = self.REGIME_MULTIPLIERS[regime]

            logger.debug(
                "Regime detected | regime=%s multiplier=%.1f breadth=%.3f "
                "market_return=%.4f volatility=%.4f",
                regime, multiplier, breadth, market_return, volatility,
            )

            return {
                "regime": regime,
                "regime_multiplier": multiplier,
                "breadth": round(breadth, 4),
                "market_return": round(market_return, 6),
                "volatility": round(volatility, 6),
            }

        except Exception as e:
            logger.warning("Regime detection failed: %s — using SIDEWAYS default", e)
            return self._default_regime()

    def _default_regime(self) -> Dict:
        return {
            "regime": "SIDEWAYS",
            "regime_multiplier": 1.0,
            "breadth": 0.5,
            "market_return": 0.0,
            "volatility": 0.0,
        }

    def add_regime_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime_multiplier as a rolling feature column to a price DataFrame.

        This is the method called from feature_engineering.py.
        Computes regime for each date using a trailing 20-day window.

        Adds columns:
            regime            — str label: BULL / BEAR / CRISIS / SIDEWAYS
            market_regime     — same as regime (alias for legacy compat)
            regime_multiplier — float: 0.3 / 0.6 / 1.0 / 1.2

        Args:
            df: DataFrame with [date, ticker, close, return] columns

        Returns:
            df with regime columns added (per-row, aligned by date)
        """

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        if "return" not in df.columns:
            df["return"] = (
                df.groupby("ticker")["close"]
                .pct_change()
                .clip(-0.5, 0.5)
            )

        # Compute regime per date using trailing 20-day window
        all_dates = sorted(df["date"].unique())
        date_regime_map = {}

        for i, date in enumerate(all_dates):
            window_start_idx = max(0, i - self.WINDOW + 1)
            window_dates = all_dates[window_start_idx: i + 1]
            window_df = df[df["date"].isin(window_dates)]
            result = self.detect(window_df)
            date_regime_map[date] = result

        # Map back to DataFrame
        df["regime"] = df["date"].map(
            lambda d: date_regime_map.get(d, {}).get("regime", "SIDEWAYS")
        )
        df["market_regime"] = df["regime"]
        df["regime_multiplier"] = df["date"].map(
            lambda d: date_regime_map.get(d, {}).get("regime_multiplier", 1.0)
        ).astype(np.float32)

        logger.info(
            "Regime features added | dates=%d regime_counts=%s",
            len(all_dates),
            df.groupby("regime").size().to_dict(),
        )

        return df