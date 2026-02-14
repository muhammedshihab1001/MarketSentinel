import pandas as pd
import numpy as np

from core.indicators.technical_indicators import TechnicalIndicators


class RiskEngine:
    """
    Institutional Portfolio Risk Engine

    PURPOSE:
    Convert market state into deployable capital decisions.

    This is NOT indicator stacking.
    This is capital protection infrastructure.
    """

    VOL_LOOKBACK = 20
    REGIME_LOOKBACK = 60

    MAX_VOL = 0.06
    CRISIS_VOL = 0.09
    VOL_FLOOR = 0.004

    GAP_LIMIT = 0.22

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    TAIL_RETURN = 0.06
    TAIL_CLUSTER_THRESHOLD = 3
    TAIL_NORMALIZER = 5

    MIN_ROWS = 80
    EPSILON = 1e-8

    ########################################################
    # SAFE UTIL
    ########################################################

    @staticmethod
    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    ########################################################
    # STRICT VALIDATION
    ########################################################

    @classmethod
    def _validate_df(cls, df):

        if df is None or df.empty:
            raise RuntimeError("RiskEngine received empty dataframe.")

        if "close" not in df.columns:
            raise RuntimeError("RiskEngine requires 'close' column.")

        if "date" in df.columns and df["date"].duplicated().any():
            raise RuntimeError("Duplicate candles detected.")

        close = pd.to_numeric(df["close"], errors="raise")

        if not np.isfinite(close).all():
            raise RuntimeError("Non-finite prices detected.")

        if (close <= 0).any():
            raise RuntimeError("Non-positive prices detected.")

        if len(df) < cls.MIN_ROWS:
            raise RuntimeError("Insufficient data for risk computation.")

        # GAP DETECTOR (prevents split / bad feed damage)
        log_returns = np.log(close).diff().dropna()

        if np.abs(log_returns).max() > cls.GAP_LIMIT:
            raise RuntimeError(
                "Extreme price gap detected — refusing risk calc."
            )

    ########################################################
    # SAFE INDICATOR
    ########################################################

    @staticmethod
    def _safe_indicator(func, default=0.5):

        try:
            val = func()

            if val is None:
                return default

            val = float(val)

            if not np.isfinite(val):
                return default

            return val

        except Exception:
            return default

    ########################################################
    # VOLATILITY RISK
    ########################################################

    @classmethod
    def volatility_risk(cls, df):

        log_returns = np.log(df["close"]).diff()

        vol = log_returns.rolling(
            cls.VOL_LOOKBACK,
            min_periods=cls.VOL_LOOKBACK
        ).std()

        latest = vol.iloc[-1]

        if not np.isfinite(latest):
            return 0.5

        latest = max(latest, cls.VOL_FLOOR)

        normalized = latest / cls.MAX_VOL

        return cls._clip01(normalized)

    ########################################################
    # TAIL RISK (Crash Clustering)
    ########################################################

    @classmethod
    def tail_risk(cls, df):

        log_returns = np.log(df["close"]).diff().dropna()

        if len(log_returns) < 30:
            return 0.3

        recent = np.abs(log_returns.tail(20))

        extremes = np.sum(recent > cls.TAIL_RETURN)

        if extremes >= cls.TAIL_CLUSTER_THRESHOLD:
            return 0.9

        return cls._clip01(extremes / cls.TAIL_NORMALIZER)

    ########################################################
    # RSI RISK
    ########################################################

    @classmethod
    def rsi_risk(cls, df, signal):

        def compute():
            return TechnicalIndicators.rsi(df).iloc[-1]

        rsi = cls._safe_indicator(compute, default=50)

        if signal == "BUY":

            if rsi > cls.RSI_OVERBOUGHT:
                return cls._clip01((rsi - 70) / 30)

            if rsi < 40:
                return 0.2

        return 0.4

    ########################################################
    # TREND RISK
    ########################################################

    @classmethod
    def trend_risk(cls, df, signal):

        def ma():
            return TechnicalIndicators.moving_average(df, 20).iloc[-1]

        ma20 = cls._safe_indicator(ma, default=df["close"].iloc[-1])
        price = float(df["close"].iloc[-1])

        if signal == "BUY" and price < ma20:
            return 0.75

        return 0.25

    ########################################################
    # BOLLINGER STRETCH
    ########################################################

    @classmethod
    def bollinger_risk(cls, df, signal):

        try:
            upper, lower = TechnicalIndicators.bollinger_bands(df)

            upper = float(upper.iloc[-1])
            lower = float(lower.iloc[-1])
            price = float(df["close"].iloc[-1])

            if not np.isfinite(upper) or not np.isfinite(lower):
                return 0.5

            width = max(upper - lower, cls.EPSILON)

            stretch = (
                (price - lower) / width
                if signal == "BUY"
                else (upper - price) / width
            )

            return cls._clip01(stretch)

        except Exception:
            return 0.5

    ########################################################
    # REGIME DETECTOR
    ########################################################

    @classmethod
    def regime_risk(cls, df):

        log_returns = np.log(df["close"]).diff().dropna()

        if len(log_returns) < cls.REGIME_LOOKBACK:
            return 0.5, "UNKNOWN"

        vol = log_returns.tail(cls.REGIME_LOOKBACK).std()

        if not np.isfinite(vol):
            return 0.5, "UNKNOWN"

        if vol > cls.CRISIS_VOL:
            return 1.0, "CRISIS"

        if vol > cls.MAX_VOL:
            return 0.75, "BEAR_VOL"

        ma50 = df["close"].rolling(50).mean().iloc[-1]
        price = df["close"].iloc[-1]

        if np.isfinite(ma50) and price < ma50:
            return 0.65, "BEAR"

        return 0.25, "NORMAL"

    ########################################################
    # 🔥 CAPITAL THROTTLE (MOST IMPORTANT METHOD)
    ########################################################

    @staticmethod
    def capital_multiplier(risk_score: float) -> float:
        """
        Converts risk into deployable capital.
        """

        if risk_score >= 0.90:
            return 0.0   # HARD BLOCK

        if risk_score >= 0.75:
            return 0.10

        if risk_score >= 0.60:
            return 0.35

        if risk_score >= 0.40:
            return 0.65

        return 1.0

    ########################################################
    # FINAL COMPOSITE
    ########################################################

    @classmethod
    def analyze(cls, df: pd.DataFrame, signal: str):

        cls._validate_df(df)

        if signal == "HOLD":
            return {
                "risk_score": 0.0,
                "capital_multiplier": 0.0,
                "risk_pct": "0%",
                "regime": "IDLE",
                "components": {}
            }

        vol = cls.volatility_risk(df)
        tail = cls.tail_risk(df)
        rsi = cls.rsi_risk(df, signal)
        trend = cls.trend_risk(df, signal)
        bb = cls.bollinger_risk(df, signal)

        regime_score, regime = cls.regime_risk(df)

        if regime == "CRISIS":

            risk = max(
                0.9,
                0.5 * vol + 0.5 * tail
            )

        else:

            risk = (
                0.30 * vol +
                0.20 * tail +
                0.20 * trend +
                0.15 * rsi +
                0.15 * bb
            )

        risk = cls._clip01(max(risk, regime_score))

        capital_mult = cls.capital_multiplier(risk)

        return {
            "risk_score": round(risk, 4),
            "capital_multiplier": capital_mult,
            "risk_pct": f"{round(risk * 100, 2)}%",
            "regime": regime,
            "components": {
                "volatility": round(vol, 3),
                "tail": round(tail, 3),
                "trend": round(trend, 3),
                "rsi": round(rsi, 3),
                "bollinger": round(bb, 3),
                "regime_floor": round(regime_score, 3)
            }
        }
