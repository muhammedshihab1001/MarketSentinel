import pandas as pd
import numpy as np

from core.indicators.technical_indicators import TechnicalIndicators


class RiskEngine:
    """
    Institutional Trade Risk Engine.

    Produces a 0–100 risk score per signal.

    Safe:
    ✔ deterministic
    ✔ numeric stable
    ✔ float32 safe
    ✔ interpretable
    """

    VOL_LOOKBACK = 20

    MAX_VOL = 0.06        # 6% daily = extreme
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    ##############################################

    @staticmethod
    def _clip01(x):
        return float(np.clip(x, 0, 1))

    ##############################################
    # VOLATILITY RISK
    ##############################################

    @classmethod
    def volatility_risk(cls, df):

        returns = df["close"].pct_change()

        vol = returns.rolling(
            cls.VOL_LOOKBACK,
            min_periods=cls.VOL_LOOKBACK
        ).std()

        latest_vol = vol.iloc[-1]

        if not np.isfinite(latest_vol):
            return 0.5  # neutral risk

        return cls._clip01(latest_vol / cls.MAX_VOL)

    ##############################################
    # RSI RISK
    ##############################################

    @classmethod
    def rsi_risk(cls, df, signal):

        rsi = TechnicalIndicators.rsi(df).iloc[-1]

        if signal == "BUY":

            if rsi > cls.RSI_OVERBOUGHT:
                return cls._clip01((rsi - 70) / 30)

            if rsi < 40:
                return 0.2

        elif signal == "SELL":

            if rsi < cls.RSI_OVERSOLD:
                return cls._clip01((30 - rsi) / 30)

            if rsi > 60:
                return 0.2

        return 0.4

    ##############################################
    # TREND RISK
    ##############################################

    @staticmethod
    def trend_risk(df, signal):

        ma20 = TechnicalIndicators.moving_average(df, 20).iloc[-1]
        price = df["close"].iloc[-1]

        if not np.isfinite(ma20):
            return 0.5

        if signal == "BUY" and price < ma20:
            return 0.7

        if signal == "SELL" and price > ma20:
            return 0.7

        return 0.25

    ##############################################
    # BOLLINGER RISK
    ##############################################

    @staticmethod
    def bollinger_risk(df, signal):

        upper, lower = TechnicalIndicators.bollinger_bands(df)

        upper = upper.iloc[-1]
        lower = lower.iloc[-1]
        price = df["close"].iloc[-1]

        if not np.isfinite(upper) or not np.isfinite(lower):
            return 0.5

        band_width = upper - lower + 1e-9

        if signal == "BUY":
            stretch = (price - lower) / band_width

        else:
            stretch = (upper - price) / band_width

        return float(np.clip(stretch, 0, 1))

    ##############################################
    # FINAL COMPOSITE
    ##############################################

    @classmethod
    def analyze(cls, df: pd.DataFrame, signal: str):

        if signal == "HOLD":
            return {
                "risk_score": 0.0,
                "risk_pct": "0%",
                "components": {}
            }

        vol = cls.volatility_risk(df)
        rsi = cls.rsi_risk(df, signal)
        trend = cls.trend_risk(df, signal)
        bb = cls.bollinger_risk(df, signal)

        # Institutional weighting
        risk = (
            0.35 * vol +
            0.25 * trend +
            0.20 * rsi +
            0.20 * bb
        )

        risk = cls._clip01(risk)

        return {
            "risk_score": round(risk, 4),
            "risk_pct": f"{round(risk * 100, 2)}%",
            "components": {
                "volatility": round(vol, 3),
                "trend": round(trend, 3),
                "rsi": round(rsi, 3),
                "bollinger": round(bb, 3),
            }
        }
