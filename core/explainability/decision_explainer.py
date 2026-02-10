import math


class DecisionExplainer:
    """
    Generates bounded, deterministic explanations for signals.

    Guarantees:
    - never empty
    - NaN safe
    - capped verbosity
    """

    MAX_REASONS = 5

    @staticmethod
    def _safe(value, fallback=0.0):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        return value

    def explain(
        self,
        prediction: int,
        prob_up: float,
        sentiment: float,
        volatility: float,
        rsi: float
    ):

        prob_up = self._safe(prob_up, 0.5)
        sentiment = self._safe(sentiment, 0.0)
        volatility = self._safe(volatility, 0.0)
        rsi = self._safe(rsi, 50)

        reasons = []

        # Direction
        if prediction == 1:
            reasons.append("Model favors upward movement")
        else:
            reasons.append("Model favors downward movement")

        # Conviction
        if prob_up > 0.65:
            reasons.append("High model conviction")
        elif prob_up < 0.35:
            reasons.append("Strong downside probability")

        # Sentiment
        if sentiment > 0.2:
            reasons.append("Positive market sentiment")
        elif sentiment < -0.2:
            reasons.append("Negative market sentiment")

        # Volatility
        if volatility > 0.04:
            reasons.append("Elevated market volatility")
        else:
            reasons.append("Stable volatility regime")

        # RSI
        if rsi > 70:
            reasons.append("Overbought conditions")
        elif rsi < 30:
            reasons.append("Oversold conditions")

        if not reasons:
            reasons.append("Signal derived from composite indicators")

        return " | ".join(reasons[:self.MAX_REASONS])
