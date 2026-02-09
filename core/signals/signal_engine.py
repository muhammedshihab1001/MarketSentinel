from dataclasses import dataclass
from typing import List


@dataclass
class SignalConfig:
    prob_threshold: float = 0.60
    sentiment_threshold: float = 0.10
    volatility_cap: float = 0.05


class SignalEngine:
    """
    Converts ML predictions into BUY / SELL / HOLD signals.
    """

    def __init__(self, config: SignalConfig = SignalConfig()):
        self.config = config

    def generate_signal(
        self,
        prediction: int,
        prob_up: float,
        avg_sentiment: float,
        volatility: float | None
    ) -> str:

        # Risk control
        if volatility is not None and volatility > self.config.volatility_cap:
            return "HOLD"

        if (
            prediction == 1
            and prob_up >= self.config.prob_threshold
            and avg_sentiment >= self.config.sentiment_threshold
        ):
            return "BUY"

        if (
            prediction == 0
            and prob_up <= (1 - self.config.prob_threshold)
            and avg_sentiment <= -self.config.sentiment_threshold
        ):
            return "SELL"

        return "HOLD"


# STANDALONE FUNCTION (IMPORTANT)
def fuse_decision(
    direction_signal: str,
    prob_up: float,
    lstm_prices: List[float],
    prophet_trend: str
) -> str:
    """
    Final ensemble decision using:
    - Direction model (XGBoost)
    - LSTM forecast
    - Prophet trend
    """

    if not lstm_prices or len(lstm_prices) < 2:
        return "HOLD"

    expected_return = (lstm_prices[-1] - lstm_prices[0]) / lstm_prices[0]

    if (
        direction_signal == "BUY"
        and prophet_trend == "BULLISH"
        and expected_return > 0.02
        and prob_up >= 0.6
    ):
        return "BUY"

    if (
        direction_signal == "SELL"
        and prophet_trend == "BEARISH"
        and expected_return < -0.02
        and prob_up <= 0.4
    ):
        return "SELL"

    return "HOLD"
