from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

@dataclass
class SignalConfig:
    prob_threshold: float = 0.60
    sentiment_threshold: float = 0.10
    volatility_cap: float = 0.05
    min_expected_return: float = 0.02


# ---------------------------------------------------
# FORECAST INTERPRETER
# (Former DecisionEngine — upgraded)
# ---------------------------------------------------

class ForecastInterpreter:
    """
    Interprets model forecasts and computes confidence.
    """

    def interpret(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float
    ) -> Tuple[str, float]:

        confidence = (
            abs(predicted_return) * 0.5 +
            abs(sentiment) * 0.3 +
            abs(50 - rsi) / 50 * 0.2
        )

        if predicted_return > 0.02 and sentiment > 0 and rsi < 70:
            return "BUY", round(confidence, 3)

        if predicted_return < -0.02 and sentiment < 0:
            return "SELL", round(confidence, 3)

        return "HOLD", round(confidence, 3)


# ---------------------------------------------------
# RISK GATE
# ---------------------------------------------------

class RiskGate:
    """
    Applies portfolio-level protections.
    """

    def __init__(self, config: SignalConfig):
        self.config = config

    def allow(
        self,
        signal: str,
        prob_up: float,
        volatility: float | None
    ) -> bool:

        if volatility is not None and volatility > self.config.volatility_cap:
            return False

        if signal == "BUY" and prob_up < self.config.prob_threshold:
            return False

        if signal == "SELL" and prob_up > (1 - self.config.prob_threshold):
            return False

        return True


# ---------------------------------------------------
# ENSEMBLE ARBITER
# ---------------------------------------------------

class EnsembleArbiter:
    """
    Combines multiple model outputs into final decision.
    """

    def decide(
        self,
        base_signal: str,
        prob_up: float,
        lstm_prices: List[float],
        prophet_trend: str,
        config: SignalConfig
    ) -> str:

        if not lstm_prices or len(lstm_prices) < 2:
            return "HOLD"

        expected_return = (
            lstm_prices[-1] - lstm_prices[0]
        ) / lstm_prices[0]

        if (
            base_signal == "BUY"
            and prophet_trend == "BULLISH"
            and expected_return > config.min_expected_return
            and prob_up >= config.prob_threshold
        ):
            return "BUY"

        if (
            base_signal == "SELL"
            and prophet_trend == "BEARISH"
            and expected_return < -config.min_expected_return
            and prob_up <= (1 - config.prob_threshold)
        ):
            return "SELL"

        return "HOLD"


# ---------------------------------------------------
# MASTER DECISION ENGINE
# ---------------------------------------------------

class DecisionEngine:
    """
    Institutional-style decision stack.

    Flow:
    forecast → interpret → risk gate → ensemble → final signal
    """

    def __init__(self, config: SignalConfig = SignalConfig()):

        self.config = config
        self.interpreter = ForecastInterpreter()
        self.risk_gate = RiskGate(config)
        self.ensemble = EnsembleArbiter()

    def generate(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float,
        prob_up: float,
        volatility: float,
        lstm_prices: List[float],
        prophet_trend: str
    ):

        base_signal, confidence = self.interpreter.interpret(
            predicted_return,
            sentiment,
            rsi
        )

        if not self.risk_gate.allow(
            base_signal,
            prob_up,
            volatility
        ):
            return "HOLD", confidence

        final_signal = self.ensemble.decide(
            base_signal,
            prob_up,
            lstm_prices,
            prophet_trend,
            self.config
        )

        return final_signal, confidence
