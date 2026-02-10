from dataclasses import dataclass
from typing import List, Tuple, Dict


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
# ---------------------------------------------------

class ForecastInterpreter:

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


# ===================================================
# 🔥 NEW — STRATEGY ENGINE (Institutional Upgrade)
# ===================================================

class StrategyEngine:
    """
    Portfolio-level intelligence layer.

    Operates ABOVE DecisionEngine.

    Enables:
    ✅ top BUY ranking
    ✅ SELL alerts
    ✅ screeners
    ✅ portfolio construction
    """

    def top_opportunities(
        self,
        predictions: List[Dict],
        top_k: int = 5
    ):
        """
        Returns highest-confidence BUY signals.
        """

        buys = [
            p for p in predictions
            if p.get("signal_today") == "BUY"
        ]

        buys.sort(
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )

        return buys[:top_k]

    # ---------------------------------------------------

    def sell_alerts(
        self,
        predictions: List[Dict]
    ):
        """
        Identify strong SELL signals.
        """

        return [
            p for p in predictions
            if p.get("signal_today") == "SELL"
            and p.get("confidence", 0) > 0.6
        ]

    # ---------------------------------------------------

    def signal_distribution(
        self,
        predictions: List[Dict]
    ):
        """
        Portfolio signal mix.
        Great for dashboards.
        """

        dist = {"BUY": 0, "SELL": 0, "HOLD": 0}

        for p in predictions:
            signal = p.get("signal_today", "HOLD")
            dist[signal] += 1

        return dist
