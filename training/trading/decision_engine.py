class DecisionEngine:
    """
    Combines forecast + sentiment + indicators
    to produce trading decisions.
    """

    def generate_signal(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float
    ):

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
