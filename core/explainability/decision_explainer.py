class DecisionExplainer:
    """
    Generates human-readable explanations for trading signals.
    """

    def explain(
        self,
        prediction: int,
        prob_up: float,
        sentiment: float,
        volatility: float,
        rsi: float
    ):

        reasons = []

        # Model conviction
        if prob_up > 0.6:
            reasons.append("Model shows strong upward probability")

        elif prob_up < 0.4:
            reasons.append("Model shows strong downward probability")

        # Sentiment
        if sentiment > 0.2:
            reasons.append("Market sentiment is positive")

        elif sentiment < -0.2:
            reasons.append("Market sentiment is negative")

        # Volatility
        if volatility > 0.04:
            reasons.append("Market is highly volatile")

        else:
            reasons.append("Market volatility is stable")

        # RSI context
        if rsi > 70:
            reasons.append("Asset is overbought")

        elif rsi < 30:
            reasons.append("Asset is oversold")

        # Final explanation string
        explanation = " | ".join(reasons)

        return explanation
