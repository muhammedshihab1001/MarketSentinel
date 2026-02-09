from core.signals.signal_engine import DecisionEngine

engine = DecisionEngine()

# ---------------------------------------------------
# Strong Bullish Case
# ---------------------------------------------------

signal, confidence = engine.generate(
    predicted_return=0.04,   # +4% expected
    sentiment=0.35,
    rsi=55,
    prob_up=0.72,
    volatility=0.02,
    lstm_prices=[100, 101, 103, 104],
    prophet_trend="BULLISH"
)

print("Bullish Case ->", signal, "| confidence:", confidence)


# ---------------------------------------------------
# Strong Bearish Case
# ---------------------------------------------------

signal, confidence = engine.generate(
    predicted_return=-0.05,
    sentiment=-0.30,
    rsi=48,
    prob_up=0.25,
    volatility=0.03,
    lstm_prices=[100, 98, 97, 95],
    prophet_trend="BEARISH"
)

print("Bearish Case ->", signal, "| confidence:", confidence)


# ---------------------------------------------------
# High Risk (Should HOLD)
# ---------------------------------------------------

signal, confidence = engine.generate(
    predicted_return=0.06,
    sentiment=0.50,
    rsi=60,
    prob_up=0.80,
    volatility=0.12,  # risk gate trigger
    lstm_prices=[100, 103, 106],
    prophet_trend="BULLISH"
)

print("High Risk Case ->", signal, "| confidence:", confidence)
