from core.signals.signal_engine import SignalEngine

engine = SignalEngine()

# Strong bullish case
signal = engine.generate_signal(
    prediction=1,
    prob_up=0.72,
    avg_sentiment=0.35,
    volatility=0.02
)
print("Signal:", signal)

# Strong bearish case
signal = engine.generate_signal(
    prediction=0,
    prob_up=0.25,
    avg_sentiment=-0.30,
    volatility=0.03
)
print("Signal:", signal)

# High risk case
signal = engine.generate_signal(
    prediction=1,
    prob_up=0.80,
    avg_sentiment=0.50,
    volatility=0.12
)
print("Signal:", signal)
