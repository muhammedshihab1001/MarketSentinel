from app.services.signal_engine import SignalEngine, SignalConfig


def test_generate_signal_buy():
    engine = SignalEngine()
    signal = engine.generate_signal(
        prediction=1,
        prob_up=0.75,
        avg_sentiment=0.3,
        volatility=0.02
    )
    assert signal == "BUY"


def test_generate_signal_sell():
    engine = SignalEngine()
    signal = engine.generate_signal(
        prediction=0,
        prob_up=0.20,
        avg_sentiment=-0.4,
        volatility=0.02
    )
    assert signal == "SELL"


def test_generate_signal_hold_due_to_volatility():
    engine = SignalEngine()
    signal = engine.generate_signal(
        prediction=1,
        prob_up=0.9,
        avg_sentiment=0.5,
        volatility=0.2
    )
    assert signal == "HOLD"


def test_fuse_decision_buy():
    engine = SignalEngine()
    signal = engine.fuse_decision(
        direction_signal="BUY",
        prob_up=0.7,
        lstm_prices=[100, 103],
        prophet_trend="BULLISH"
    )
    assert signal == "BUY"


def test_fuse_decision_sell():
    engine = SignalEngine()
    signal = engine.fuse_decision(
        direction_signal="SELL",
        prob_up=0.25,
        lstm_prices=[100, 96],
        prophet_trend="BEARISH"
    )
    assert signal == "SELL"


def test_fuse_decision_hold():
    engine = SignalEngine()
    signal = engine.fuse_decision(
        direction_signal="BUY",
        prob_up=0.55,
        lstm_prices=[100, 100.5],
        prophet_trend="NEUTRAL"
    )
    assert signal == "HOLD"
