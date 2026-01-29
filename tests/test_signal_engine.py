from app.services.signal_engine import SignalEngine


def test_buy_signal():
    engine = SignalEngine()
    assert engine.generate_signal(1, 0.7, 0.2, 0.01) == "BUY"


def test_sell_signal():
    engine = SignalEngine()
    assert engine.generate_signal(0, 0.2, -0.3, 0.01) == "SELL"


def test_hold_signal_high_volatility():
    engine = SignalEngine()
    assert engine.generate_signal(1, 0.9, 0.5, 0.2) == "HOLD"
