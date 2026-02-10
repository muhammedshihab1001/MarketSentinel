from core.risk.position_sizer import PositionSizer


def test_position_never_exceeds_cap():

    sizer = PositionSizer()

    allocation = sizer.size_position(
        signal="BUY",
        confidence=1.0,
        volatility=0.01,
        portfolio_value=100000
    )

    assert allocation <= 10000  # 10% cap
