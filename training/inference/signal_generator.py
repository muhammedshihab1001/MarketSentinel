import numpy as np
from core.signals.signal_engine import DecisionEngine
from core.schema.feature_schema import MODEL_FEATURES


class SignalGenerator:
    """
    Institutional signal adapter.
    Walk-forward now uses production decision logic.
    """

    def __init__(self):
        self.engine = DecisionEngine()

    ###################################################

    def generate(self, model, df_slice):

        X = df_slice.loc[:, MODEL_FEATURES]

        probs = model.predict_proba(X)[:, 1]

        # Conservative forecast assumptions
        forecast_up = probs * 0.04
        forecast_down = (1 - probs) * -0.03

        signals = {}

        for i, row in enumerate(df_slice.itertuples()):

            decision = self.engine.generate(
                ticker=row.ticker,
                price_df=None,
                current_price=row.close,
                forecast_up=float(forecast_up[i]),
                forecast_down=float(forecast_down[i]),
                prob_up=float(probs[i]),
                volatility=getattr(row, "volatility", 0.02),
                regime=getattr(row, "regime", None)
            )

            signals[row.ticker] = decision["signal"]

        return signals
