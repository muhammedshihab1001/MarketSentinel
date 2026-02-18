import numpy as np

from core.signals.signal_engine import DecisionEngine


class SignalGenerator:
    """
    Institutional signal adapter.

    Ensures walk-forward uses the SAME logic
    as live trading.
    """

    def __init__(self):
        self.engine = DecisionEngine()

    ###################################################

    def generate(self, model, df_slice):
        """
        Returns:
            signals_dict
        """

        probs = model.predict_proba(
            df_slice.iloc[:, df_slice.columns.get_indexer(model.feature_names_in_)]
        )[:, 1]

        forecasts_up = probs * 0.04
        forecasts_down = (1 - probs) * -0.03

        signals = {}

        for i, row in enumerate(df_slice.itertuples()):

            ticker = row.ticker

            decision = self.engine.generate(
                ticker=ticker,
                price_df=None,  # optional for training
                current_price=row.close,
                forecast_up=forecasts_up[i],
                forecast_down=forecasts_down[i],
                prob_up=float(probs[i]),
                volatility=getattr(row, "volatility", 0.02),
                regime=getattr(row, "regime", None)
            )

            signals[ticker] = decision["signal"]

        return signals
