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
import numpy as np
import pandas as pd
import logging

from core.signals.signal_engine import DecisionEngine
from core.schema.feature_schema import MODEL_FEATURES


logger = logging.getLogger("marketsentinel.signal_generator")


class SignalGenerator:
    """
    Institutional Signal Generator v4

    - Cross-sectional ranking
    - Probability spread guard
    - Z-score filtering
    - Edge filtering
    - Walk-forward compatible (returns signals, probs)
    """

    ###################################################

    MIN_SPREAD = 0.04          # minimum cross-sectional spread
    MIN_Z_SCORE = 0.5          # minimum normalized conviction
    TOP_K_PERCENT = 0.30       # trade only strongest 30%

    ###################################################

    def __init__(self):
        self.engine = DecisionEngine()

    ###################################################
    # CROSS SECTION NORMALIZATION
    ###################################################

    def _normalize_probs(self, probs):

        mean = np.mean(probs)
        std = np.std(probs)

        if std < 1e-6:
            return np.zeros_like(probs)

        return (probs - mean) / std

    ###################################################
    # SPREAD GUARD
    ###################################################

    def _spread_ok(self, probs):

        spread = float(np.max(probs) - np.min(probs))

        if spread < self.MIN_SPREAD:
            logger.info("Probability spread too small — skipping day.")
            return False

        return True

    ###################################################
    # GENERATE SIGNALS
    ###################################################

    def generate(self, model, df_slice):

        if df_slice.empty:
            return [], []

        X = df_slice.loc[:, MODEL_FEATURES]

        probs = model.predict_proba(X)[:, 1]

        # --- SPREAD CHECK ---
        if not self._spread_ok(probs):
            return ["HOLD"] * len(probs), probs

        z_scores = self._normalize_probs(probs)

        df = pd.DataFrame({
            "ticker": df_slice["ticker"].values,
            "prob": probs,
            "z": z_scores
        })

        # Filter weak conviction
        df = df[np.abs(df["z"]) >= self.MIN_Z_SCORE]

        if df.empty:
            return ["HOLD"] * len(probs), probs

        # Rank strongest signals
        k = max(1, int(len(df) * self.TOP_K_PERCENT))
        df = df.sort_values("z", ascending=False).head(k)

        selected = set(df["ticker"].values)

        signals = []

        for i, row in enumerate(df_slice.itertuples()):

            if row.ticker not in selected:
                signals.append("HOLD")
                continue

            prob = probs[i]

            # Convert probability to directional forecast
            forecast_up = prob * 0.05
            forecast_down = (1 - prob) * -0.04

            decision = self.engine.generate(
                ticker=row.ticker,
                price_df=None,
                current_price=row.close,
                forecast_up=float(forecast_up),
                forecast_down=float(forecast_down),
                prob_up=float(prob),
                volatility=getattr(row, "volatility", 0.02),
                regime=getattr(row, "regime", None)
            )

            signals.append(decision["signal"])

        return signals, probs

        return signals
