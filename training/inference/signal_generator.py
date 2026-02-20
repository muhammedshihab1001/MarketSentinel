import numpy as np
import pandas as pd
import logging

from core.signals.signal_engine import DecisionEngine
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema
)

logger = logging.getLogger("marketsentinel.signal_generator")


class SignalGenerator:
    """
    Institutional Signal Generator v2

    Improvements:
    - Hard probability band
    - Symmetric long/short selection
    - Cross-sectional z normalization
    - Spread guard
    - Conviction filter
    - Deterministic output
    """

    MIN_SPREAD = 0.05
    MIN_Z_SCORE = 0.75

    PROB_BUY = 0.65
    PROB_SELL = 0.35

    TOP_K_PERCENT = 0.25
    MIN_PROB_STD = 1e-5

    ###################################################

    def __init__(self):
        self.engine = DecisionEngine()

    ###################################################
    # VALIDATION
    ###################################################

    def _validate_probabilities(self, probs):

        if not np.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities detected.")

        std = float(np.std(probs))

        if std < self.MIN_PROB_STD:
            logger.warning("Probability collapse detected — returning HOLD.")
            return False

        return True

    ###################################################
    # NORMALIZATION
    ###################################################

    def _normalize_probs(self, probs):

        mean = np.mean(probs)
        std = np.std(probs)

        if std < 1e-8:
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
    # GENERATE
    ###################################################

    def generate(self, model, df_slice):

        if df_slice is None or df_slice.empty:
            return [], []

        features = validate_feature_schema(
            df_slice.loc[:, MODEL_FEATURES]
        )

        probs = model.predict_proba(features)[:, 1]

        if not self._validate_probabilities(probs):
            return ["HOLD"] * len(probs), probs

        if not self._spread_ok(probs):
            return ["HOLD"] * len(probs), probs

        z_scores = self._normalize_probs(probs)

        df = pd.DataFrame({
            "ticker": df_slice["ticker"].values,
            "prob": probs,
            "z": z_scores
        })

        # Hard probability band
        df["direction"] = np.where(
            df["prob"] >= self.PROB_BUY, 1,
            np.where(df["prob"] <= self.PROB_SELL, -1, 0)
        )

        # Remove neutral
        df = df[df["direction"] != 0]

        if df.empty:
            return ["HOLD"] * len(probs), probs

        # Conviction filter
        df = df[np.abs(df["z"]) >= self.MIN_Z_SCORE]

        if df.empty:
            return ["HOLD"] * len(probs), probs

        k = max(1, int(len(df) * self.TOP_K_PERCENT))

        # Long side
        longs = df[df["direction"] == 1] \
            .sort_values("z", ascending=False) \
            .head(k)

        # Short side
        shorts = df[df["direction"] == -1] \
            .sort_values("z", ascending=True) \
            .head(k)

        selected = set(longs["ticker"]).union(
            set(shorts["ticker"])
        )

        signals = []

        for i, row in enumerate(df_slice.itertuples(index=False)):

            if row.ticker not in selected:
                signals.append("HOLD")
                continue

            prob = float(probs[i])

            direction = 1 if prob >= self.PROB_BUY else -1

            decision = self.engine.generate(
                ticker=row.ticker,
                price_df=None,
                current_price=row.close,
                forecast_up=prob,
                forecast_down=1 - prob,
                prob_up=prob,
                volatility=float(getattr(row, "volatility", 0.02)),
                regime=getattr(row, "regime_feature", None)
            )

            signals.append(decision["signal"])

        return signals, probs