from __future__ import annotations

import warnings
import os
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


########################################################
# STRICT NUMERIC DETERMINISM
########################################################

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


MIN_HISTORY = 300
EPSILON = 1e-9
MAX_FORWARD_FILL_RATIO = 0.05

MAX_FORECAST_MULTIPLIER = 5.0   # prevents explosion
MIN_FORECAST_MULTIPLIER = 0.2


########################################################
# CONFIG
########################################################

@dataclass(frozen=True)
class SarimaxConfig:
    order: tuple = (1, 1, 1)
    seasonal_order: tuple = (0, 0, 0, 0)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 200
    trend: Optional[str] = None
    frequency: str = "B"


########################################################
# DATA PREP
########################################################

def prepare_series(df: pd.DataFrame, freq: str) -> pd.Series:

    if df is None or df.empty:
        raise RuntimeError("Empty dataframe supplied to SARIMAX.")

    required = {"date", "close"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"SARIMAX requires columns {required}. Missing: {missing}"
        )

    series = df.copy()

    series["date"] = pd.to_datetime(
        series["date"],
        utc=True,
        errors="raise"
    )

    series = series.sort_values("date")

    if len(series) < MIN_HISTORY:
        raise RuntimeError(
            f"Insufficient history for SARIMAX. Required >= {MIN_HISTORY}, got {len(series)}"
        )

    if (series["close"] <= 0).any():
        raise RuntimeError("Non-positive prices detected.")

    if series["close"].isna().any():
        raise RuntimeError("NaNs detected in price series.")

    ########################################################
    # LOG TRANSFORM
    ########################################################

    series["close"] = np.log(
        series["close"].astype("float64")
    )

    series = series.set_index("date")["close"]

    ########################################################
    # FORCE BUSINESS FREQUENCY
    ########################################################

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=freq
    )

    reindexed = series.reindex(full_index)

    fill_ratio = reindexed.isna().mean()

    if fill_ratio > MAX_FORWARD_FILL_RATIO:
        raise RuntimeError(
            f"Too many missing sessions ({fill_ratio:.2%}). "
            "Provider integrity questionable."
        )

    reindexed = reindexed.ffill()

    reindexed.index.freq = freq

    return reindexed


########################################################
# MODEL WRAPPER
########################################################

class SarimaxModel:

    def __init__(self, config: Optional[SarimaxConfig] = None):
        self.config = config or SarimaxConfig()
        self._fitted = None
        self._train_index = None
        self._fingerprint = None

    ########################################################
    # TRAIN
    ########################################################

    def fit(self, df: pd.DataFrame) -> "SarimaxModel":

        series = prepare_series(
            df,
            freq=self.config.frequency
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning
            )

            try:

                model = SARIMAX(
                    series,
                    order=self.config.order,
                    seasonal_order=self.config.seasonal_order,
                    enforce_stationarity=self.config.enforce_stationarity,
                    enforce_invertibility=self.config.enforce_invertibility,
                    trend=self.config.trend,
                )

                fitted = model.fit(
                    disp=False,
                    maxiter=self.config.maxiter
                )

            except ConvergenceWarning as exc:
                raise RuntimeError(
                    "SARIMAX failed to converge."
                ) from exc

            except Exception as exc:
                raise RuntimeError(
                    f"SARIMAX training failure: {exc}"
                ) from exc

        if not getattr(fitted, "mle_retvals", {}).get("converged", False):
            raise RuntimeError(
                "SARIMAX optimizer did not converge."
            )

        self._fitted = fitted
        self._train_index = series.index
        self._fingerprint = self._hash_model()

        return self

    ########################################################
    # MODEL HASH (CRITICAL)
    ########################################################

    def _hash_model(self) -> str:

        params = np.ascontiguousarray(
            self._fitted.params.astype("float64")
        )

        return hashlib.sha256(params.tobytes()).hexdigest()

    def fingerprint(self) -> str:

        if self._fingerprint is None:
            raise RuntimeError("Model not trained.")

        return self._fingerprint

    ########################################################
    # FORECAST
    ########################################################

    def forecast(self, steps: int = 60) -> Dict[str, Any]:

        if self._fitted is None:
            raise RuntimeError(
                "Attempted forecast before model training."
            )

        if steps <= 0:
            raise RuntimeError(
                "Forecast steps must be positive."
            )

        try:
            forecast = self._fitted.forecast(
                steps=steps
            )
        except Exception as exc:
            raise RuntimeError(
                f"SARIMAX forecast failure: {exc}"
            ) from exc

        if np.isnan(forecast).any():
            raise RuntimeError(
                "SARIMAX produced NaN forecast."
            )

        prices = np.exp(
            forecast.values.astype("float64")
        )

        ####################################################
        # EXPLOSION GUARD
        ####################################################

        start_price = prices[0]

        if (
            prices.max() > start_price * MAX_FORECAST_MULTIPLIER
            or prices.min() < start_price * MIN_FORECAST_MULTIPLIER
        ):
            raise RuntimeError(
                "SARIMAX forecast exploded — regime shift suspected."
            )

        slope = float(prices[-1] - prices[0])
        volatility = float(np.std(prices))

        normalized_slope = slope / max(
            prices[0],
            EPSILON
        )

        if abs(normalized_slope) < (
            volatility / max(prices[0], EPSILON)
        ) * 0.35:
            trend = "SIDEWAYS"
        else:
            trend = "BULLISH" if normalized_slope > 0 else "BEARISH"

        return {
            "trend": trend,
            "forecast_volatility": volatility,
            "normalized_slope": normalized_slope,
            "model_fingerprint": self._fingerprint
        }

    ########################################################
    # METADATA
    ########################################################

    def get_params(self) -> Dict[str, Any]:

        return {
            "order": self.config.order,
            "seasonal_order": self.config.seasonal_order,
            "trend": self.config.trend,
            "frequency": self.config.frequency,
        }

    def get_training_range(self) -> Dict[str, str]:

        if self._train_index is None:
            raise RuntimeError(
                "Training metadata unavailable."
            )

        return {
            "start": str(self._train_index.min()),
            "end": str(self._train_index.max()),
        }

    ########################################################
    # ARTIFACT ACCESS
    ########################################################

    @property
    def fitted_model(self):

        if self._fitted is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        return self._fitted
