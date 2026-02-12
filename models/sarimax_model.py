import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


MIN_HISTORY = 300
EPSILON = 1e-9


########################################################
# DATA PREP
########################################################

def prepare_series(df):

    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("SARIMAX requires ['date','close'].")

    series = df.copy()

    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date")

    if len(series) < MIN_HISTORY:
        raise RuntimeError("Insufficient history for SARIMAX.")

    if (series["close"] <= 0).any():
        raise RuntimeError("Invalid prices detected.")

    # log transform stabilizes variance
    series["close"] = np.log(series["close"])

    return series.set_index("date")["close"]


########################################################
# TRAIN
########################################################

def train_sarimax(df):

    series = prepare_series(df)

    model = SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False)

    return fitted


########################################################
# FORECAST
########################################################

def forecast_sarimax(model, steps=60):

    forecast = model.forecast(steps=steps)

    if np.isnan(forecast).any():
        raise RuntimeError("SARIMAX produced NaN forecast.")

    prices = np.exp(forecast.values)

    slope = float(prices[-1] - prices[0])
    volatility = float(np.std(prices))

    normalized_slope = slope / max(prices[0], EPSILON)

    if abs(normalized_slope) < volatility / max(prices[0], EPSILON) * 0.35:
        trend = "SIDEWAYS"
    else:
        trend = "BULLISH" if normalized_slope > 0 else "BEARISH"

    return {
        "trend": trend,
        "forecast_volatility": volatility,
        "normalized_slope": normalized_slope
    }
