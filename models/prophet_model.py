from prophet import Prophet
import pandas as pd
import numpy as np


CHANGEPOINT_PRIOR = 0.02
INTERVAL_WIDTH = 0.80
CHANGEPOINT_RANGE = 0.9


# ---------------------------------------------------
# DATA PREP
# ---------------------------------------------------

def prepare_prophet_dataframe(df):

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"Expected columns ['date','close'], found {df.columns.tolist()}"
        )

    prophet_df = df[["date", "close"]].copy()

    prophet_df.rename(
        columns={"date": "ds", "close": "y"},
        inplace=True
    )

    prophet_df["ds"] = pd.to_datetime(
        prophet_df["ds"]
    ).dt.tz_localize(None)

    prophet_df["y"] = prophet_df["y"].astype("float64")

    prophet_df = prophet_df.sort_values("ds")
    prophet_df = prophet_df.drop_duplicates("ds")

    # Winsorization — numeric-safe
    lower = prophet_df["y"].quantile(0.01)
    upper = prophet_df["y"].quantile(0.99)

    prophet_df["y"] = prophet_df["y"].clip(lower, upper)

    if len(prophet_df) < 250:
        raise RuntimeError(
            "Prophet requires sufficient history for stable trend."
        )

    return prophet_df


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

def train_prophet(df, random_seed: int = 42):

    prophet_df = prepare_prophet_dataframe(df)

    model = Prophet(
        growth="linear",                      # explicit
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=CHANGEPOINT_PRIOR,
        changepoint_range=CHANGEPOINT_RANGE,
        interval_width=INTERVAL_WIDTH,
        uncertainty_samples=1000
    )

    # CRITICAL — deterministic Stan backend
    model.fit(
        prophet_df,
        seed=random_seed
    )

    return model


# ---------------------------------------------------
# FORECAST
# ---------------------------------------------------

def forecast_prophet(model, periods=30):

    future = model.make_future_dataframe(
        periods=periods,
        freq="D"
    )

    forecast = model.predict(future)

    tail = forecast.tail(periods)

    if tail[["yhat", "yhat_upper", "yhat_lower"]].isna().any().any():
        raise RuntimeError(
            "Forecast contains NaN values."
        )

    yhat = tail["yhat"]

    trend = (
        "BULLISH"
        if yhat.iloc[-1] > yhat.iloc[0]
        else "BEARISH"
    )

    volatility = float(np.std(yhat.values))

    return {
        "trend": trend,
        "upper": float(tail["yhat_upper"].max()),
        "lower": float(tail["yhat_lower"].min()),
        "forecast_volatility": volatility
    }
