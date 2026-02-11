from prophet import Prophet
import pandas as pd
import numpy as np


CHANGEPOINT_PRIOR = 0.02
INTERVAL_WIDTH = 0.80


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

    prophet_df = prophet_df.sort_values("ds")

    prophet_df = prophet_df.drop_duplicates("ds")

    # Light outlier protection (winsorization)
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

def train_prophet(df):

    prophet_df = prepare_prophet_dataframe(df)

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=CHANGEPOINT_PRIOR,
        interval_width=INTERVAL_WIDTH
    )

    model.fit(prophet_df)

    return model


# ---------------------------------------------------
# FORECAST
# ---------------------------------------------------

def forecast_prophet(model, periods=30):

    future = model.make_future_dataframe(periods=periods)

    forecast = model.predict(future)

    tail = forecast.tail(periods)

    if tail["yhat"].isna().any():
        raise RuntimeError(
            "Forecast contains NaN values."
        )

    yhat = tail["yhat"]
    upper = tail["yhat_upper"]
    lower = tail["yhat_lower"]

    trend = "BULLISH" if yhat.iloc[-1] > yhat.iloc[0] else "BEARISH"

    volatility = float(np.std(yhat.values))

    return {
        "trend": trend,
        "upper": float(upper.max()),
        "lower": float(lower.min()),
        "forecast_volatility": volatility
    }
