from prophet import Prophet
import pandas as pd
import numpy as np

from cmdstanpy import cmdstan_path


CHANGEPOINT_PRIOR = 0.01
INTERVAL_WIDTH = 0.70
CHANGEPOINT_RANGE = 0.80
UNCERTAINTY_SAMPLES = 200


# ---------------------------------------------------
# CMDSTAN SAFETY CHECK
# ---------------------------------------------------

def _validate_cmdstan():
    """
    Fail closed if CmdStan is unavailable.
    No manual path resolution.
    Let cmdstanpy manage installation paths.
    """

    try:
        path = cmdstan_path()

        if path is None:
            raise RuntimeError

    except Exception:
        raise RuntimeError(
            "CmdStan is not installed. "
            "Run: python -m cmdstanpy.install_cmdstan"
        )


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

    # Conservative winsorization
    lower = prophet_df["y"].quantile(0.02)
    upper = prophet_df["y"].quantile(0.98)

    prophet_df["y"] = prophet_df["y"].clip(lower, upper)

    if len(prophet_df) < 300:
        raise RuntimeError(
            "Insufficient history for structural trend detection."
        )

    return prophet_df


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

def train_prophet(df, random_seed: int = 42):

    _validate_cmdstan()

    prophet_df = prepare_prophet_dataframe(df)

    model = Prophet(
        growth="linear",

        # Structural trend only
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,

        changepoint_prior_scale=CHANGEPOINT_PRIOR,
        changepoint_range=CHANGEPOINT_RANGE,
        interval_width=INTERVAL_WIDTH,
        uncertainty_samples=UNCERTAINTY_SAMPLES,

        stan_backend="CMDSTANPY"
    )

    model.fit(
        prophet_df,
        seed=random_seed
    )

    return model


# ---------------------------------------------------
# FORECAST
# ---------------------------------------------------

def forecast_prophet(model, periods=60):

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

    yhat = tail["yhat"].values

    slope = float(yhat[-1] - yhat[0])
    volatility = float(np.std(yhat))

    if abs(slope) < volatility * 0.25:
        trend = "SIDEWAYS"
    else:
        trend = "BULLISH" if slope > 0 else "BEARISH"

    return {
        "trend": trend,
        "upper": float(tail["yhat_upper"].max()),
        "lower": float(tail["yhat_lower"].min()),
        "forecast_volatility": volatility,
        "slope": slope
    }
