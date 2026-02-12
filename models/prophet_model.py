from prophet import Prophet
import pandas as pd
import numpy as np

from cmdstanpy import cmdstan_path


CHANGEPOINT_PRIOR = 0.05
INTERVAL_WIDTH = 0.80
CHANGEPOINT_RANGE = 0.90
UNCERTAINTY_SAMPLES = 120

MIN_HISTORY = 350
EPSILON = 1e-9


########################################################

def _validate_cmdstan():

    try:

        path = cmdstan_path()

        if path is None:
            raise RuntimeError

    except Exception:

        raise RuntimeError(
            "CmdStan is not installed. "
            "Run: python -m cmdstanpy.install_cmdstan"
        )


########################################################

def prepare_prophet_dataframe(df):

    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Expected columns ['ds','y'], found {df.columns.tolist()}"
        )

    prophet_df = df.copy()

    prophet_df["ds"] = pd.to_datetime(
        prophet_df["ds"]
    ).dt.tz_localize(None)

    prophet_df["y"] = prophet_df["y"].astype("float64")

    prophet_df = prophet_df.sort_values("ds")
    prophet_df = prophet_df.drop_duplicates("ds")

    if len(prophet_df) < MIN_HISTORY:
        raise RuntimeError(
            "Insufficient history for structural trend detection."
        )

    if (prophet_df["y"] <= 0).any():
        raise RuntimeError("Prices must be positive for log transform.")

    # log transform
    prophet_df["y"] = np.log(prophet_df["y"])

    # soft winsorization (safer than hard clipping)
    lower = prophet_df["y"].quantile(0.005)
    upper = prophet_df["y"].quantile(0.995)

    prophet_df["y"] = prophet_df["y"].clip(lower, upper)

    return prophet_df


########################################################

def train_prophet(df, random_seed: int = 42):

    _validate_cmdstan()

    prophet_df = prepare_prophet_dataframe(df)

    n_changepoints = min(15, len(prophet_df) // 25)

    model = Prophet(

        growth="linear",

        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,

        changepoint_prior_scale=CHANGEPOINT_PRIOR,
        changepoint_range=CHANGEPOINT_RANGE,
        n_changepoints=n_changepoints,

        interval_width=INTERVAL_WIDTH,
        uncertainty_samples=UNCERTAINTY_SAMPLES,

        seasonality_mode="multiplicative",

        stan_backend="CMDSTANPY"
    )

    ####################################################
    # Add institutional-grade seasonality
    ####################################################

    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5
    )

    ####################################################

    model.fit(
        prophet_df,
        seed=random_seed
    )

    return model


########################################################

def forecast_prophet(model, periods=60):

    future = model.make_future_dataframe(
        periods=periods,
        freq="D"
    )

    forecast = model.predict(future)

    tail = forecast.tail(periods)

    if tail[["yhat", "yhat_upper", "yhat_lower"]].isna().any().any():
        raise RuntimeError("Forecast contains NaN values.")

    yhat_log = tail["yhat"].values
    yhat = np.exp(yhat_log)

    slope = float(yhat[-1] - yhat[0])
    volatility = float(np.std(yhat))

    normalized_slope = slope / max(yhat[0], EPSILON)

    if abs(normalized_slope) < volatility / max(yhat[0], EPSILON) * 0.30:
        trend = "SIDEWAYS"
    else:
        trend = "BULLISH" if normalized_slope > 0 else "BEARISH"

    return {
        "trend": trend,
        "upper": float(np.exp(tail["yhat_upper"]).max()),
        "lower": float(np.exp(tail["yhat_lower"]).min()),
        "forecast_volatility": volatility,
        "normalized_slope": normalized_slope
    }
