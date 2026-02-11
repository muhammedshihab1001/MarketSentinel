from prophet import Prophet
import pandas as pd
import numpy as np
import os

from cmdstanpy import cmdstan_path, set_cmdstan_path

CHANGEPOINT_PRIOR = 0.02
INTERVAL_WIDTH = 0.80
CHANGEPOINT_RANGE = 0.9


# ---------------------------------------------------
# CMDSTAN RESOLUTION (INSTITUTIONAL)
# ---------------------------------------------------

def _resolve_cmdstan():
    """
    Institutional CmdStan resolver.

    Resolution order:
    1. CMDSTAN env
    2. cmdstanpy installed path
    3. artifacts/cmdstan
    """

    # 1. Explicit env
    env_path = os.getenv("CMDSTAN")

    if env_path:

        if not os.path.exists(env_path):
            raise RuntimeError(
                f"CMDSTAN env path invalid: {env_path}"
            )

        set_cmdstan_path(env_path)
        return env_path

    # 2. cmdstanpy default install
    try:

        default_path = cmdstan_path()

        if default_path and os.path.exists(default_path):
            set_cmdstan_path(default_path)
            return default_path

    except Exception:
        pass

    # 3. Institutional artifact location
    artifact_path = "artifacts/cmdstan"

    if os.path.exists(artifact_path):
        set_cmdstan_path(artifact_path)
        return artifact_path

    # FAIL CLOSED
    raise RuntimeError(
        "CmdStan not found.\n"
        "Install during image build or environment bootstrap.\n"
        "Runtime downloads are forbidden."
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

    resolved_path = _resolve_cmdstan()

    prophet_df = prepare_prophet_dataframe(df)

    model = Prophet(
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=CHANGEPOINT_PRIOR,
        changepoint_range=CHANGEPOINT_RANGE,
        interval_width=INTERVAL_WIDTH,
        uncertainty_samples=1000,
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
