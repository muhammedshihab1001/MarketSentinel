from prophet import Prophet
import pandas as pd


def train_prophet(df):
    # Validate schema
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"Expected columns ['date', 'close'], found {df.columns.tolist()}"
        )

    prophet_df = df[["date", "close"]].rename(
        columns={"date": "ds", "close": "y"}
    )

    # 🔥 CRITICAL FIX: remove timezone info
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(prophet_df)
    return model

def forecast_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    yhat = forecast["yhat"].tail(periods)
    upper = forecast["yhat_upper"].tail(periods)
    lower = forecast["yhat_lower"].tail(periods)

    trend = "BULLISH" if yhat.iloc[-1] > yhat.iloc[0] else "BEARISH"

    return {
        "trend": trend,
        "upper": float(upper.max()),
        "lower": float(lower.min())
    }
