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
