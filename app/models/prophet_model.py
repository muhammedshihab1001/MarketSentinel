from prophet import Prophet


def train_prophet(df):
    prophet_df = df[["Date", "Close"]].rename(
        columns={"Date": "ds", "Close": "y"}
    )

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(prophet_df)
    return model
