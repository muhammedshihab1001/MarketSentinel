import datetime
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from app.services.data_fetcher import StockPriceFetcher
from app.models.lstm_model import build_lstm_model

LOOKBACK_WINDOW = 60
FORECAST_HORIZON = 30

END_DATE = datetime.date.today().isoformat()

fetcher = StockPriceFetcher()
df = fetcher.fetch("AAPL", "2018-01-01", END_DATE)

if df.empty:
    raise ValueError("No price data fetched for LSTM training")

if "close" not in df.columns:
    raise ValueError(f"Expected 'close' column, found {df.columns.tolist()}")

prices = df[["close"]].values

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)


def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_prices, LOOKBACK_WINDOW)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = build_lstm_model((LOOKBACK_WINDOW, 1))
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("models/lstm_price_forecast.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")

print("LSTM model & scaler saved")
