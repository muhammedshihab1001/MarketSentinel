import joblib
import tensorflow as tf

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet


class ModelLoader:

    def __init__(self):

        self.xgb = joblib.load("artifacts/xgboost/model.pkl")

        self.lstm = tf.keras.models.load_model(
            "artifacts/lstm/lstm_price_forecast.h5",
            compile=False
        )

        self.lstm_scaler = joblib.load(
            "artifacts/lstm/lstm_scaler.pkl"
        )

        self.prophet = joblib.load(
            "artifacts/prophet/prophet_trend.pkl"
        )

    # -----------------------------

    def lstm_forecast(self, recent_prices):

        return forecast_lstm(
            self.lstm,
            self.lstm_scaler,
            recent_prices
        )

    # -----------------------------

    def prophet_forecast(self):

        return forecast_prophet(self.prophet)
