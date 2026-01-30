import joblib
import tensorflow as tf

XGBOOST_PATH = "models/xgboost_direction.pkl"
LSTM_PATH = "models/lstm_price_forecast.h5"
LSTM_SCALER_PATH = "models/lstm_scaler.pkl"
PROPHET_PATH = "models/prophet_trend.pkl"


class ModelLoader:
    def __init__(self):
        self.xgb = joblib.load(XGBOOST_PATH)
        self.lstm = tf.keras.models.load_model(LSTM_PATH)
        self.lstm_scaler = joblib.load(LSTM_SCALER_PATH)
        self.prophet = joblib.load(PROPHET_PATH)
