import joblib
import tensorflow as tf

XGBOOST_PATH = "artifacts/xgboost_direction.pkl"
LSTM_PATH = "artifacts/lstm_price_forecast.h5"
LSTM_SCALER_PATH = "artifacts/lstm_scaler.pkl"
PROPHET_PATH = "artifacts/prophet_trend.pkl"


class ModelLoader:
    def __init__(self):
        self.xgb = joblib.load(XGBOOST_PATH)
        self.lstm = tf.keras.models.load_model(LSTM_PATH,compile=False)
        self.lstm_scaler = joblib.load(LSTM_SCALER_PATH)
        self.prophet = joblib.load(PROPHET_PATH)
