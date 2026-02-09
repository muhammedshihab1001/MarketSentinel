import numpy as np


def get_xgb_predictions():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    return y_true, y_pred


def get_lstm_predictions():
    y_true = np.array([100, 102, 105, 107])
    y_pred = np.array([101, 103, 104, 108])
    return y_true, y_pred


def get_prophet_predictions():
    y_true = np.array([200, 202, 204, 206])
    y_pred = np.array([201, 203, 205, 207])
    return y_true, y_pred
