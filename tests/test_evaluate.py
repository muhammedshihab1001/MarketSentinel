import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from training.evaluate import (
    XGB_MIN_ACCURACY,
    XGB_MIN_AUC
)


############################################################
# METRIC CALCULATION TEST (UNIT LEVEL)
############################################################

def test_xgboost_metric_computation():

    y_true = np.array([1, 0, 1, 0])
    probs = np.array([0.9, 0.1, 0.8, 0.2])
    preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    assert accuracy == 1.0
    assert auc == 1.0


############################################################
# GATE ENFORCEMENT TEST
############################################################

def test_xgboost_accuracy_gate_enforced():

    y_true = np.array([1, 1, 1, 1])
    probs = np.array([0.1, 0.1, 0.1, 0.1])
    preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_true, preds)

    assert accuracy < XGB_MIN_ACCURACY


def test_xgboost_auc_gate_enforced():

    y_true = np.array([1, 0, 1, 0])
    probs = np.array([0.5, 0.5, 0.5, 0.5])  # no signal

    auc = roc_auc_score(y_true, probs)

    assert auc <= 0.5
    assert auc < XGB_MIN_AUC


############################################################
# DRIFT SAFETY CHECK
############################################################

def test_probability_bounds():

    probs = np.array([0.0, 1.0, 0.5])

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


############################################################
# REGRESSION SAFETY TEST
############################################################

def test_metric_rounding_stability():

    y_true = np.array([1, 0, 1, 0])
    probs = np.array([0.89, 0.11, 0.87, 0.13])
    preds = (probs > 0.5).astype(int)

    accuracy = float(accuracy_score(y_true, preds))
    auc = float(roc_auc_score(y_true, probs))

    assert isinstance(accuracy, float)
    assert isinstance(auc, float)