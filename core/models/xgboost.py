"""
MarketSentinel v4.5.0

XGBoost regression model wrapper with safer inference handling,
feature integrity checks, and production-like robustness.
"""

import hashlib
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("marketsentinel.xgboost")

# ── Training constants ─────────────────────────────────────────────

SEED = 42
NUM_BOOST_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 75
VALIDATION_SPLIT = 0.15

MIN_BOOST_ROUNDS = 25
MIN_TRAIN_ROWS = 120

# ── Sanity thresholds ──────────────────────────────────────────────

MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
MAX_MODEL_SIZE_MB = 150
MAX_ABS_SCORE = 50
EPSILON = 1e-12

MIN_VALIDATION_ROWS = 50
MAX_TARGET_ABS = 50


class SafeXGBRegressor:

    STANDARDIZE_INFERENCE = True

    def __init__(self):

        self.model = None
        self.feature_names = None
        self.training_cols = None

        self.feature_checksum = None
        self.param_checksum = None
        self.importance_checksum = None
        self.training_fingerprint = None
        self.booster_checksum = None

        self.best_iteration = None
        self.train_rmse = None
        self.valid_rmse = None

        self.model_type = "xgboost_regressor"

        random.seed(SEED)
        np.random.seed(SEED)

    # ==========================================================
    # FIT (STRICT - KEEP AS IS)
    # ==========================================================

    def fit(self, X, y):

        if X is None or X.empty:
            raise RuntimeError("Training feature matrix is empty.")

        if len(X) < MIN_TRAIN_ROWS:
            raise RuntimeError(f"Training dataset too small ({len(X)} rows).")

        if len(X) != len(y):
            raise RuntimeError("Feature/label length mismatch.")

        if X.isnull().any().any():
            raise RuntimeError("NaN values detected in training features.")

        X = X.astype(np.float32)
        y = np.asarray(y, dtype=np.float32)

        if np.std(y) < MIN_TARGET_STD:
            raise RuntimeError(f"Target variance too small ({np.std(y):.2e}).")

        y = np.clip(y, -MAX_TARGET_ABS, MAX_TARGET_ABS)

        self.feature_names = list(X.columns)
        self.training_cols = X.shape[1]

        self.training_fingerprint = hashlib.sha256(
            X.to_numpy().tobytes() + y.tobytes()
        ).hexdigest()

        self.feature_checksum = hashlib.sha256(
            json.dumps(sorted(self.feature_names)).encode()
        ).hexdigest()

        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))

        if len(X) - split_idx < MIN_VALIDATION_ROWS:
            split_idx = len(X) - MIN_VALIDATION_ROWS

        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]

        X_valid = X.iloc[split_idx:]
        y_valid = y[split_idx:]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, "train")]

        if len(X_valid) >= MIN_VALIDATION_ROWS:
            dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=self.feature_names)
            evals.append((dvalid, "valid"))

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "eta": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2,
            "gamma": 0.05,
            "reg_alpha": 0.3,
            "reg_lambda": 1.2,
            "tree_method": "hist",
            "seed": SEED,
            "verbosity": 0,
        }

        self.param_checksum = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
            evals_result=evals_result,
        )

        self.best_iteration = getattr(self.model, "best_iteration", NUM_BOOST_ROUNDS)

        # FIX: Ensure at least 1 boosting round is used for prediction
        if self.best_iteration < 1:
            logger.warning("Model stopped extremely early (iter=%d).", self.best_iteration)
            self.best_iteration = 1
        elif self.best_iteration < MIN_BOOST_ROUNDS:
            logger.warning("Model stopped extremely early (iter=%d).", self.best_iteration)

        self.train_rmse = float(evals_result["train"]["rmse"][-1])
        self.valid_rmse = float(evals_result.get("valid", {}).get("rmse", [self.train_rmse])[-1])

        self.booster_checksum = hashlib.sha256(
            self.model.save_raw().tobytes() if hasattr(self.model.save_raw(), 'tobytes') else self.model.save_raw()
        ).hexdigest()

        logger.info(
            "XGBoost trained | iter=%d | features=%d | train_rmse=%.5f | valid_rmse=%.5f",
            self.best_iteration,
            len(self.feature_names),
            self.train_rmse,
            self.valid_rmse,
        )

        return self

    # ==========================================================
    # PREDICT (FIXED + SAFE)
    # ==========================================================

    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        if X is None:
            raise RuntimeError("Inference input is None.")

        # 🔥 FIX 1: Convert numpy → DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        if X.empty:
            raise RuntimeError("Inference features empty.")

        # 🔥 FIX 2: Add missing features safely
        missing = set(self.feature_names) - set(X.columns)

        if missing:
            logger.warning("Missing features filled with 0: %s", list(missing))
            for col in missing:
                X[col] = 0.0

        # enforce order
        X = X.reindex(columns=self.feature_names)

        # 🔥 FIX 3: Handle NaN / inf safely
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        X = X.astype(np.float32)

        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

        scores = self.model.predict(
            dmatrix,
            iteration_range=(0, self.best_iteration),
        )

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Non-finite inference scores.")

        scores = np.clip(scores, -MAX_ABS_SCORE, MAX_ABS_SCORE)

        score_std = float(np.std(scores))

        if score_std < MIN_SCORE_STD:
            logger.warning("Low score dispersion (std=%.2e).", score_std)

        if self.STANDARDIZE_INFERENCE and score_std > MIN_SCORE_STD:
            scores = (scores - np.mean(scores)) / (score_std + EPSILON)

        return scores.astype(np.float32)


# ==========================================================
# FACTORY
# ==========================================================

def build_xgboost_pipeline():
    logger.info("Building XGBoost Regressor | MarketSentinel v4.5.0")
    return SafeXGBRegressor()