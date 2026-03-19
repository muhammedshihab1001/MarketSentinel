"""
MarketSentinel v4.5.1

XGBoost regression model wrapper with safer inference handling,
feature integrity checks, and production-like robustness.

Changes from v4.5.0:
  FIX: NUM_BOOST_ROUNDS reduced 1200 → 400
       EARLY_STOPPING_ROUNDS reduced 75 → 30
       MIN_BOOST_ROUNDS reduced 25 → 10

  Reason: with ~193 rows per ticker × 30 tickers = ~5790 training
  rows and 15% validation split (~870 rows), the previous settings
  caused XGBoost to stop at iteration 0-22 because the validation
  loss plateau was reached immediately on tiny data.

  These settings will produce meaningful training on current data
  (~9 months). Revert to NUM_BOOST_ROUNDS=1200 / EARLY_STOPPING=75
  once 2+ years of data are available for 100 tickers.
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

# ── Training constants ─────────────────────────────────────────

SEED = 42

# Adjusted for current data volume (~9 months, 30 tickers)
# Revert to 1200/75/25 when 2+ years × 100 tickers available
NUM_BOOST_ROUNDS = 400
EARLY_STOPPING_ROUNDS = 30
MIN_BOOST_ROUNDS = 10

VALIDATION_SPLIT = 0.15

# ── Sanity thresholds ──────────────────────────────────────────

MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
MAX_MODEL_SIZE_MB = 150
MAX_ABS_SCORE = 50
EPSILON = 1e-12

MIN_VALIDATION_ROWS = 30   # lowered from 50 — matches reduced data volume
MAX_TARGET_ABS = 50
MIN_TRAIN_ROWS = 80        # lowered from 120 — CV portfolio may have sparse tickers


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
    # FIT
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
            split_idx = max(len(X) - MIN_VALIDATION_ROWS, int(len(X) * 0.7))

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
            "max_depth": 4,        # reduced from 6 — less overfit on small data
            "eta": 0.05,           # slightly higher than 0.03 — faster convergence
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
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

        if self.best_iteration < 1:
            logger.warning("Model stopped at iter=0. Forcing iter=1.")
            self.best_iteration = 1
        elif self.best_iteration < MIN_BOOST_ROUNDS:
            logger.warning(
                "Model stopped very early (iter=%d). "
                "Consider adding more training data.",
                self.best_iteration,
            )

        self.train_rmse = float(evals_result["train"]["rmse"][-1])
        self.valid_rmse = float(
            evals_result.get("valid", {}).get("rmse", [self.train_rmse])[-1]
        )

        raw = self.model.save_raw()
        self.booster_checksum = hashlib.sha256(
            raw.tobytes() if hasattr(raw, "tobytes") else raw
        ).hexdigest()

        logger.info(
            "XGBoost trained | iter=%d | features=%d | "
            "train_rmse=%.5f | valid_rmse=%.5f",
            self.best_iteration,
            len(self.feature_names),
            self.train_rmse,
            self.valid_rmse,
        )

        return self

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        if X is None:
            raise RuntimeError("Inference input is None.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        if X.empty:
            raise RuntimeError("Inference features empty.")

        missing = set(self.feature_names) - set(X.columns)
        if missing:
            logger.warning("Missing features filled with 0: %s", list(missing))
            for col in missing:
                X[col] = 0.0

        X = X.reindex(columns=self.feature_names)
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
    logger.info("Building XGBoost Regressor | MarketSentinel v4.5.1")
    return SafeXGBRegressor()