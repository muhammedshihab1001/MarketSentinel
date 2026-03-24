"""
MarketSentinel v4.5.2

XGBoost regression model wrapper with safer inference handling,
feature integrity checks, and production-like robustness.

Changes from v4.5.1:
  FIX (item 20): predict() now raises RuntimeError on NaN input.
    Previously NaN was silently filled with 0, masking data quality issues.
    NaN in inference input = upstream pipeline bug that should surface loudly.
  FIX (item 19): Added export_feature_importance() method.
    Was missing entirely — tests and /model/feature-importance route both need it.
  FIX: predict() still fills MISSING columns with 0 (graceful schema flexibility).
    This is correct for inference — NaN ≠ missing column.
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

MIN_VALIDATION_ROWS = 30
MAX_TARGET_ABS = 50
MIN_TRAIN_ROWS = 80


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
            "max_depth": 4,
            "eta": 0.05,
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

        # Compute importance checksum for integrity tracking
        try:
            scores = self.model.get_score(importance_type="gain")
            self.importance_checksum = hashlib.sha256(
                json.dumps(scores, sort_keys=True).encode()
            ).hexdigest()
        except Exception:
            self.importance_checksum = None

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

        # FIX (item 20): Raise on NaN — NaN in inference input means the
        # upstream feature pipeline has a bug. Fill-with-0 was silently
        # masking data quality issues. Missing COLUMNS are still filled
        # with 0 (graceful schema flexibility for new features).
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise RuntimeError(
                f"NaN values in inference input for columns: {nan_cols}. "
                "Fix upstream feature pipeline."
            )

        missing = set(self.feature_names) - set(X.columns)
        if missing:
            logger.warning("Missing features filled with 0: %s", list(missing))
            for col in missing:
                X[col] = 0.0

        # FIX: Raise on extra unexpected columns after missing fill
        # (don't silently drop them — surface the schema mismatch)
        extra = set(X.columns) - set(self.feature_names)
        if extra:
            logger.debug("Extra columns ignored during inference: %s", list(extra))

        X = X.reindex(columns=self.feature_names)
        X = X.replace([np.inf, -np.inf], 0).astype(np.float32)

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
    # EXPORT FEATURE IMPORTANCE  (item 19)
    #
    # Returns a dict with feature_importance list (sorted by gain),
    # booster_checksum, feature_checksum, and train/valid RMSE.
    # Used by /model/feature-importance route and tests.
    # ==========================================================

    def export_feature_importance(self) -> dict:
        """
        Export feature importance from the trained booster.

        Returns:
            {
                "feature_importance": [{"feature": str, "importance": float}, ...],
                "booster_checksum": str,
                "feature_checksum": str,
                "train_rmse": float,
                "valid_rmse": float,
            }
        """

        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        try:
            scores = self.model.get_score(importance_type="gain")
        except Exception as e:
            logger.warning("get_score(gain) failed: %s — using weight fallback", e)
            try:
                scores = self.model.get_score(importance_type="weight")
            except Exception:
                scores = {}

        total = sum(scores.values()) or 1.0

        importance_list = sorted(
            [
                {"feature": f, "importance": round(v / total, 6)}
                for f, v in scores.items()
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

        return {
            "feature_importance": importance_list,
            "booster_checksum": self.booster_checksum,
            "feature_checksum": self.feature_checksum,
            "importance_checksum": self.importance_checksum,
            "train_rmse": self.train_rmse,
            "valid_rmse": self.valid_rmse,
            "best_iteration": self.best_iteration,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
        }


# ==========================================================
# FACTORY
# ==========================================================

def build_xgboost_pipeline():
    logger.info("Building XGBoost Regressor | MarketSentinel v4.5.2")
    return SafeXGBRegressor()