"""
MarketSentinel v4.4.0

XGBoost regression model wrapper with training safety guards,
feature integrity checksums, and standardized inference scoring.
"""

import hashlib
import json
import logging
import os
import random
import time

import numpy as np
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
MIN_PRED_ENTROPY = 0.01
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
    # FIT
    # ==========================================================

    def fit(self, X, y):

        if X is None or X.empty:
            raise RuntimeError("Training feature matrix is empty.")

        if len(X) < MIN_TRAIN_ROWS:
            raise RuntimeError(
                f"Training dataset too small ({len(X)} rows)."
            )

        if len(X) != len(y):
            raise RuntimeError("Feature/label length mismatch.")

        if X.isnull().any().any():
            raise RuntimeError("NaN values detected in training features.")

        X = X.astype(np.float32)
        y = np.asarray(y, dtype=np.float32)

        if np.std(y) < MIN_TARGET_STD:
            raise RuntimeError(
                f"Target variance too small ({np.std(y):.2e})."
            )

        y = np.clip(y, -MAX_TARGET_ABS, MAX_TARGET_ABS)

        self.feature_names = list(X.columns)
        self.training_cols = X.shape[1]

        # fingerprint
        self.training_fingerprint = hashlib.sha256(
            X.to_numpy().tobytes() + y.tobytes()
        ).hexdigest()

        # feature checksum
        self.feature_checksum = hashlib.sha256(
            json.dumps(sorted(self.feature_names)).encode()
        ).hexdigest()

        # ── time-series safe validation split ───────────────

        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))

        if len(X) - split_idx < MIN_VALIDATION_ROWS:
            split_idx = len(X) - MIN_VALIDATION_ROWS

        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]

        X_valid = X.iloc[split_idx:]
        y_valid = y[split_idx:]

        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self.feature_names,
        )

        evals = [(dtrain, "train")]

        if len(X_valid) >= MIN_VALIDATION_ROWS:

            dvalid = xgb.DMatrix(
                X_valid,
                label=y_valid,
                feature_names=self.feature_names,
            )

            evals.append((dvalid, "valid"))

        else:

            logger.warning(
                "Validation set too small (%d rows) — skipping validation.",
                len(X_valid),
            )

        # GPU support
        use_gpu = os.getenv("XGB_USE_GPU", "false").lower() == "true"
        n_threads = int(os.getenv("XGB_NTHREAD", "-1"))

        tree_method = "gpu_hist" if use_gpu else "hist"

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
            "tree_method": tree_method,
            "seed": SEED,
            "nthread": n_threads,
            "verbosity": 0,
        }

        self.param_checksum = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        evals_result = {}

        try:

            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUNDS,
                evals=evals,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
                evals_result=evals_result,
            )

        except xgb.core.XGBoostError:

            logger.warning("GPU unavailable — falling back to CPU.")

            params["tree_method"] = "hist"

            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUNDS,
                evals=evals,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
                evals_result=evals_result,
            )

        self.best_iteration = (
            self.model.best_iteration
            if getattr(self.model, "best_iteration", None) is not None
            else NUM_BOOST_ROUNDS
        )

        if self.best_iteration < MIN_BOOST_ROUNDS:

            logger.warning(
                "Model stopped extremely early (iter=%d).",
                self.best_iteration
            )

        self.train_rmse = float(evals_result["train"]["rmse"][-1])

        if "valid" in evals_result:
            self.valid_rmse = float(evals_result["valid"]["rmse"][-1])
        else:
            self.valid_rmse = self.train_rmse

        raw_model = self.model.save_raw("ubj")

        model_size_mb = len(raw_model) / (1024 * 1024)

        if model_size_mb > MAX_MODEL_SIZE_MB:

            raise RuntimeError(
                f"Model artifact too large: {model_size_mb:.1f} MB"
            )

        self.booster_checksum = hashlib.sha256(raw_model).hexdigest()

        preds = self.model.predict(
            dtrain,
            iteration_range=(0, self.best_iteration),
        )

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite values in training predictions.")

        pred_std = float(np.std(preds))

        if pred_std < MIN_SCORE_STD:
            raise RuntimeError(
                f"Score collapse detected (std={pred_std:.2e})."
            )

        logger.info(
            "XGBoost trained | iter=%d | features=%d | train_rmse=%.5f | valid_rmse=%.5f",
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

        if X is None or X.empty:
            raise RuntimeError("Inference features empty.")

        missing = set(self.feature_names) - set(X.columns)

        if missing:
            raise RuntimeError(f"Missing inference features: {missing}")

        X = X.reindex(columns=self.feature_names)

        X = X.astype(np.float32)

        if X.isnull().any().any():
            raise RuntimeError("NaN values in inference features.")

        dmatrix = xgb.DMatrix(
            X,
            feature_names=self.feature_names,
        )

        scores = self.model.predict(
            dmatrix,
            iteration_range=(0, self.best_iteration),
        )

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Non-finite inference scores.")

        scores = np.clip(scores, -MAX_ABS_SCORE, MAX_ABS_SCORE)

        score_std = float(np.std(scores))

        if score_std < MIN_SCORE_STD:

            logger.warning(
                "Low score dispersion (std=%.2e).",
                score_std,
            )

        if self.STANDARDIZE_INFERENCE and score_std > MIN_SCORE_STD:

            scores = (scores - np.mean(scores)) / (score_std + EPSILON)

        return scores.astype(np.float32)

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    def export_feature_importance(self):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        raw_gain = self.model.get_score(importance_type="gain") or {}

        importance = {
            name: float(raw_gain.get(name, 0.0))
            for name in self.feature_names
        }

        total_gain = sum(importance.values()) + EPSILON

        normalized = {
            k: v / total_gain
            for k, v in importance.items()
        }

        sorted_imp = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        self.importance_checksum = hashlib.sha256(
            json.dumps(sorted_imp).encode()
        ).hexdigest()

        return {
            "feature_importance": sorted_imp,
            "importance_checksum": self.importance_checksum,
            "booster_checksum": self.booster_checksum,
            "best_iteration": self.best_iteration,
            "param_checksum": self.param_checksum,
            "feature_checksum": self.feature_checksum,
            "train_rmse": self.train_rmse,
            "valid_rmse": self.valid_rmse,
            "training_fingerprint": self.training_fingerprint,
        }

    # ==========================================================
    # METADATA EXPORT
    # ==========================================================

    def export_model_metadata(self):

        return {
            "model_type": self.model_type,
            "feature_count": len(self.feature_names),
            "best_iteration": self.best_iteration,
            "train_rmse": self.train_rmse,
            "valid_rmse": self.valid_rmse,
            "feature_checksum": self.feature_checksum,
            "param_checksum": self.param_checksum,
            "booster_checksum": self.booster_checksum,
            "training_fingerprint": self.training_fingerprint,
            "timestamp": int(time.time()),
        }


# ==========================================================
# FACTORY
# ==========================================================

def build_xgboost_pipeline():

    logger.info("Building XGBoost Regressor | MarketSentinel v4.4.0")

    return SafeXGBRegressor()