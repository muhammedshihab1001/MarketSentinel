import xgboost as xgb
import numpy as np
import logging
import hashlib
import json
import random
import os

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
NUM_BOOST_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 75
VALIDATION_SPLIT = 0.15

MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
MIN_PRED_ENTROPY = 0.01
MAX_MODEL_SIZE_MB = 150
MAX_ABS_SCORE = 50
EPSILON = 1e-12

STANDARDIZE_INFERENCE = True  # improves cross-sectional ranking stability


class SafeXGBRegressor:

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

        os.environ["PYTHONHASHSEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    ###################################################
    # TRAIN
    ###################################################

    def fit(self, X, y):

        if X is None or X.empty:
            raise RuntimeError("Training features empty.")

        if len(X) != len(y):
            raise RuntimeError("Feature/label length mismatch.")

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in training features.")

        X = X.astype(np.float32)
        y = np.asarray(y, dtype=np.float32)

        if np.std(y) < MIN_TARGET_STD:
            raise RuntimeError("Target variance too small.")

        self.feature_names = list(X.columns)
        self.training_cols = X.shape[1]

        ###################################################
        # TRAINING FINGERPRINT
        ###################################################

        dataset_payload = X.to_numpy().tobytes() + y.tobytes()
        self.training_fingerprint = hashlib.sha256(dataset_payload).hexdigest()

        ###################################################
        # FEATURE CHECKSUM
        ###################################################

        checksum_str = json.dumps(self.feature_names, sort_keys=False)
        self.feature_checksum = hashlib.sha256(
            checksum_str.encode()
        ).hexdigest()

        ###################################################
        # TIME ORDERED SPLIT
        ###################################################

        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))

        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]

        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self.feature_names,
        )

        dvalid = xgb.DMatrix(
            X_valid,
            label=y_valid,
            feature_names=self.feature_names,
        )

        ###################################################
        # PARAMETERS (Stable for Noisy Yahoo Data)
        ###################################################

        use_gpu = os.getenv("XGB_USE_GPU", "false").lower() == "true"

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
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "seed": SEED,
            "nthread": 1,
            "verbosity": 0,
        }

        self.param_checksum = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        ###################################################
        # TRAIN
        ###################################################

        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
            evals_result=evals_result
        )

        # Robust best_iteration handling
        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            self.best_iteration = self.model.best_iteration
        else:
            self.best_iteration = NUM_BOOST_ROUNDS

        self.train_rmse = float(evals_result["train"]["rmse"][-1])
        self.valid_rmse = float(evals_result["valid"]["rmse"][-1])

        ###################################################
        # OVERFIT CHECK
        ###################################################

        if self.valid_rmse > self.train_rmse * 1.5:
            logger.warning(
                "Potential overfitting | train_rmse=%.5f | valid_rmse=%.5f",
                self.train_rmse,
                self.valid_rmse
            )

        ###################################################
        # MODEL SIZE CHECK
        ###################################################

        raw_model = self.model.save_raw()
        model_size_mb = len(raw_model) / (1024 * 1024)

        if model_size_mb > MAX_MODEL_SIZE_MB:
            raise RuntimeError("Model artifact too large.")

        self.booster_checksum = hashlib.sha256(raw_model).hexdigest()

        ###################################################
        # TRAINING SANITY CHECK
        ###################################################

        preds = self.model.predict(
            dtrain,
            iteration_range=(0, self.best_iteration)
        )

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite training predictions.")

        pred_std = float(np.std(preds))

        if pred_std < MIN_SCORE_STD:
            raise RuntimeError("Score collapse detected.")

        hist, _ = np.histogram(preds, bins=30)
        probs = hist / (hist.sum() + EPSILON)
        entropy = -np.sum(probs * np.log(probs + EPSILON))

        if entropy < MIN_PRED_ENTROPY:
            raise RuntimeError("Prediction entropy too low.")

        logger.info(
            "XGB trained | pred_std=%.6f | entropy=%.4f | train_rmse=%.5f | valid_rmse=%.5f",
            pred_std,
            entropy,
            self.train_rmse,
            self.valid_rmse
        )

        return self

    ###################################################
    # PREDICT
    ###################################################

    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        if X is None or X.empty:
            raise RuntimeError("Inference feature matrix empty.")

        if len(X.columns) != self.training_cols:
            raise RuntimeError("Feature dimension mismatch.")

        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise RuntimeError(f"Missing inference features: {missing}")

        X = X.loc[:, self.feature_names].astype(np.float32)

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in inference features.")

        dmatrix = xgb.DMatrix(
            X,
            feature_names=self.feature_names,
        )

        scores = self.model.predict(
            dmatrix,
            iteration_range=(0, self.best_iteration)
        )

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Non-finite inference scores.")

        scores = np.clip(scores, -MAX_ABS_SCORE, MAX_ABS_SCORE)

        score_std = float(np.std(scores))

        if score_std < MIN_SCORE_STD:
            logger.warning("Inference score dispersion low.")

        # Optional cross-sectional normalization
        if STANDARDIZE_INFERENCE and score_std > MIN_SCORE_STD:
            scores = (scores - np.mean(scores)) / (score_std + EPSILON)

        return scores

    ###################################################
    # FEATURE IMPORTANCE
    ###################################################

    def export_feature_importance(self):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        raw_gain = self.model.get_score(importance_type="gain")

        importance = {
            name: float(raw_gain.get(name, 0.0))
            for name in self.feature_names
        }

        total_gain = sum(importance.values()) + EPSILON

        normalized = {
            k: (v / total_gain)
            for k, v in importance.items()
        }

        sorted_imp = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        checksum_str = json.dumps(sorted_imp, sort_keys=False)
        self.importance_checksum = hashlib.sha256(
            checksum_str.encode()
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


def build_xgboost_pipeline():
    logger.info("Building XGBoost Regression Model (Stable CV Version)")
    return SafeXGBRegressor()