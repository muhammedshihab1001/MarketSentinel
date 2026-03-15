"""
MarketSentinel v4.1.0

XGBoost regression model wrapper with training safety guards,
feature integrity checksums, and standardised inference scoring.

Features:
    - Time-ordered train/validation split (no shuffling — financial data)
    - Early stopping with configurable rounds
    - Feature + parameter + booster checksums for artifact integrity
    - Prediction entropy and dispersion guards
    - Optional GPU support via XGB_USE_GPU env var
    - Configurable thread count via XGB_NTHREAD env var
"""

import hashlib
import json
import logging
import os
import random

import numpy as np
import xgboost as xgb

logger = logging.getLogger("marketsentinel.xgboost")

# ── Training constants ────────────────────────────────────────────────────────
SEED                  = 42
NUM_BOOST_ROUNDS      = 1200
EARLY_STOPPING_ROUNDS = 75
VALIDATION_SPLIT      = 0.15

# ── Sanity thresholds ─────────────────────────────────────────────────────────
MIN_SCORE_STD    = 1e-6
MIN_TARGET_STD   = 1e-6
MIN_PRED_ENTROPY = 0.01
MAX_MODEL_SIZE_MB = 150
MAX_ABS_SCORE    = 50
EPSILON          = 1e-12


class SafeXGBRegressor:
    """
    Production-hardened XGBoost regressor with integrity checks,
    feature validation, and standardised inference output.

    Training flow:
        fit(X, y) → time-ordered split → xgb.train() → sanity checks
    Inference flow:
        predict(X) → feature alignment → clip → optional standardise
    """

    # ── Inference standardisation (class-level — can be overridden per instance)
    STANDARDIZE_INFERENCE = True

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.model              = None
        self.feature_names      = None
        self.training_cols      = None

        # Integrity checksums
        self.feature_checksum   = None
        self.param_checksum     = None
        self.importance_checksum = None
        self.training_fingerprint = None
        self.booster_checksum   = None

        # Training metrics
        self.best_iteration     = None
        self.train_rmse         = None
        self.valid_rmse         = None

        self.model_type = "xgboost_regressor"

        # Seed numpy + random for reproducible training
        # NOTE: PYTHONHASHSEED only works if set before interpreter startup
        # — setting os.environ here at runtime has no effect on hash randomisation.
        # These two seeds are sufficient for XGBoost + numpy reproducibility.
        random.seed(SEED)
        np.random.seed(SEED)

    # ────────────────────────────────────────────────────────────────────────
    # FIT
    # ────────────────────────────────────────────────────────────────────────

    def fit(self, X, y) -> "SafeXGBRegressor":
        """
        Train XGBoost regressor on (X, y).

        Parameters
        ----------
        X : pd.DataFrame — feature matrix (MODEL_FEATURES columns)
        y : array-like   — continuous regression target

        Uses a time-ordered train/validation split (no shuffling).
        """

        # ── Input guards ──────────────────────────────────────────────────────
        if X is None or X.empty:
            raise RuntimeError("Training feature matrix is empty.")
        if len(X) != len(y):
            raise RuntimeError(
                f"Feature/label length mismatch: X={len(X)}, y={len(y)}."
            )
        if X.isnull().any().any():
            raise RuntimeError("NaN values detected in training features.")

        X = X.astype(np.float32)
        y = np.asarray(y, dtype=np.float32)

        if np.std(y) < MIN_TARGET_STD:
            raise RuntimeError(
                f"Target variance too small ({np.std(y):.2e}). "
                "Check label construction."
            )

        self.feature_names  = list(X.columns)
        self.training_cols  = X.shape[1]

        # ── Integrity checksums ───────────────────────────────────────────────
        self.training_fingerprint = hashlib.sha256(
            X.to_numpy().tobytes() + y.tobytes()
        ).hexdigest()

        self.feature_checksum = hashlib.sha256(
            json.dumps(self.feature_names).encode()
        ).hexdigest()

        # ── Time-ordered split ────────────────────────────────────────────────
        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        X_valid, y_valid = X.iloc[split_idx:], y[split_idx:]

        dtrain = xgb.DMatrix(
            X_train, label=y_train, feature_names=self.feature_names
        )
        evals = [(dtrain, "train")]

        if len(X_valid) >= 10:
            dvalid = xgb.DMatrix(
                X_valid, label=y_valid, feature_names=self.feature_names
            )
            evals.append((dvalid, "valid"))

        # ── Parameters ───────────────────────────────────────────────────────
        use_gpu   = os.getenv("XGB_USE_GPU", "false").lower() == "true"
        n_threads = int(os.getenv("XGB_NTHREAD", "-1"))   # -1 = all available cores

        params = {
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "max_depth":        6,
            "eta":              0.03,
            "subsample":        0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2,
            "gamma":            0.05,
            "reg_alpha":        0.3,
            "reg_lambda":       1.2,
            "tree_method":      "gpu_hist" if use_gpu else "hist",
            "seed":             SEED,
            "nthread":          n_threads,
            "verbosity":        0,
        }

        self.param_checksum = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        # ── Train ─────────────────────────────────────────────────────────────
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

        self.best_iteration = (
            self.model.best_iteration
            if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None
            else NUM_BOOST_ROUNDS
        )

        self.train_rmse = float(evals_result["train"]["rmse"][-1])
        self.valid_rmse = (
            float(evals_result["valid"]["rmse"][-1])
            if "valid" in evals_result
            else self.train_rmse
        )

        # ── Overfit check ─────────────────────────────────────────────────────
        if self.valid_rmse > self.train_rmse * 1.5:
            logger.warning(
                "Potential overfitting detected | "
                "train_rmse=%.5f | valid_rmse=%.5f | ratio=%.2f",
                self.train_rmse, self.valid_rmse,
                self.valid_rmse / (self.train_rmse + EPSILON),
            )

        # ── Model size check ──────────────────────────────────────────────────
        # Use "ubj" (universal binary JSON) format — consistent across XGBoost versions
        raw_model      = self.model.save_raw("ubj")
        model_size_mb  = len(raw_model) / (1024 * 1024)

        if model_size_mb > MAX_MODEL_SIZE_MB:
            raise RuntimeError(
                f"Model artifact too large: {model_size_mb:.1f} MB "
                f"(limit={MAX_MODEL_SIZE_MB} MB)."
            )

        self.booster_checksum = hashlib.sha256(raw_model).hexdigest()

        # ── Training sanity check ─────────────────────────────────────────────
        preds = self.model.predict(
            dtrain,
            iteration_range=(0, self.best_iteration),
        )

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite values in training predictions.")

        pred_std = float(np.std(preds))
        if pred_std < MIN_SCORE_STD:
            raise RuntimeError(
                f"Score collapse: prediction std={pred_std:.2e} < {MIN_SCORE_STD}."
            )

        hist, _  = np.histogram(preds, bins=30)
        probs    = hist / (hist.sum() + EPSILON)
        entropy  = float(-np.sum(probs * np.log(probs + EPSILON)))

        if entropy < MIN_PRED_ENTROPY:
            raise RuntimeError(
                f"Prediction entropy too low: {entropy:.4f} < {MIN_PRED_ENTROPY}."
            )

        logger.info(
            "XGBoost trained | best_iter=%d | pred_std=%.6f | entropy=%.4f "
            "| train_rmse=%.5f | valid_rmse=%.5f | size_mb=%.2f",
            self.best_iteration, pred_std, entropy,
            self.train_rmse, self.valid_rmse, model_size_mb,
        )

        return self

    # ────────────────────────────────────────────────────────────────────────
    # PREDICT
    # ────────────────────────────────────────────────────────────────────────

    def predict(self, X) -> np.ndarray:
        """
        Run inference and return standardised scores.

        Parameters
        ----------
        X : pd.DataFrame — must contain all training feature columns

        Returns
        -------
        np.ndarray of float32 scores, optionally standardised to zero-mean unit-std.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        if X is None or X.empty:
            raise RuntimeError("Inference feature matrix is empty.")

        # ── Missing features check FIRST (more informative than col count) ────
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise RuntimeError(
                f"Missing inference features: {missing}. "
                f"Expected {self.training_cols} features."
            )

        # ── Dimension check ───────────────────────────────────────────────────
        if len(X.columns) != self.training_cols:
            raise RuntimeError(
                f"Feature dimension mismatch: got {len(X.columns)}, "
                f"expected {self.training_cols}."
            )

        X = X.loc[:, self.feature_names].astype(np.float32)

        if X.isnull().any().any():
            raise RuntimeError("NaN values detected in inference features.")

        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

        scores = self.model.predict(
            dmatrix,
            iteration_range=(0, self.best_iteration),
        )

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Non-finite values in inference scores.")

        scores = np.clip(scores, -MAX_ABS_SCORE, MAX_ABS_SCORE)

        score_std = float(np.std(scores))
        if score_std < MIN_SCORE_STD:
            logger.warning(
                "Low inference score dispersion (std=%.2e) — "
                "model may be producing flat predictions.",
                score_std,
            )

        if self.STANDARDIZE_INFERENCE and score_std > MIN_SCORE_STD:
            scores = (scores - np.mean(scores)) / (score_std + EPSILON)

        return scores

    # ────────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ────────────────────────────────────────────────────────────────────────

    def export_feature_importance(self) -> dict:
        """
        Export normalised feature importance by gain.
        Returns a dict with importance ranking and all integrity checksums.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        raw_gain   = self.model.get_score(importance_type="gain") or {}
        importance = {
            name: float(raw_gain.get(name, 0.0))
            for name in self.feature_names
        }

        total_gain = sum(importance.values()) + EPSILON
        normalised = {k: v / total_gain for k, v in importance.items()}

        sorted_imp = sorted(
            normalised.items(), key=lambda x: x[1], reverse=True
        )

        self.importance_checksum = hashlib.sha256(
            json.dumps(sorted_imp).encode()
        ).hexdigest()

        return {
            "feature_importance":    sorted_imp,
            "importance_checksum":   self.importance_checksum,
            "booster_checksum":      self.booster_checksum,
            "best_iteration":        self.best_iteration,
            "param_checksum":        self.param_checksum,
            "feature_checksum":      self.feature_checksum,
            "train_rmse":            self.train_rmse,
            "valid_rmse":            self.valid_rmse,
            "training_fingerprint":  self.training_fingerprint,
        }


# ────────────────────────────────────────────────────────────────────────────
# FACTORY
# ────────────────────────────────────────────────────────────────────────────

def build_xgboost_pipeline() -> SafeXGBRegressor:
    """Factory function — returns a fresh untrained SafeXGBRegressor."""
    logger.info("Building XGBoost Regressor | MarketSentinel v4.1.0")
    return SafeXGBRegressor()