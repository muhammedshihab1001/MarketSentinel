import xgboost as xgb
import numpy as np
import logging
import hashlib
import json

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
NUM_BOOST_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 150
MIN_VALIDATION_ROWS = 300
MIN_MINORITY_SAMPLES = 50
EPSILON = 1e-12


###################################################
# CLASS WEIGHT (LEGACY SAFE — UNUSED IN REGRESSION)
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))

    if pos == 0 or neg == 0:
        logger.warning(
            "Label collapse detected (pos=%.0f neg=%.0f) — allowed in regression mode.",
            pos, neg
        )
        return 1.0

    if min(pos, neg) < MIN_MINORITY_SAMPLES:
        logger.warning(
            "Very small minority class detected (pos=%.0f neg=%.0f).",
            pos, neg
        )

    weight = neg / pos
    weight = float(np.clip(weight, 0.5, 10.0))

    logger.info(
        "Class distribution | pos=%.0f neg=%.0f (weight unused in regression mode)",
        pos,
        neg,
    )

    return weight


###################################################
# SAFE REGRESSION MODEL
###################################################

class SafeXGBClassifier:

    def __init__(self, pos_weight):
        self.pos_weight = float(pos_weight)
        self.model = None
        self.feature_names = None
        self.feature_checksum = None
        self.training_checksum = None
        self.param_checksum = None
        self.training_rows = None
        self.training_cols = None
        self.importance_checksum = None

        np.random.seed(SEED)

    ###################################################
    # TRAIN
    ###################################################

    def fit(self, X, y):

        if X is None or X.empty:
            raise RuntimeError("Training feature matrix empty.")

        if len(X) != len(y):
            raise RuntimeError("Feature/label length mismatch.")

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in training features.")

        X = X.copy().astype(np.float32)
        y = np.asarray(y, dtype=np.float32)

        if np.std(y) < MIN_TARGET_STD:
            raise RuntimeError("Target variance too small — cannot train alpha model.")

        self.training_rows = X.shape[0]
        self.training_cols = X.shape[1]

        logger.info(
            "XGBoost training | rows=%s cols=%s | target_std=%.8f",
            self.training_rows,
            self.training_cols,
            float(np.std(y)),
        )

        ###################################################
        # LOCK FEATURE CONTRACT
        ###################################################

        self.feature_names = list(X.columns)

        checksum_str = json.dumps(self.feature_names, sort_keys=False)
        self.feature_checksum = hashlib.sha256(
            checksum_str.encode()
        ).hexdigest()

        training_bytes = X.to_numpy().tobytes()
        self.training_checksum = hashlib.sha256(
            training_bytes
        ).hexdigest()

        ###################################################
        # TIME SPLIT FOR EARLY STOPPING
        ###################################################

        n = len(X)
        use_early_stopping = n >= MIN_VALIDATION_ROWS * 2
        split_idx = int(n * 0.9)

        if split_idx <= 0 or split_idx >= n:
            raise RuntimeError("Invalid validation split.")

        X_train_part = X.iloc[:split_idx]
        y_train_part = y[:split_idx]

        X_val_part = X.iloc[split_idx:]
        y_val_part = y[split_idx:]

        if len(X_val_part) < MIN_VALIDATION_ROWS:
            use_early_stopping = False
            logger.warning("Validation set too small — disabling early stopping.")

        dtrain_part = xgb.DMatrix(
            X_train_part,
            label=y_train_part,
            feature_names=self.feature_names,
        )

        ###################################################
        # IMPROVED REGRESSION PARAMETERS
        ###################################################

        params = {
            "max_depth": 7,
            "eta": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "objective": "reg:pseudohubererror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": "cpu",
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

        if use_early_stopping:

            dval_part = xgb.DMatrix(
                X_val_part,
                label=y_val_part,
                feature_names=self.feature_names,
            )

            self.model = xgb.train(
                params,
                dtrain_part,
                num_boost_round=NUM_BOOST_ROUNDS,
                evals=[(dval_part, "validation")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )

        else:
            self.model = xgb.train(
                params,
                dtrain_part,
                num_boost_round=NUM_BOOST_ROUNDS,
                verbose_eval=False,
            )

        ###################################################
        # TRAIN SANITY CHECK
        ###################################################

        dtrain_full = xgb.DMatrix(
            X,
            feature_names=self.feature_names,
        )

        preds = self.model.predict(dtrain_full)

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite predictions detected.")

        if np.std(preds) < MIN_SCORE_STD:
            raise RuntimeError("Score collapse detected during training.")

        if self.model.feature_names != self.feature_names:
            raise RuntimeError("Booster feature order mismatch.")

        logger.info(
            "XGBoost regression training completed successfully | pred_std=%.8f",
            float(np.std(preds)),
        )

        return self

    ###################################################
    # TRUE REGRESSION PREDICTION
    ###################################################

    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        if X is None or X.empty:
            raise RuntimeError("Inference feature matrix empty.")

        if len(X.columns) != self.training_cols:
            raise RuntimeError("Inference feature dimension mismatch.")

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

        scores = self.model.predict(dmatrix)

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Non-finite inference scores.")

        if np.std(scores) < MIN_SCORE_STD:
            logger.warning("Inference score collapse detected.")

        return scores

    ###################################################
    # BACKWARD-COMPATIBLE PROBA WRAPPER
    ###################################################

    def predict_proba(self, X):
        scores = self.predict(X)
        scaled = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - scaled, scaled])

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

        total_gain = sum(importance.values())

        normalized = {
            k: (v / total_gain if total_gain > 0 else 0.0)
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

        logger.info("Top 5 normalized feature importances:")
        for name, val in sorted_imp[:5]:
            logger.info("  %s : %.4f", name, val)

        return {
            "feature_importance": sorted_imp,
            "importance_checksum": self.importance_checksum,
        }


###################################################
# BUILD MODEL
###################################################

def build_xgboost_pipeline(y):

    pos_weight = compute_class_weight(y)

    model = SafeXGBClassifier(pos_weight)

    logger.info(
        "XGBoost REGRESSION alpha model built (robust objective, deterministic CPU mode)."
    )

    return model