import xgboost as xgb
import numpy as np
import logging
import hashlib
import json

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_PROB_STD = 1e-5
NUM_BOOST_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 50
MIN_VALIDATION_ROWS = 300
MIN_MINORITY_SAMPLES = 50
EPSILON = 1e-12


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    if min(pos, neg) < MIN_MINORITY_SAMPLES:
        logger.warning(
            "Very small minority class detected (pos=%.0f neg=%.0f).",
            pos, neg
        )

    weight = neg / pos
    weight = float(np.clip(weight, 0.5, 10.0))

    logger.info(
        "Class distribution | pos=%.0f neg=%.0f scale_pos_weight=%.3f",
        pos, neg, weight
    )

    return weight


###################################################
# SAFE PRODUCTION MODEL
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
        y = np.asarray(y)

        self.training_rows = X.shape[0]
        self.training_cols = X.shape[1]

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            self.training_rows,
            self.training_cols
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
        # SAFE TIME SPLIT
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
            feature_names=self.feature_names
        )

        ###################################################
        # PARAMETERS
        ###################################################

        params = {
            "max_depth": 3,
            "eta": 0.03,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 6,
            "gamma": 0.25,
            "reg_alpha": 1.2,
            "reg_lambda": 3.5,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cpu",
            "scale_pos_weight": self.pos_weight,
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
                feature_names=self.feature_names
            )

            self.model = xgb.train(
                params,
                dtrain_part,
                num_boost_round=NUM_BOOST_ROUNDS,
                evals=[(dval_part, "validation")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False
            )

        else:
            self.model = xgb.train(
                params,
                dtrain_part,
                num_boost_round=NUM_BOOST_ROUNDS,
                verbose_eval=False
            )

        ###################################################
        # TRAIN SANITY CHECK
        ###################################################

        dtrain_full = xgb.DMatrix(
            X,
            feature_names=self.feature_names
        )

        preds = self.model.predict(dtrain_full)

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite predictions detected.")

        # 🔥 KEEP collapse guard during training only
        if np.std(preds) < MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected during training.")

        if np.any(preds < -EPSILON) or np.any(preds > 1 + EPSILON):
            raise RuntimeError("Invalid probability range detected.")

        if self.model.feature_names != self.feature_names:
            raise RuntimeError("Booster feature order mismatch.")

        logger.info("XGBoost training completed successfully.")

        return self

    ###################################################
    # INFERENCE
    ###################################################

    def predict_proba(self, X):

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
            feature_names=self.feature_names
        )

        probs = self.model.predict(dmatrix)

        if not np.all(np.isfinite(probs)):
            raise RuntimeError("Non-finite inference probabilities.")

        # 🔥 REMOVED inference-time collapse exception
        if np.std(probs) < MIN_PROB_STD:
            logger.warning("Inference probability collapse detected.")

        if np.any(probs < -EPSILON) or np.any(probs > 1 + EPSILON):
            raise RuntimeError("Invalid probability range detected.")

        return np.column_stack([1 - probs, probs])

    ###################################################
    # FEATURE IMPORTANCE
    ###################################################

    def export_feature_importance(self):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        raw_gain = self.model.get_score(importance_type="gain")

        if not raw_gain:
            raise RuntimeError("No feature importance available.")

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
            reverse=True
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
            "importance_checksum": self.importance_checksum
        }


###################################################
# BUILD MODEL
###################################################

def build_xgboost_pipeline(y):

    pos_weight = compute_class_weight(y)

    model = SafeXGBClassifier(pos_weight)

    logger.info(
        "XGBoost model built (deterministic CPU mode, early stopping safe)."
    )

    return model