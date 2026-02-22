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


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    weight = neg / pos
    weight = float(np.clip(weight, 0.9, 5.0))

    logger.info("Computed class weight = %.3f", weight)

    return weight


###################################################
# SAFE PRODUCTION MODEL
###################################################

class SafeXGBClassifier:

    def __init__(self, pos_weight):
        self.pos_weight = pos_weight
        self.model = None
        self.feature_names = None
        self.feature_checksum = None
        self.training_checksum = None
        self.param_checksum = None

        np.random.seed(SEED)

    ###################################################
    # TRAIN
    ###################################################

    def fit(self, X, y):

        if X is None or X.empty:
            raise RuntimeError("Training feature matrix empty.")

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in training features.")

        X = X.astype(np.float32)
        y = np.asarray(y)

        # 🔒 Lock feature order
        self.feature_names = list(X.columns)

        checksum_str = json.dumps(self.feature_names, sort_keys=False)
        self.feature_checksum = hashlib.sha256(
            checksum_str.encode()
        ).hexdigest()

        # 🔒 Training data checksum
        training_bytes = X.to_numpy().tobytes()
        self.training_checksum = hashlib.sha256(training_bytes).hexdigest()

        dtrain = xgb.DMatrix(X, label=y)

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

        # 🔐 Parameter fingerprint
        self.param_checksum = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        # 🔒 Simple validation split for early stopping
        val_split = int(0.9 * len(X))
        dtrain_part = xgb.DMatrix(X.iloc[:val_split], label=y[:val_split])
        dval_part = xgb.DMatrix(X.iloc[val_split:], label=y[val_split:])

        self.model = xgb.train(
            params,
            dtrain_part,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dval_part, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )

        preds = self.model.predict(dtrain)

        mean = float(np.mean(preds))
        std = float(np.std(preds))

        logger.info(
            "Train prob stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
            mean,
            std,
            float(np.min(preds)),
            float(np.max(preds))
        )

        if std < MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected.")

        if np.any(preds < 0) or np.any(preds > 1):
            raise RuntimeError("Invalid probability range detected.")

        # 🔒 Booster feature name integrity
        booster_features = self.model.feature_names
        if booster_features != self.feature_names:
            raise RuntimeError("Booster feature order mismatch.")

        return self

    ###################################################
    # INFERENCE
    ###################################################

    def predict_proba(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        if X is None or X.empty:
            raise RuntimeError("Inference feature matrix empty.")

        missing = set(self.feature_names) - set(X.columns)
        extra = set(X.columns) - set(self.feature_names)

        if missing:
            raise RuntimeError(f"Missing inference features: {missing}")

        if extra:
            logger.warning("Extra inference features ignored: %s", extra)

        X = X.loc[:, self.feature_names].astype(np.float32)

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in inference features.")

        checksum_str = json.dumps(self.feature_names, sort_keys=False)
        current_checksum = hashlib.sha256(
            checksum_str.encode()
        ).hexdigest()

        if current_checksum != self.feature_checksum:
            raise RuntimeError("Feature order integrity violated.")

        dmatrix = xgb.DMatrix(X)
        probs = self.model.predict(dmatrix)

        if np.std(probs) < MIN_PROB_STD:
            raise RuntimeError("Inference probability collapse detected.")

        if np.any(probs < 0) or np.any(probs > 1):
            raise RuntimeError("Invalid probability range detected.")

        return np.column_stack([1 - probs, probs])

    ###################################################
    # FEATURE IMPORTANCE
    ###################################################

    def export_feature_importance(self):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        score = self.model.get_score(importance_type="gain")

        sorted_imp = sorted(
            score.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Top 5 features:")
        for name, val in sorted_imp[:5]:
            logger.info("  %s : %.4f", name, val)

        return sorted_imp


###################################################
# BUILD MODEL
###################################################

def build_xgboost_pipeline(y):

    pos_weight = compute_class_weight(y)
    model = SafeXGBClassifier(pos_weight)

    logger.info("XGBoost model built (deterministic CPU mode, early stopping enabled).")

    return model