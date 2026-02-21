import xgboost as xgb
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_PROB_STD = 1e-5
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_ROUNDS = 40


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
# SAFE PRODUCTION MODEL (CPU LOCKED)
###################################################

class SafeXGBClassifier:

    def __init__(self, pos_weight):
        self.pos_weight = pos_weight
        self.model = None
        self.feature_names = None

    def fit(self, X, y):

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        # Enforce deterministic dtype
        X = X.astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y
        )

        self.feature_names = list(X.columns)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "max_depth": 3,
            "eta": 0.03,
            "subsample": 0.65,
            "colsample_bytree": 0.6,
            "min_child_weight": 6,
            "gamma": 0.25,
            "reg_alpha": 1.2,
            "reg_lambda": 3.5,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",      # XGBoost 2.x compliant
            "device": "cpu",            # Explicit CPU lock
            "scale_pos_weight": self.pos_weight,
            "seed": SEED,
            "verbosity": 0,
        }

        evals = [(dtrain, "train"), (dval, "validation")]

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=800,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )

        preds = self.model.predict(xgb.DMatrix(X))

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

        return self

    def predict_proba(self, X):
        X = X.astype(np.float32)
        dmatrix = xgb.DMatrix(X)
        probs = self.model.predict(dmatrix)
        return np.column_stack([1 - probs, probs])

    def export_feature_importance(self):

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

    logger.info("XGBoost model built successfully (CPU mode).")

    return model