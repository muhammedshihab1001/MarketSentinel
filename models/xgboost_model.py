import xgboost as xgb
import numpy as np
import logging

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_PROB_STD = 1e-5
NUM_BOOST_ROUNDS = 600


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

    ###################################################
    # TRAIN
    ###################################################

    def fit(self, X, y):

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in training features.")

        X = X.astype(np.float32)

        # 🔒 Lock feature order permanently
        self.feature_names = list(X.columns)

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
            "verbosity": 0,
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            verbose_eval=False
        )

        # 🔎 Probability sanity check
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

        return self

    ###################################################
    # INFERENCE
    ###################################################

    def predict_proba(self, X):

        if self.model is None:
            raise RuntimeError("Model not trained.")

        # 🔒 Strict feature contract enforcement
        missing = set(self.feature_names) - set(X.columns)
        extra = set(X.columns) - set(self.feature_names)

        if missing:
            raise RuntimeError(f"Missing inference features: {missing}")

        if extra:
            logger.warning("Extra inference features ignored: %s", extra)

        # Enforce training order exactly
        X = X.loc[:, self.feature_names].astype(np.float32)

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in inference features.")

        dmatrix = xgb.DMatrix(X)
        probs = self.model.predict(dmatrix)

        if np.std(probs) < MIN_PROB_STD:
            raise RuntimeError("Inference probability collapse detected.")

        return np.column_stack([1 - probs, probs])

    ###################################################
    # FEATURE IMPORTANCE
    ###################################################

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

    logger.info("XGBoost model built successfully (CPU deterministic mode).")

    return model