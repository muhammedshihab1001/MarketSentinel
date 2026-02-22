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
# CLASS WEIGHT (INDUSTRIAL STABLE VERSION)
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    total = pos + neg

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    if min(pos, neg) < MIN_MINORITY_SAMPLES:
        logger.warning(
            "Very small minority class detected (pos=%.0f neg=%.0f).",
            pos, neg
        )

    imbalance_ratio = max(pos, neg) / min(pos, neg)

    logger.info(
        "Class distribution | pos=%.0f neg=%.0f ratio=%.3f",
        pos, neg, imbalance_ratio
    )

    # Standard scale_pos_weight formula
    weight = neg / pos

    # Soft clamp — avoid extreme instability
    weight = float(np.clip(weight, 0.5, 10.0))

    logger.info("Computed scale_pos_weight = %.3f", weight)

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
        # TRAIN PROBABILITY SANITY CHECK
        ###################################################

        dtrain_full = xgb.DMatrix(
            X,
            feature_names=self.feature_names
        )

        preds = self.model.predict(dtrain_full)

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite predictions detected.")

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

        if np.any(preds < -EPSILON) or np.any(preds > 1 + EPSILON):
            raise RuntimeError("Invalid probability range detected.")

        if self.model.feature_names != self.feature_names:
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

        if np.std(probs) < MIN_PROB_STD:
            raise RuntimeError("Inference probability collapse detected.")

        if np.any(probs < -EPSILON) or np.any(probs > 1 + EPSILON):
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

    logger.info(
        "XGBoost model built (deterministic CPU mode, fold-safe early stopping)."
    )

    return model