from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import logging
import threading
from sklearn.model_selection import train_test_split

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_PROB_STD = 1e-5
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_ROUNDS = 40

###################################################
# GPU DETECTION
###################################################

_GPU_AVAILABLE = None
_GPU_LOCK = threading.Lock()


def _gpu_available():
    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    with _GPU_LOCK:
        if _GPU_AVAILABLE is not None:
            return _GPU_AVAILABLE

        try:
            dtrain = xgb.DMatrix(
                np.random.rand(16, 4),
                label=np.random.randint(0, 2, 16)
            )

            xgb.train(
                {"tree_method": "gpu_hist", "max_depth": 1, "verbosity": 0},
                dtrain,
                num_boost_round=1
            )

            _GPU_AVAILABLE = True
            logger.info("CUDA backend verified.")
        except Exception:
            _GPU_AVAILABLE = False
            logger.info("CUDA unavailable — using CPU.")

        return _GPU_AVAILABLE


def _tree_method():
    return "gpu_hist" if _gpu_available() else "hist"


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    if len(y) == 0:
        raise RuntimeError("Empty labels passed to XGBoost.")

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    weight = neg / pos
    weight = float(np.clip(weight, 0.9, 5.0))

    logger.info("Computed class weight = %.3f", weight)

    return weight


###################################################
# SAFE CLASSIFIER
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):

        if not hasattr(X, "shape"):
            raise RuntimeError("Invalid feature matrix.")

        arr = X.to_numpy() if hasattr(X, "to_numpy") else X

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature values detected.")

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        # 🔥 Internal validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y
        )

        model = super().fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )

        preds = model.predict_proba(X)[:, 1]

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
            raise RuntimeError("Probability collapse detected after training.")

        return model

    ###################################################
    # FEATURE IMPORTANCE EXPORT
    ###################################################

    def export_feature_importance(self, feature_names):

        booster = self.get_booster()
        score = booster.get_score(importance_type="gain")

        importances = {
            feature_names[int(k[1:])] if k.startswith("f") else k: v
            for k, v in score.items()
        }

        sorted_imp = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Top 5 features:")
        for name, val in sorted_imp[:5]:
            logger.info("  %s : %.4f", name, val)

        return sorted_imp


###################################################
# PARAM BUILDER
###################################################

def _base_params(pos_weight):

    params = dict(

        n_estimators=800,  # allow early stopping to choose
        max_depth=3,
        learning_rate=0.03,

        subsample=0.65,
        colsample_bytree=0.6,
        colsample_bylevel=0.6,

        min_child_weight=6,
        gamma=0.25,
        reg_alpha=1.2,
        reg_lambda=3.5,

        max_bin=256,

        tree_method=_tree_method(),
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        scale_pos_weight=pos_weight,
        verbosity=0
    )

    logger.info(
        "XGBoost params | depth=%s trees=%s lr=%.3f",
        params["max_depth"],
        params["n_estimators"],
        params["learning_rate"]
    )

    return params


###################################################
# BUILD MODEL
###################################################

def build_xgboost_pipeline(y):

    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    model = SafeXGBClassifier(**params)

    logger.info("XGBoost model built successfully.")

    return model