import xgboost as xgb
import numpy as np
import logging
import hashlib
import json

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
NUM_BOOST_ROUNDS = 600
MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
MIN_GROUP_SIZE = 5
EPSILON = 1e-12


###################################################
# PRODUCTION RANKING MODEL (PAIRWISE RANKING)
###################################################

class SafeXGBRanker:

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.training_cols = None
        self.feature_checksum = None
        self.param_checksum = None
        self.importance_checksum = None

        np.random.seed(SEED)

    ###################################################
    # TRAIN
    ###################################################

    def fit(self, X, y, groups):

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

        if groups is None or len(groups) == 0:
            raise RuntimeError("Ranking requires group definition.")

        if min(groups) < MIN_GROUP_SIZE:
            raise RuntimeError("Cross-sectional group too small.")

        if sum(groups) != len(X):
            raise RuntimeError("Group sizes do not match training rows.")

        self.feature_names = list(X.columns)
        self.training_cols = X.shape[1]

        ###################################################
        # LOCK FEATURE CONTRACT
        ###################################################

        checksum_str = json.dumps(self.feature_names, sort_keys=False)
        self.feature_checksum = hashlib.sha256(
            checksum_str.encode()
        ).hexdigest()

        ###################################################
        # BUILD DMATRIX
        ###################################################

        dtrain = xgb.DMatrix(
            X,
            label=y,
            feature_names=self.feature_names,
        )

        dtrain.set_group(groups)

        ###################################################
        # RANKING PARAMETERS (PROPER OBJECTIVE)
        ###################################################

        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
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

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            verbose_eval=False,
        )

        ###################################################
        # SANITY CHECK
        ###################################################

        preds = self.model.predict(dtrain)

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite training predictions.")

        if np.std(preds) < MIN_SCORE_STD:
            raise RuntimeError("Score collapse during training.")

        logger.info(
            "XGBoost ranking trained | pred_std=%.6f",
            float(np.std(preds)),
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

        scores = self.model.predict(dmatrix)

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Non-finite inference scores.")

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
        }


###################################################
# BUILD MODEL
###################################################

def build_xgboost_pipeline():

    logger.info(
        "Building XGBoost Pairwise Ranking Model (production safe)"
    )

    return SafeXGBRanker()