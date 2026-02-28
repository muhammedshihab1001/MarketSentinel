import xgboost as xgb
import numpy as np
import logging
import hashlib
import json
import random

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
NUM_BOOST_ROUNDS = 800
EARLY_STOPPING_ROUNDS = 50

MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
MIN_GROUP_SIZE = 5
EPSILON = 1e-12


###################################################
# PRODUCTION RANKING MODEL (INSTITUTIONAL GRADE)
###################################################

class SafeXGBRanker:

    def __init__(self):

        self.model = None
        self.feature_names = None
        self.training_cols = None

        self.feature_checksum = None
        self.param_checksum = None
        self.importance_checksum = None
        self.training_fingerprint = None

        random.seed(SEED)
        np.random.seed(SEED)

    ###################################################
    # INTERNAL: CREATE VALIDATION SPLIT
    ###################################################

    def _split_train_validation(self, X, y, groups):

        # last 20% of groups used for validation
        group_boundaries = np.cumsum(groups)
        total_groups = len(groups)
        val_groups = max(1, int(total_groups * 0.2))

        split_idx = group_boundaries[-val_groups]

        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]

        X_val = X.iloc[split_idx:]
        y_val = y[split_idx:]

        train_groups = groups[:-val_groups]
        val_groups = groups[-val_groups:]

        return X_train, y_train, train_groups, X_val, y_val, val_groups

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
            raise RuntimeError("Group sizes mismatch.")

        self.feature_names = list(X.columns)
        self.training_cols = X.shape[1]

        # fingerprint dataset for reproducibility
        self.training_fingerprint = hashlib.sha256(
            X.to_numpy().tobytes()
        ).hexdigest()

        ###################################################
        # LOCK FEATURE CONTRACT
        ###################################################

        checksum_str = json.dumps(self.feature_names, sort_keys=False)
        self.feature_checksum = hashlib.sha256(
            checksum_str.encode()
        ).hexdigest()

        ###################################################
        # TRAIN/VALIDATION SPLIT
        ###################################################

        X_train, y_train, train_groups, X_val, y_val, val_groups = \
            self._split_train_validation(X, y, groups)

        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self.feature_names,
        )
        dtrain.set_group(train_groups)

        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=self.feature_names,
        )
        dval.set_group(val_groups)

        ###################################################
        # PARAMETERS
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
        # TRAIN WITH EARLY STOPPING
        ###################################################

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        ###################################################
        # SANITY CHECK
        ###################################################

        preds = self.model.predict(dtrain)

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite training predictions.")

        if np.std(preds) < MIN_SCORE_STD:
            raise RuntimeError("Score collapse detected.")

        logger.info(
            "XGB Ranking trained | rounds=%d | pred_std=%.6f",
            self.model.best_iteration,
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

        # prevent extreme explosions
        scores = np.clip(scores, -50, 50)

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
        "Building Institutional XGBoost Pairwise Ranking Model"
    )

    return SafeXGBRanker()