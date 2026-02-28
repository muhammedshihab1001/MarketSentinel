import xgboost as xgb
import numpy as np
import logging
import hashlib
import json

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_SCORE_STD = 1e-6
MIN_TARGET_STD = 1e-6
NUM_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 150
MIN_GROUP_SIZE = 5
EPSILON = 1e-12


###################################################
# SAFE RANKING MODEL (GROUP-SAFE + EARLY STOP)
###################################################

class SafeXGBRanker:

    def __init__(self):
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

    def fit(self, X, y, groups):

        if X is None or X.empty:
            raise RuntimeError("Training feature matrix empty.")

        if len(X) != len(y):
            raise RuntimeError("Feature/label length mismatch.")

        if X.isnull().any().any():
            raise RuntimeError("NaN detected in training features.")

        X = X.copy().astype(np.float32)
        y = np.asarray(y, dtype=np.float32)

        if np.std(y) < MIN_TARGET_STD:
            raise RuntimeError("Target variance too small.")

        if groups is None or len(groups) == 0:
            raise RuntimeError("Ranking requires group definition.")

        if min(groups) < MIN_GROUP_SIZE:
            raise RuntimeError("Insufficient cross-sectional group size.")

        if sum(groups) != len(X):
            raise RuntimeError("Group sizes do not match training rows.")

        self.training_rows = X.shape[0]
        self.training_cols = X.shape[1]

        logger.info(
            "XGBoost ranking | rows=%s cols=%s | target_std=%.8f",
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
        # GROUP-SAFE TRAIN / VALIDATION SPLIT
        ###################################################

        cumulative = np.cumsum(groups)
        total_groups = len(groups)

        split_group = int(total_groups * 0.8)

        # ensure at least 5 validation groups
        if total_groups - split_group < 5:
            split_group = total_groups - 5

        split_row = cumulative[split_group - 1]

        X_train = X.iloc[:split_row]
        y_train = y[:split_row]
        groups_train = groups[:split_group]

        X_valid = X.iloc[split_row:]
        y_valid = y[split_row:]
        groups_valid = groups[split_group:]

        ###################################################
        # DMATRICES
        ###################################################

        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self.feature_names,
        )
        dtrain.set_group(groups_train)

        dvalid = xgb.DMatrix(
            X_valid,
            label=y_valid,
            feature_names=self.feature_names,
        )
        dvalid.set_group(groups_valid)

        ###################################################
        # RANKING PARAMETERS (STABLE + CONSERVATIVE)
        ###################################################

        params = {
            "max_depth": 5,
            "eta": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.2,
            "reg_alpha": 1.0,
            "reg_lambda": 3.0,
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
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
        # TRAIN WITH EARLY STOPPING
        ###################################################

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        ###################################################
        # TRAIN SANITY CHECK
        ###################################################

        preds = self.model.predict(dtrain)

        if not np.all(np.isfinite(preds)):
            raise RuntimeError("Non-finite predictions detected.")

        if np.std(preds) < MIN_SCORE_STD:
            raise RuntimeError("Score collapse detected during training.")

        logger.info(
            "XGBoost ranking training completed | pred_std=%.8f | best_iter=%s",
            float(np.std(preds)),
            self.model.best_iteration,
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

        return scores

    ###################################################
    # BACKWARD COMPAT
    ###################################################

    def predict_proba(self, X):
        scores = self.predict(X)
        scaled = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
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

    model = SafeXGBRanker()

    logger.info(
        "XGBoost RANKING alpha model built (group-safe validation, deterministic CPU mode)."
    )

    return model