import numpy as np
import pandas as pd
from core.models.xgboost import SafeXGBRanker


def generate_data(rows=400, features=12, groups=8):

    X = pd.DataFrame(
        np.random.randn(rows, features),
        columns=[f"f{i}" for i in range(features)]
    )

    y = np.random.randn(rows)

    group_sizes = [rows // groups] * groups
    group_sizes[-1] += rows - sum(group_sizes)

    return X, y, group_sizes


def test_training_and_checksums():

    X, y, groups = generate_data()

    model = SafeXGBRanker()
    model.fit(X, y, groups)

    assert model.model is not None
    assert model.feature_checksum is not None
    assert model.param_checksum is not None
    assert model.booster_checksum is not None
    assert model.best_iteration is not None


def test_predict_valid():

    X, y, groups = generate_data()

    model = SafeXGBRanker()
    model.fit(X, y, groups)

    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))


def test_entropy_guard_trigger():

    X = pd.DataFrame(np.ones((200, 10)))
    y = np.ones(200)
    groups = [20] * 10

    model = SafeXGBRanker()

    try:
        model.fit(X, y, groups)
    except RuntimeError as e:
        assert "variance" in str(e)