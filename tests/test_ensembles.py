import numpy as np
import pytest

from bestatml import BestatClassifier, BestatRegressor


@pytest.mark.parametrize("method", ["best_model", "weighted_blending", "voting", "stacking", "auto"])
def test_classifier_ensemble_methods(small_classification_data, method):
    X, y = small_classification_data
    model = BestatClassifier(cv=3, tune_hyperparameters=False, max_models=3, ensemble_method=method, random_state=42)
    model.fit(X, y)
    assert model.ensemble_method_ in {"best_model", "weighted_blending", "voting", "stacking"}
    pred = model.predict(X)
    assert pred.shape[0] == len(X)


def test_regressor_weighted_blending(small_regression_data):
    X, y = small_regression_data
    model = BestatRegressor(cv=3, tune_hyperparameters=False, max_models=3, ensemble_method="weighted_blending", random_state=42)
    model.fit(X, y)
    assert model.ensemble_method_ in {"weighted_blending", "best_model"}
    assert np.isfinite(model.predict(X)).all()
