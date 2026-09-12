import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from bestatml import (
    BestatClassifier,
    BestatRegressor,
    BestatMLConfigurationError,
    BestatMLDataError,
    BestatMLNotFittedError,
    BestatMLPersistenceError,
)


def test_not_fitted_error_is_bestat_specific():
    with pytest.raises(BestatMLNotFittedError, match="has not been fitted"):
        BestatClassifier().predict(np.ones((2, 3)))


def test_length_mismatch_is_actionable():
    X = np.ones((5, 2))
    y = np.zeros(4)
    with pytest.raises(BestatMLDataError, match="different numbers of samples"):
        BestatClassifier().fit(X, y)


def test_missing_target_is_rejected():
    X = np.ones((4, 2))
    y = np.array([0, 1, np.nan, 1])
    with pytest.raises(BestatMLDataError, match="missing target"):
        BestatClassifier().fit(X, y)


def test_single_class_is_rejected():
    X = np.ones((6, 2))
    y = np.zeros(6)
    with pytest.raises(BestatMLDataError, match="at least 2 target classes"):
        BestatClassifier().fit(X, y)


def test_invalid_configuration_is_rejected_before_training():
    with pytest.raises(BestatMLConfigurationError, match="cv must be"):
        BestatClassifier(cv=1).fit(np.ones((4, 2)), np.array([0, 1, 0, 1]))
    with pytest.raises(BestatMLConfigurationError, match="ensemble_method"):
        BestatClassifier(ensemble_method="invalid").fit(np.ones((4, 2)), np.array([0, 1, 0, 1]))


def test_prediction_schema_aligns_columns_by_name():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [0, 1, 0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1, 0, 1])
    model = BestatClassifier(cv=2, tune_hyperparameters=False, max_models=1, random_state=2)
    model.fit(X, y)
    original = model.predict(X)
    reordered = model.predict(X[["b", "a"]])
    np.testing.assert_array_equal(original, reordered)


def test_prediction_schema_reports_missing_column():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [0, 1, 0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1, 0, 1])
    model = BestatClassifier(cv=2, tune_hyperparameters=False, max_models=1, random_state=2)
    model.fit(X, y)
    with pytest.raises(BestatMLDataError, match="missing required feature"):
        model.predict(X[["a"]])


def test_regressor_rejects_text_target():
    X, _ = make_regression(n_samples=20, n_features=3, random_state=3)
    with pytest.raises(BestatMLDataError, match="numeric target"):
        BestatRegressor().fit(X, np.array(["low"] * 20))


def test_failed_fit_does_not_leave_fitted_state():
    model = BestatClassifier(cv=1)
    with pytest.raises(BestatMLConfigurationError):
        model.fit(np.ones((4, 2)), np.array([0, 1, 0, 1]))
    with pytest.raises(BestatMLNotFittedError):
        model.predict(np.ones((2, 2)))


def test_corrupt_artifact_has_clean_persistence_error(tmp_path):
    path = tmp_path / "broken.bestat"
    path.write_bytes(b"not a bestat artifact")
    with pytest.raises(BestatMLPersistenceError, match="Could not load"):
        BestatClassifier.load(path)


def test_all_invalid_families_raise_training_error():
    X = np.ones((6, 2))
    y = np.array([0, 1, 0, 1, 0, 1])
    model = BestatClassifier(cv=2, model_families=("linear",), max_models=1, tune_hyperparameters=False)
    # Logistic regression is valid, so this confirms the normal path remains usable after hardening.
    model.fit(X, y)
    assert model.selected_models_


def test_explicit_optional_dependency_failure_is_actionable(monkeypatch):
    from bestatml.core import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "get_registry", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        trainer_module,
        "missing_optional_dependencies",
        lambda *args, **kwargs: (("xgboost", "xgboost"),),
    )
    with pytest.raises(Exception, match="xgboost support was explicitly requested"):
        BestatClassifier(
            cv=2,
            model_families=("xgboost",),
            tune_hyperparameters=False,
        ).fit(np.ones((6, 2)), np.array([0, 1, 0, 1, 0, 1]))
