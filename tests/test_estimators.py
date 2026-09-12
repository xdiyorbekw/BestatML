import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from bestatml import BestatClassifier, BestatRegressor


def test_classifier_fit_predict_and_clone(small_classification_data):
    X, y = small_classification_data
    model = BestatClassifier(cv=3, tune_hyperparameters=False, max_models=3, random_state=42)
    cloned = clone(model)
    assert cloned.get_params()["cv"] == 3
    model.fit(X, y)
    assert len(model.selected_models_) >= 1
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape == (len(X),)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
    check_is_fitted(model)


def test_regressor_fit_predict(small_regression_data):
    X, y = small_regression_data
    model = BestatRegressor(cv=3, tune_hyperparameters=False, max_models=3, random_state=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_mixed_dataframe_and_unseen_category():
    X = pd.DataFrame({
        "age": [20, 31, None, 44, 51, 28, 35, 41],
        "city": ["a", "b", "a", "b", "a", "b", "a", "b"],
    })
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    model = BestatClassifier(cv=2, tune_hyperparameters=False, max_models=2, random_state=42)
    model.fit(X, y)
    Xt = pd.DataFrame({"age": [27, None], "city": ["never-seen", "a"]})
    pred = model.predict(Xt)
    assert pred.shape == (2,)


def test_sklearn_tags_and_default_style_numeric_classification():
    from sklearn.base import is_classifier

    X = pd.DataFrame({
        "Pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "Sex": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        "Age": [22, 30, 25, None, 35, 20, 40, 29, 18, 36, 27, 31],
        "Fare": [71.2, 12.5, 7.8, 53.1, 10.2, 8.0, 26.5, 13.4, 7.9, 20.1, 11.0, 51.4],
        "Embarked_Q": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "Embarked_S": [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    })
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    model = BestatClassifier(cv=3, tune_hyperparameters=False, max_models=3, random_state=42)
    assert is_classifier(model)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)
