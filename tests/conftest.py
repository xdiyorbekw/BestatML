import pytest


@pytest.fixture
def small_classification_data():
    from sklearn.datasets import make_classification
    return make_classification(n_samples=80, n_features=8, n_informative=5, n_redundant=1, random_state=7)


@pytest.fixture
def small_regression_data():
    from sklearn.datasets import make_regression
    return make_regression(n_samples=80, n_features=8, n_informative=5, noise=4.0, random_state=7)
