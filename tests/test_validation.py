import numpy as np
import pandas as pd
import pytest

from bestatml import BestatClassifier
from bestatml.exceptions import DataValidationError, BestatMLValidationError


def test_dataframe_schema_is_preserved():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["x", "y", "x", "y", "x", "y"]})
    y = [0, 1, 0, 1, 0, 1]
    model = BestatClassifier(cv=2, tune_hyperparameters=False, max_models=1, random_state=1)
    model.fit(X, y)
    original = model.predict(X)
    reordered = model.predict(X[["b", "a"]])
    np.testing.assert_array_equal(original, reordered)


def test_tiny_class_count_has_actionable_error():
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 0, 0, 1])
    model = BestatClassifier(cv=3, tune_hyperparameters=False, max_models=1, random_state=1)
    with pytest.raises(BestatMLValidationError, match="Smallest class contains"):
        model.fit(X, y)
