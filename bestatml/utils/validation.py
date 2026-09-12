from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y, check_array

from bestatml.exceptions import BestatMLDataError


def _header(message: str, kind: str = "DataError") -> str:
    return f"BestatML[{kind}]: {message}"


def _validate_target_shape(y: np.ndarray) -> np.ndarray:
    if y.ndim > 2 or (y.ndim == 2 and 1 not in y.shape):
        raise BestatMLDataError(
            _header("Target y must be one-dimensional.")
            + f"\n\nReceived shape:\n    {y.shape}\n\n"
            "How to fix:\n    Pass one target value per input row."
        )
    return np.ravel(y)


def _validate_missing_target(y: np.ndarray) -> None:
    try:
        count = int(np.asarray(pd.isna(y)).sum())
    except (TypeError, ValueError):
        count = 0
    if count:
        raise BestatMLDataError(
            _header("Target y contains missing values.")
            + f"\n\nFound:\n    {count} missing target values.\n\n"
            "How to fix:\n    Remove or resolve missing targets before calling fit()."
        )


def _validate_dataframe_cells(X: pd.DataFrame) -> None:
    for column in X.columns:
        if X[column].dtype != object:
            continue
        bad = X[column].map(lambda value: isinstance(value, (dict, list, tuple, set, np.ndarray)))
        if bool(bad.any()):
            examples = [repr(v) for v in X.loc[bad, column].head(3).tolist()]
            raise BestatMLDataError(
                _header(f"Column {column!r} contains nested object values.")
                + f"\n\nProblem:\n    Examples: {examples}\n\n"
                "Expected:\n    Scalar numeric, boolean, categorical, string, or datetime-like values.\n\n"
                "How to fix:\n    Flatten nested values into real features or remove the affected column."
            )


def validate_training_data(X: Any, y: Any) -> tuple[Any, np.ndarray, tuple[str, ...] | None]:
    if X is None or y is None:
        raise BestatMLDataError(_header("Both X and y are required for fit()."))

    feature_names: tuple[str, ...] | None = None
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if isinstance(X, pd.DataFrame):
        feature_names = tuple(str(c) for c in X.columns)
        if any(str(c) != c for c in X.columns):
            X = X.copy()
            X.columns = feature_names
        if X.columns.duplicated().any():
            duplicates = tuple(map(str, X.columns[X.columns.duplicated()].unique()))
            raise BestatMLDataError(
                _header("X contains duplicate column names.")
                + f"\n\nDuplicates:\n    {duplicates}\n\n"
                "How to fix:\n    Rename duplicate columns before fitting."
            )
        if X.shape[1] == 0:
            raise BestatMLDataError(
                _header("X contains no feature columns.")
                + "\n\nHow to fix:\n    Provide at least one feature column."
            )
        if X.shape[0] == 0:
            raise BestatMLDataError(
                _header("X contains no samples.")
                + "\n\nHow to fix:\n    Provide at least one training row."
            )
        if not hasattr(y, "__len__") or len(y) != len(X):
            received = len(y) if hasattr(y, "__len__") else "unknown"
            raise BestatMLDataError(
                _header("X and y contain different numbers of samples.")
                + f"\n\nX samples:\n    {len(X)}\ny samples:\n    {received}\n\n"
                "Expected:\n    One target value for every input row.\n\n"
                "How to fix:\n    Align X and y before calling fit()."
            )
        _validate_dataframe_cells(X)
        y_arr = _validate_target_shape(np.asarray(y))
        _validate_missing_target(y_arr)
        return X, y_arr, feature_names

    if sparse.issparse(X):
        if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] == 0:
            raise BestatMLDataError(_header("X must be a non-empty 2D sparse matrix."))
        if not hasattr(y, "__len__") or len(y) != X.shape[0]:
            received = len(y) if hasattr(y, "__len__") else "unknown"
            raise BestatMLDataError(
                _header("X and y contain different numbers of samples.")
                + f"\n\nX samples:\n    {X.shape[0]}\ny samples:\n    {received}"
            )
        y_arr = _validate_target_shape(np.asarray(y))
        _validate_missing_target(y_arr)
        try:
            X_checked, y_checked = check_X_y(
                X, y_arr, accept_sparse=True, ensure_all_finite="allow-nan"
            )
        except (TypeError, ValueError) as exc:
            raise BestatMLDataError(
                _header("X/y failed sklearn input validation.")
                + f"\n\nProblem:\n    {exc}\n\nHow to fix:\n    Inspect feature values and target shape/types."
            ) from exc
        return X_checked, y_checked, None

    try:
        arr = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise BestatMLDataError(
            _header("X could not be converted to a supported array-like structure.")
            + f"\n\nProblem:\n    {exc}\n\n"
            "Supported inputs:\n    pandas.DataFrame, pandas.Series, NumPy arrays, and supported scipy sparse matrices."
        ) from exc
    if arr.ndim != 2 or arr.shape[1] == 0 or arr.shape[0] == 0:
        raise BestatMLDataError(
            _header("X must be a non-empty 2D array-like object.")
            + f"\n\nReceived shape:\n    {arr.shape}\n\n"
            "How to fix:\n    Pass data with shape (n_samples, n_features)."
        )
    if not hasattr(y, "__len__") or len(y) != arr.shape[0]:
        received = len(y) if hasattr(y, "__len__") else "unknown"
        raise BestatMLDataError(
            _header("X and y contain different numbers of samples.")
            + f"\n\nX samples:\n    {arr.shape[0]}\ny samples:\n    {received}"
        )
    y_arr = _validate_target_shape(np.asarray(y))
    _validate_missing_target(y_arr)
    try:
        X_checked, y_checked = check_X_y(
            X, y_arr, accept_sparse=True, ensure_all_finite="allow-nan"
        )
    except (TypeError, ValueError) as exc:
        raise BestatMLDataError(
            _header("X/y failed sklearn input validation.")
            + f"\n\nProblem:\n    {exc}\n\nHow to fix:\n    Check feature values, target values, and array dimensions."
        ) from exc
    return X_checked, y_checked, None


def validate_prediction_data(
    X: Any,
    feature_names: Sequence[str] | None,
    n_features_in: int,
) -> Any:
    if X is None:
        raise BestatMLDataError(_header("Prediction data X cannot be None."))
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if feature_names is not None:
        if not isinstance(X, pd.DataFrame):
            raise BestatMLDataError(
                _header("Prediction data must be a pandas DataFrame.")
                + "\n\nReason:\n    The estimator was fitted with a named DataFrame schema.\n\n"
                "How to fix:\n    Pass a DataFrame containing the trained feature names."
            )
        expected = tuple(map(str, feature_names))
        incoming = tuple(map(str, X.columns))
        if len(set(incoming)) != len(incoming):
            raise BestatMLDataError(_header("Prediction data contains duplicate column names."))
        missing = [c for c in expected if c not in incoming]
        extra = [c for c in incoming if c not in expected]
        if missing:
            raise BestatMLDataError(
                _header("Prediction data is missing required feature(s).")
                + f"\n\nMissing:\n    {missing}\n\nExpected feature count:\n    {len(expected)}\n"
                f"Received feature count:\n    {len(incoming)}\n\nHow to fix:\n    Provide every feature used during fit()."
            )
        if extra:
            raise BestatMLDataError(
                _header("Prediction data contains unexpected feature(s).")
                + f"\n\nUnexpected:\n    {extra}\n\nHow to fix:\n    Drop columns not present during fit()."
            )
        _validate_dataframe_cells(X)
        if tuple(incoming) != expected:
            X = X.copy()
            X.columns = incoming
        return X.loc[:, list(expected)]

    try:
        checked = check_array(X, accept_sparse=True, ensure_all_finite="allow-nan")
    except (TypeError, ValueError) as exc:
        raise BestatMLDataError(
            _header("Prediction data failed sklearn validation.")
            + f"\n\nProblem:\n    {exc}\n\nHow to fix:\n    Ensure X is a valid 2D array with trained feature types."
        ) from exc
    if checked.shape[1] != n_features_in:
        raise BestatMLDataError(
            _header("Prediction data has the wrong number of features.")
            + f"\n\nExpected features:\n    {n_features_in}\nReceived:\n    {checked.shape[1]}\n\n"
            "How to fix:\n    Pass the same feature set used during fit()."
        )
    return checked


def validate_task_target(y: np.ndarray, *, task: str) -> None:
    target_type = type_of_target(y)
    if task == "classification":
        try:
            unique = np.unique(y)
        except TypeError as exc:
            raise BestatMLDataError(
                _header("Classification target labels are not mutually comparable.")
                + "\n\nHow to fix:\n    Use a single consistent label type (for example, all strings or all integers)."
            ) from exc
        if len(unique) < 2:
            raise BestatMLDataError(
                _header("Classification requires at least 2 target classes.")
                + f"\n\nDetected unique classes:\n    {len(unique)}\nValues:\n    {unique.tolist()}\n\n"
                "How to fix:\n    Provide training data containing at least two classes."
            )
        if target_type not in {"binary", "multiclass"}:
            raise BestatMLDataError(
                _header("Target y is not a supported classification target.")
                + f"\n\nDetected target type:\n    {target_type}\n\nHow to fix:\n    Provide one categorical class label per row."
            )
    else:
        if target_type not in {"continuous", "continuous-multioutput", "binary", "multiclass"}:
            raise BestatMLDataError(
                _header("BestatRegressor received a non-numeric or unsupported target.")
                + f"\n\nDetected target type:\n    {target_type}\n\nHow to fix:\n    Use BestatClassifier for categorical targets or provide a numeric regression target."
            )
        if not np.issubdtype(y.dtype, np.number):
            raise BestatMLDataError(
                _header("BestatRegressor requires a numeric target.")
                + f"\n\nDetected dtype:\n    {y.dtype}\n\nHow to fix:\n    Convert the regression target to numeric values or use BestatClassifier."
            )


def validate_groups(groups: Any, n_samples: int) -> None:
    if groups is None:
        return
    try:
        length = len(groups)
    except TypeError as exc:
        raise BestatMLDataError(
            _header("groups must be array-like with one value per sample.")
        ) from exc
    if length != n_samples:
        raise BestatMLDataError(
            _header("groups and X contain different numbers of samples.")
            + f"\n\nX samples:\n    {n_samples}\ngroups values:\n    {length}\n\nHow to fix:\n    Provide exactly one group label per row."
        )
