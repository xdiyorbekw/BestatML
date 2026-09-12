from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bestatml.preprocessing.detector import DataTypeDetector


class DateTimeExpander(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> "DateTimeExpander":
        del y
        self.columns_ = tuple(X.columns) if isinstance(X, pd.DataFrame) else tuple()
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DateTimeExpander expects a pandas DataFrame.")
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            dt = pd.to_datetime(X[col], errors="coerce")
            prefix = str(col)
            out[f"{prefix}__year"] = dt.dt.year
            out[f"{prefix}__month"] = dt.dt.month
            out[f"{prefix}__day"] = dt.dt.day
            out[f"{prefix}__weekday"] = dt.dt.weekday
            out[f"{prefix}__hour"] = dt.dt.hour
        return out

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns_:
            prefix = str(col)
            names.extend([f"{prefix}__year", f"{prefix}__month", f"{prefix}__day", f"{prefix}__weekday", f"{prefix}__hour"])
        return np.asarray(names, dtype=object)


class SchemaPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe schema detector + preprocessing graph.

    This transformer is fit inside each CV training fold because it lives inside every
    candidate model Pipeline. It deliberately never consumes y.
    """

    def __init__(
        self,
        *,
        scale_numeric: bool = False,
        impute_missing: bool = True,
        encode_categorical: bool = True,
        remove_low_variance: bool = False,
        variance_threshold: float = 0.0,
    ) -> None:
        self.scale_numeric = scale_numeric
        self.impute_missing = impute_missing
        self.encode_categorical = encode_categorical
        self.remove_low_variance = remove_low_variance
        self.variance_threshold = variance_threshold

    def fit(self, X: Any, y: Any = None) -> "SchemaPreprocessor":
        del y
        if not isinstance(X, pd.DataFrame):
            self.input_is_dataframe_ = False
            self.feature_names_in_ = np.asarray([f"feature_{i}" for i in range(np.asarray(X).shape[1])], dtype=object)
            self.detector_ = None
            self.column_transformer_ = Pipeline([
                ("imputer", SimpleImputer(strategy="median" if self.impute_missing else "mean")),
                ("scaler", StandardScaler() if self.scale_numeric else "passthrough"),
            ])
            self.column_transformer_.fit(X)
            self.variance_filter_ = VarianceThreshold(self.variance_threshold) if self.remove_low_variance else "passthrough"
            if self.remove_low_variance:
                self.variance_filter_.fit(self.column_transformer_.transform(X))
            return self

        self.input_is_dataframe_ = True
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.detector_ = DataTypeDetector().fit(X)
        groups = self.detector_.get_feature_groups()
        self.numeric_features_ = groups.numeric
        self.categorical_features_ = groups.categorical
        self.datetime_features_ = groups.datetime

        numeric_steps = []
        if self.impute_missing:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
        numeric_pipe: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"

        cat_steps = []
        if self.impute_missing:
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        if self.encode_categorical:
            cat_steps.append((
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ))
        cat_pipe: Any = Pipeline(cat_steps) if cat_steps else "passthrough"

        dt_pipe = Pipeline([
            ("expand", DateTimeExpander()),
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers = []
        if groups.numeric:
            transformers.append(("numeric", numeric_pipe, list(groups.numeric)))
        if groups.categorical:
            transformers.append(("categorical", cat_pipe, list(groups.categorical)))
        if groups.datetime:
            transformers.append(("datetime", dt_pipe, list(groups.datetime)))

        if not transformers:
            raise ValueError("No usable features were detected in X.")
        self.column_transformer_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
        transformed = self.column_transformer_.fit_transform(X)
        self.variance_filter_ = VarianceThreshold(self.variance_threshold) if self.remove_low_variance else "passthrough"
        if self.remove_low_variance:
            self.variance_filter_.fit(transformed)
        full_names = self._column_transformer_feature_names()
        self.processed_feature_names_ = full_names
        if self.remove_low_variance:
            support = self.variance_filter_.get_support()
            self.removed_features_ = tuple(full_names[~support].tolist())
            self.processed_feature_names_ = full_names[support]
        else:
            self.removed_features_ = tuple()
        return self

    def transform(self, X: Any) -> Any:
        transformed = self.column_transformer_.transform(X)
        if self.remove_low_variance:
            transformed = self.variance_filter_.transform(transformed)
        return transformed

    def _column_transformer_feature_names(self) -> np.ndarray:
        try:
            return np.asarray(self.column_transformer_.get_feature_names_out(), dtype=object)
        except Exception:
            if not self.input_is_dataframe_:
                return np.asarray(self.feature_names_in_, dtype=object)
            return np.asarray([str(x) for x in self.feature_names_in_], dtype=object)

    def get_feature_names_out(self, input_features=None):
        del input_features
        return np.asarray(self.processed_feature_names_, dtype=object)

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return tuple(getattr(self, "numeric_features_", tuple()))

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(getattr(self, "categorical_features_", tuple()))
