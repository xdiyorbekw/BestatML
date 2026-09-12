from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class FeatureGroups:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    datetime: tuple[str, ...]


class DataTypeDetector(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> "DataTypeDetector":
        del y
        if not isinstance(X, pd.DataFrame):
            self.feature_names_in_ = None
            self.groups_ = FeatureGroups(tuple(), tuple(), tuple())
            return self
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        numeric, categorical, datetime = [], [], []
        for col in X.columns:
            series = X[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime.append(str(col))
            elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric.append(str(col))
            else:
                categorical.append(str(col))
        self.groups_ = FeatureGroups(tuple(numeric), tuple(categorical), tuple(datetime))
        return self

    def transform(self, X: Any) -> Any:
        return X

    def get_feature_groups(self) -> FeatureGroups:
        return self.groups_
