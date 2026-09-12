from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class CompatibleClassifier(ClassifierMixin, BaseEstimator):
    """Adapter that gives third-party classifiers modern sklearn estimator tags."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: Any, y: Any, **fit_params: Any):
        self.estimator.fit(X, y, **fit_params)
        self.classes_ = np.asarray(getattr(self.estimator, "classes_", np.unique(y)))
        return self

    def predict(self, X: Any):
        return self.estimator.predict(X)

    def predict_proba(self, X: Any):
        return self.estimator.predict_proba(X)

    def decision_function(self, X: Any):
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X)
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else np.max(proba, axis=1)

    def get_params(self, deep: bool = True):
        return {"estimator": self.estimator}

    def set_params(self, **params: Any):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if params:
            self.estimator.set_params(**params)
        return self


class CompatibleRegressor(RegressorMixin, BaseEstimator):
    """Adapter that gives third-party regressors modern sklearn estimator tags."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: Any, y: Any, **fit_params: Any):
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any):
        return self.estimator.predict(X)

    def get_params(self, deep: bool = True):
        return {"estimator": self.estimator}

    def set_params(self, **params: Any):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if params:
            self.estimator.set_params(**params)
        return self
