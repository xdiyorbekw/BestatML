from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from bestatml.core.base import BaseBestatEstimator


class BestatRegressor(RegressorMixin, BaseBestatEstimator):
    """Automated leakage-safe heterogeneous ensemble regressor.

    Parameters are inherited from :class:`BaseBestatEstimator`.
    """

    task = "regression"

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination (R²)."""
        return float(r2_score(y, self.predict(X)))
