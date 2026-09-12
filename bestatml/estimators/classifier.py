from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from bestatml.core.base import BaseBestatEstimator


class BestatClassifier(ClassifierMixin, BaseBestatEstimator):
    """Automated leakage-safe heterogeneous ensemble classifier.

    Parameters are inherited from :class:`BaseBestatEstimator`. The estimator
    evaluates multiple compatible model families with leakage-safe CV and OOF
    predictions, then constructs the configured or automatically selected ensemble.
    """

    task = "classification"

    def score(self, X: Any, y: Any) -> float:
        """Return mean classification accuracy, matching sklearn classifier conventions."""
        return float(accuracy_score(y, self.predict(X)))
