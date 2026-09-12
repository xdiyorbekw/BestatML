from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, get_scorer


@dataclass
class EnsembleResult:
    method: str
    weights: np.ndarray | None
    meta_learner: Any | None


def _proxy_tags(base: BaseEstimator, *, classification: bool):
    tags = BaseEstimator.__sklearn_tags__(base)
    tags.estimator_type = "classifier" if classification else "regressor"
    return tags


class PredictionProxy(BaseEstimator):
    def __init__(self, prediction: np.ndarray, *, classification: bool, classes: np.ndarray | None = None):
        self._prediction = prediction
        self._classification = classification
        self.classes_ = classes

    def predict(self, X: Any) -> np.ndarray:
        del X
        if self._classification and self._prediction.ndim == 2:
            return self.classes_[np.argmax(self._prediction, axis=1)]  # type: ignore[index]
        return np.asarray(self._prediction)

    def predict_proba(self, X: Any) -> np.ndarray:
        del X
        if not self._classification:
            raise AttributeError
        return np.asarray(self._prediction)

    def __sklearn_tags__(self):
        return _proxy_tags(self, classification=self._classification)

    def decision_function(self, X: Any) -> np.ndarray:
        del X
        if self._classification:
            if self._prediction.ndim == 2 and self._prediction.shape[1] == 2:
                return self._prediction[:, 1]
            return np.max(self._prediction, axis=1)
        return np.asarray(self._prediction)


def score_predictions(y_true: np.ndarray, prediction: np.ndarray, *, classification: bool, classes: np.ndarray | None, scoring: str) -> float:
    proxy = PredictionProxy(prediction, classification=classification, classes=classes)
    scorer = get_scorer(scoring)
    X_dummy = np.zeros((len(y_true), 1), dtype=float)
    return float(scorer(proxy, X_dummy, y_true))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def optimize_weights(
    oof_predictions: list[np.ndarray],
    y: np.ndarray,
    *,
    classification: bool,
    classes: np.ndarray | None,
    scoring: str,
) -> np.ndarray:
    """Optimize non-negative normalized blend weights against strictly OOF predictions."""
    try:
        from scipy.optimize import minimize
    except Exception:
        # Fallback is deterministic and still OOF based.
        scores = [score_predictions(y, p, classification=classification, classes=classes, scoring=scoring) for p in oof_predictions]
        shifted = np.asarray(scores) - np.min(scores)
        return normalize_weights(shifted + 1e-6)

    stacked = np.stack(oof_predictions, axis=1)
    n_models = len(oof_predictions)

    def blend(w: np.ndarray) -> np.ndarray:
        if classification:
            if stacked.ndim != 3:
                raise ValueError("Classification probability tensor must be 3D.")
            return np.tensordot(w, stacked, axes=(0, 1))
        return np.tensordot(w, stacked, axes=(0, 1))

    def objective(w: np.ndarray) -> float:
        pred = blend(w)
        try:
            return -score_predictions(y, pred, classification=classification, classes=classes, scoring=scoring)
        except Exception:
            return 1e9

    x0 = np.full(n_models, 1.0 / n_models)
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    result = minimize(objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n_models, constraints=constraints)
    if not result.success:
        return x0
    return normalize_weights(result.x)


def make_blend(oof_predictions: list[np.ndarray], weights: np.ndarray, *, classification: bool) -> np.ndarray:
    stacked = np.stack(oof_predictions, axis=1)
    if classification:
        return np.tensordot(weights, stacked, axes=(0, 1))
    return np.tensordot(weights, stacked, axes=(0, 1))


def select_diverse_models(names: list[str], scores: dict[str, float], oof: dict[str, np.ndarray], max_models: int, *, classification: bool) -> list[str]:
    if not names:
        return []
    ordered = sorted(names, key=lambda n: scores[n], reverse=True)
    selected = [ordered[0]]
    while len(selected) < min(max_models, len(names)):
        best = None
        best_value = -np.inf
        for name in ordered:
            if name in selected:
                continue
            pred = np.asarray(oof[name])
            flat = pred[:, 1] if classification and pred.ndim == 2 and pred.shape[1] == 2 else pred.reshape(len(pred), -1)
            flat = flat.ravel() if flat.ndim > 1 else flat
            diversities = []
            for other in selected:
                op = np.asarray(oof[other])
                oflat = op[:, 1] if classification and op.ndim == 2 and op.shape[1] == 2 else op.reshape(len(op), -1)
                oflat = oflat.ravel() if oflat.ndim > 1 else oflat
                if np.std(flat) == 0 or np.std(oflat) == 0:
                    corr = 1.0
                else:
                    corr = float(np.corrcoef(flat, oflat)[0, 1])
                    if not np.isfinite(corr):
                        corr = 1.0
                diversities.append(1.0 - abs(corr))
            score_component = scores[name]
            diversity_component = float(np.mean(diversities)) if diversities else 0.0
            value = 0.75 * score_component + 0.25 * diversity_component
            if value > best_value:
                best_value, best = value, name
        if best is None:
            break
        selected.append(best)
    return selected
