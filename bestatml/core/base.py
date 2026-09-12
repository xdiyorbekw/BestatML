from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from bestatml.core.configuration import resolve_budget
from bestatml.core.trainer import EnsembleTrainer
from bestatml.exceptions import (
    BestatMLConfigurationError,
    BestatMLDataError,
    BestatMLEnsembleError,
    BestatMLNotFittedError,
    BestatMLTrainingError,
    BestatMLValidationError,
)
from bestatml.explainability.importance import permutation_feature_importance
from bestatml.metrics.classification import default_classification_scoring
from bestatml.metrics.regression import default_regression_scoring
from bestatml.persistence.serializer import load_model, save_model
from bestatml.utils.validation import validate_groups, validate_prediction_data, validate_task_target, validate_training_data


class BaseBestatEstimator(BaseEstimator, ABC):
    task: str
    _default_model_families = ("linear", "tree", "random_forest", "boosting", "catboost", "xgboost", "lightgbm")

    def __init__(
        self,
        random_state: int | None = 42,
        n_jobs: int = 1,
        verbose: int = 0,
        progress: bool = False,
        cv: int | Any = 5,
        scoring: str | None = None,
        max_models: int = 4,
        ensemble_method: str = "auto",
        optimize: bool = True,
        tune_hyperparameters: bool = True,
        search_budget: str = "balanced",
        auto_preprocess: bool = True,
        impute_missing: bool = True,
        encode_categorical: bool = True,
        scale_features: bool = True,
        remove_low_variance: bool = False,
        variance_threshold: float = 0.0,
        categorical_strategy: str = "onehot",
        use_stacking: bool = True,
        use_weighted_blending: bool = True,
        use_voting: bool = True,
        meta_learner: Any = None,
        blend_weight_strategy: str = "oof_optimize",
        model_families: tuple[str, ...] = ("linear", "tree", "random_forest", "boosting", "catboost", "xgboost", "lightgbm"),
        preprocessing: Any = None,
    ) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.progress = progress
        self.cv = cv
        self.scoring = scoring
        self.max_models = max_models
        self.ensemble_method = ensemble_method
        self.optimize = optimize
        self.tune_hyperparameters = tune_hyperparameters
        self.search_budget = search_budget
        self.auto_preprocess = auto_preprocess
        self.impute_missing = impute_missing
        self.encode_categorical = encode_categorical
        self.scale_features = scale_features
        self.remove_low_variance = remove_low_variance
        self.variance_threshold = variance_threshold
        self.categorical_strategy = categorical_strategy
        self.use_stacking = use_stacking
        self.use_weighted_blending = use_weighted_blending
        self.use_voting = use_voting
        self.meta_learner = meta_learner
        self.blend_weight_strategy = blend_weight_strategy
        self.model_families = model_families
        self.preprocessing = preprocessing

    @property
    def _safe_search_jobs(self) -> int:
        return max(1, min(int(self.n_jobs), 4))

    def fit(self, X: Any, y: Any, *, groups: Any = None):
        self._validate_configuration()
        X, y_arr, feature_names = validate_training_data(X, y)
        validate_task_target(y_arr, task=self.task)
        validate_groups(groups, len(y_arr))
        self._fit_reset()
        self.n_features_in_ = int(X.shape[1])
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        else:
            self.feature_names_in_ = None
        self.classes_ = np.unique(y_arr) if self.task == "classification" else None
        self.scoring_ = self.scoring or (default_classification_scoring(y_arr) if self.task == "classification" else default_regression_scoring())
        self._resolved_budget = resolve_budget(self.search_budget)
        if self.task == "classification":
            counts = np.unique(y_arr, return_counts=True)[1]
            self.class_distribution_ = {str(cls): int(count) for cls, count in zip(self.classes_, counts, strict=True)}
            if float(np.min(counts)) / float(np.max(counts)) < 0.1:
                logging.getLogger("bestatml").warning("Severe class imbalance detected; selected tree candidates use class weighting where supported.")
        self.preprocessing_ = self.preprocessing

        try:
            result = EnsembleTrainer(self).fit(X, y_arr, groups=groups)
        except (BestatMLValidationError, BestatMLTrainingError, BestatMLEnsembleError):
            self._fit_reset()
            raise
        except Exception as exc:
            self._fit_reset()
            raise BestatMLTrainingError(
                "BestatML[TrainingError]: Training could not be completed.\n\n"
                f"Problem:\n    {exc}\n\n"
                "How to fix:\n    Inspect the candidate diagnostics, reduce the search budget, "
                "or provide a compatible model/preprocessing configuration."
            ) from exc
        self._result_ = result
        self.selected_models_ = tuple(result.selected_names)
        self.model_scores_ = {r.name: r.score for r in result.candidate_results}
        self.fold_scores_ = {r.name: r.fold_scores for r in result.candidate_results}
        self.oof_predictions_ = {name: pred.copy() for name, pred in result.oof_predictions.items()}
        self.fitted_models_ = {r.name: r.fitted_estimator for r in result.candidate_results if r.name in self.selected_models_}
        self.best_model_name_ = max(self.model_scores_, key=self.model_scores_.get)
        self.best_model_ = self.fitted_models_[self.best_model_name_] if self.best_model_name_ in self.fitted_models_ else self.fitted_models_[self.selected_models_[0]]
        self.ensemble_method_ = result.ensemble.method
        self.ensemble_weights_ = result.ensemble.weights
        self.meta_learner_ = result.ensemble.meta_learner
        self._input_training_data_ = None
        if isinstance(X, pd.DataFrame):
            # Keep only schema, not data, to avoid silently retaining potentially huge user datasets.
            self.detected_numeric_features_ = tuple(X.select_dtypes(include=["number"]).columns.astype(str))
            self.detected_categorical_features_ = tuple(X.select_dtypes(exclude=["number", "datetime"]).columns.astype(str))
            self.processed_feature_names_ = None
        else:
            self.detected_numeric_features_ = tuple(f"feature_{i}" for i in range(X.shape[1]))
            self.detected_categorical_features_ = tuple()
            self.processed_feature_names_ = None
        self.training_summary_ = self._make_summary(X, y_arr, result)
        return self

    def predict(self, X: Any):
        self._ensure_fitted()
        X = validate_prediction_data(X, self.feature_names_in_, self.n_features_in_)
        predictions = self._base_predictions(X)
        if self.ensemble_method_ == "best_model":
            return self._predict_from_estimator(self.fitted_models_[self.selected_models_[0]], X)
        if self.ensemble_method_ == "voting":
            if self.task == "classification":
                proba = np.mean([predictions[n] for n in self.selected_models_], axis=0)
                return self.classes_[np.argmax(proba, axis=1)]
            return np.mean([predictions[n] for n in self.selected_models_], axis=0)
        if self.ensemble_method_ == "weighted_blending":
            weights = np.asarray(self.ensemble_weights_)
            if self.task == "classification":
                proba = np.tensordot(weights, np.stack([predictions[n] for n in self.selected_models_], axis=1), axes=(0, 1))
                return self.classes_[np.argmax(proba, axis=1)]
            return np.tensordot(weights, np.stack([predictions[n] for n in self.selected_models_], axis=1), axes=(0, 1))
        if self.ensemble_method_ == "stacking":
            meta = self._meta_matrix([predictions[n] for n in self.selected_models_])
            if self.task == "classification":
                proba = self.meta_learner_.predict_proba(meta)
                return self.classes_[np.argmax(proba, axis=1)]
            return self.meta_learner_.predict(meta)
        raise BestatMLEnsembleError(f"BestatML[EnsembleError]: Unknown fitted ensemble method {self.ensemble_method_!r}.")

    def _base_predictions(self, X: Any) -> dict[str, np.ndarray]:
        out = {}
        for name in self.selected_models_:
            model = self.fitted_models_[name]
            if self.task == "classification":
                raw = model.predict_proba(X)
                aligned = np.zeros((len(X), len(self.classes_)), dtype=float)
                local_classes = getattr(model, "classes_", self.classes_)
                for j, cls in enumerate(local_classes):
                    target_idx = int(np.flatnonzero(self.classes_ == cls)[0])
                    aligned[:, target_idx] = raw[:, j]
                out[name] = aligned
            else:
                out[name] = np.asarray(model.predict(X))
        return out

    def predict_proba(self, X: Any):
        self._ensure_fitted()
        if self.task != "classification":
            raise AttributeError("predict_proba is only available on BestatClassifier.")
        X = validate_prediction_data(X, self.feature_names_in_, self.n_features_in_)
        predictions = self._base_predictions(X)
        if self.ensemble_method_ == "best_model":
            return predictions[self.selected_models_[0]]
        if self.ensemble_method_ in {"voting", "weighted_blending"}:
            weights = self.ensemble_weights_ if self.ensemble_method_ == "weighted_blending" else np.full(len(self.selected_models_), 1.0 / len(self.selected_models_))
            proba = np.tensordot(np.asarray(weights), np.stack([predictions[n] for n in self.selected_models_], axis=1), axes=(0, 1))
            return self._normalize_probabilities(proba)
        if self.ensemble_method_ == "stacking":
            meta = self._meta_matrix([predictions[n] for n in self.selected_models_])
            return self._normalize_probabilities(self.meta_learner_.predict_proba(meta))
        raise BestatMLEnsembleError(f"BestatML[EnsembleError]: Unknown fitted ensemble method {self.ensemble_method_!r}.")

    def predict_log_proba(self, X: Any):
        return np.log(np.clip(self.predict_proba(X), 1e-15, 1.0))

    def decision_function(self, X: Any):
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def save(self, path: str) -> None:
        self._ensure_fitted()
        save_model(self, path)

    @classmethod
    def load(cls, path: str):
        return load_model(cls, path)

    def get_feature_importance(self, X: Any, y: Any, *, n_repeats: int = 5, random_state: int | None = None) -> pd.DataFrame:
        self._ensure_fitted()
        if not isinstance(n_repeats, int) or n_repeats < 1:
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: n_repeats must be an integer >= 1.")
        X = validate_prediction_data(X, self.feature_names_in_, self.n_features_in_)
        score_seed = self.random_state if random_state is None else random_state
        result = permutation_feature_importance(self, X, y, scoring=self.scoring_, n_repeats=n_repeats, random_state=score_seed)
        self._feature_importances_cache = result.copy()
        return result

    def summary(self) -> dict[str, Any]:
        self._ensure_fitted()
        return dict(self.training_summary_)

    @property
    def feature_importances_(self):
        self._ensure_fitted()
        if not hasattr(self, "_feature_importances_cache"):
            raise AttributeError("feature_importances_ is available after get_feature_importance(X, y).")
        return self._feature_importances_cache

    def _make_summary(self, X: Any, y: np.ndarray, result: Any) -> dict[str, Any]:
        dataset = {
            "samples": int(len(y)),
            "features": int(X.shape[1]),
            "numeric_features": len(self.detected_numeric_features_),
            "categorical_features": len(self.detected_categorical_features_),
        }
        if self.task == "classification":
            target = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=True)}
        else:
            target = {"min": float(np.min(y)), "max": float(np.max(y)), "mean": float(np.mean(y)), "std": float(np.std(y))}
        return {
            "task": self.task,
            "dataset": dataset,
            "target_distribution": target,
            "preprocessing": {
                "automatic": bool(self.auto_preprocess and self.preprocessing is None),
                "missing_values": bool(self.impute_missing),
                "categorical_encoding": self.categorical_strategy if self.encode_categorical else "disabled",
                "scaling": bool(self.scale_features),
                "low_variance_filter": bool(self.remove_low_variance),
                "variance_threshold": float(self.variance_threshold),
            },
            "candidate_models": list(self.model_scores_),
            "model_scores": dict(self.model_scores_),
            "selected_models": list(self.selected_models_),
            "ensemble_method": self.ensemble_method_,
            "ensemble_weights": None if self.ensemble_weights_ is None else self.ensemble_weights_.tolist(),
            "scoring": self.scoring_,
            "cv": result.cv_description,
            "timings": dict(result.timings),
            "warnings": list(result.warnings),
        }

    def _predict_from_estimator(self, estimator: Any, X: Any):
        return estimator.predict(X)

    def _meta_matrix(self, predictions: list[np.ndarray]) -> np.ndarray:
        return np.hstack([np.asarray(p) if np.asarray(p).ndim == 2 else np.asarray(p).reshape(-1, 1) for p in predictions])

    @staticmethod
    def _normalize_probabilities(proba: np.ndarray) -> np.ndarray:
        proba = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
        row_sum = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, np.where(row_sum == 0, 1.0, row_sum))

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "selected_models_") or not hasattr(self, "ensemble_method_"):
            raise BestatMLNotFittedError(
                f"BestatML[NotFittedError]: This {type(self).__name__} has not been fitted yet.\n\n"
                "How to fix:\n    Call model.fit(X, y) before prediction or inspection."
            )

    def _fit_reset(self) -> None:
        # fitted-state attributes are intentionally created only after the training pipeline succeeds.
        for name in (
            "n_features_in_", "feature_names_in_", "classes_", "scoring_", "_resolved_budget",
            "class_distribution_", "preprocessing_", "selected_models_", "model_scores_", "fold_scores_",
            "best_model_", "best_model_name_", "ensemble_method_",
            "ensemble_weights_", "meta_learner_", "training_summary_", "fitted_models_", "oof_predictions_",
            "detected_numeric_features_", "detected_categorical_features_", "processed_feature_names_",
            "_result_", "_feature_importances_cache",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _validate_configuration(self) -> None:
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise BestatMLConfigurationError(
                "BestatML[ConfigurationError]: n_jobs must be a non-zero integer; use -1 for all cores."
            )
        if not isinstance(self.max_models, int) or self.max_models < 1:
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: max_models must be an integer >= 1.")
        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: verbose must be an integer >= 0.")
        if not isinstance(self.variance_threshold, (int, float)) or self.variance_threshold < 0:
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: variance_threshold must be >= 0.")
        if self.ensemble_method not in {"auto", "best_model", "weighted_blending", "voting", "stacking"}:
            raise BestatMLConfigurationError(
                "BestatML[ConfigurationError]: ensemble_method must be auto, best_model, weighted_blending, voting, or stacking."
            )
        if self.categorical_strategy != "onehot":
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: categorical_strategy currently supports only 'onehot'.")
        if self.blend_weight_strategy != "oof_optimize":
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: blend_weight_strategy currently supports only 'oof_optimize'.")
        known_families = {"linear", "tree", "random_forest", "boosting", "catboost", "xgboost", "lightgbm"}
        if not isinstance(self.model_families, (tuple, list)) or not self.model_families:
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: model_families must not be empty.")
        unknown = sorted(set(self.model_families) - known_families)
        if unknown:
            raise BestatMLConfigurationError(
                f"BestatML[ConfigurationError]: Unknown model family/families: {unknown}. "
                f"Supported families: {sorted(known_families)}."
            )
        if self.preprocessing is not None and not hasattr(self.preprocessing, "fit"):
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: preprocessing must be sklearn-compatible.")
        if isinstance(self.cv, int):
            if self.cv < 2:
                raise BestatMLConfigurationError(f"BestatML[ConfigurationError]: cv must be an integer >= 2. Received cv={self.cv!r}.")
        elif not hasattr(self.cv, "split"):
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: cv must be an integer >= 2 or a sklearn-compatible CV splitter.")
        if not isinstance(self.search_budget, (str,)) and not hasattr(self.search_budget, "inner_cv"):
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: search_budget must be a supported preset or SearchBudget.")
        if isinstance(self.search_budget, str) and self.search_budget not in {"lightweight", "balanced", "aggressive"}:
            raise BestatMLConfigurationError("BestatML[ConfigurationError]: search_budget must be lightweight, balanced, or aggressive.")
        for name in ("auto_preprocess", "impute_missing", "encode_categorical", "scale_features", "remove_low_variance", "use_stacking", "use_weighted_blending", "use_voting", "tune_hyperparameters", "optimize", "progress"):
            if not isinstance(getattr(self, name), bool):
                raise BestatMLConfigurationError(f"BestatML[ConfigurationError]: {name} must be a boolean.")

    def __repr__(self) -> str:
        if hasattr(self, "training_summary_"):
            return (
                f"{type(self).__name__}(fitted=True, task={self.task!r}, "
                f"selected_models={len(self.selected_models_)}, ensemble={self.ensemble_method_!r}, "
                f"cv_score={max(self.model_scores_.values()):.4f})"
            )
        return (
            f"{type(self).__name__}(ensemble={self.ensemble_method_ if hasattr(self, 'ensemble_method_') else self.ensemble_method!r}, "
            f"preprocessing={'custom' if self.preprocessing is not None else 'auto'}, "
            f"tuning={self.tune_hyperparameters}, cv={self.cv!r}, random_state={self.random_state!r})"
        )
