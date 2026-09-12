from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bestatml.core.configuration import SearchBudget
from bestatml.core.ensemble import EnsembleResult, make_blend, optimize_weights, score_predictions, select_diverse_models
from bestatml.core.validation import build_cv
from bestatml.exceptions import BestatMLEnsembleError, BestatMLDependencyError, BestatMLTrainingError
from bestatml.models.registry import ModelSpec, get_registry, missing_optional_dependencies
from bestatml.optimization.tuning import tune_pipeline
from bestatml.preprocessing.pipeline import SchemaPreprocessor
from bestatml.utils.logging import get_logger
from bestatml.utils.random import child_seed


@dataclass
class CandidateResult:
    name: str
    family: str
    score: float
    oof_prediction: np.ndarray
    fitted_estimator: Any
    fold_scores: tuple[float, ...]
    failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class TrainingResult:
    candidate_results: list[CandidateResult]
    selected_names: list[str]
    oof_predictions: dict[str, np.ndarray]
    ensemble: EnsembleResult
    timings: dict[str, float]
    warnings: list[str]
    cv_description: str


def _to_dense(x: Any) -> Any:
    return x.toarray() if hasattr(x, "toarray") else x


class EnsembleTrainer:
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.logger = get_logger()

    def fit(self, X: Any, y: np.ndarray, *, groups: Any = None) -> TrainingResult:
        estimator = self.estimator
        started = time.perf_counter()
        families = tuple(estimator.model_families)
        specs = get_registry(estimator.task, families, estimator.random_state, estimator.n_jobs)
        if not specs:
            missing = missing_optional_dependencies(estimator.task, families)
            if missing and tuple(estimator.model_families) != estimator._default_model_families:
                family, package = missing[0]
                raise BestatMLDependencyError(
                    f"BestatML[DependencyError]: {family} support was explicitly requested, "
                    f"but {package} is not installed.\n\n"
                    f"How to fix:\n    Install it with `python -m pip install {package}` "
                    "or install BestatML's optional boosting dependencies."
                )
            raise BestatMLTrainingError(
                "BestatML[TrainingError]: No compatible model candidates are enabled.\n\n"
                f"Requested model families:\n    {families}\n\n"
                "How to fix:\n    Enable at least one supported model family."
            )

        budget = estimator._resolved_budget
        cv = build_cv(
            estimator.cv,
            classification=estimator.task == "classification",
            y=y,
            random_state=estimator.random_state,
        )
        results: list[CandidateResult] = []
        warnings_list: list[str] = []
        for index, spec in enumerate(specs):
            try:
                result = self._evaluate_candidate(spec, X, y, cv, budget, groups=groups, index=index)
            except Exception as exc:
                message = f"Candidate {spec.name!r} failed and was skipped: {type(exc).__name__}: {exc}"
                warnings_list.append(message)
                self.logger.debug(message, exc_info=True)
                if estimator.verbose:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            results.append(result)
            if estimator.verbose:
                self.logger.info("Candidate %-32s score=%.6f", spec.name, result.score)

        if not results:
            detail = "\n".join(f"- {message}" for message in warnings_list)
            raise BestatMLTrainingError(
                "BestatML[TrainingError]: All candidate models failed during cross-validation.\n\n"
                "Candidate diagnostics:\n"
                f"{detail or '- No candidate diagnostics were recorded.'}\n\n"
                "How to fix:\n    Check the reported candidate errors, reduce the search budget, "
                "adjust preprocessing, or restrict model_families to compatible candidates."
            )

        scores = {r.name: r.score for r in results}
        oof_predictions = {r.name: r.oof_prediction for r in results}
        selected_names = select_diverse_models(
            list(scores), scores, oof_predictions, estimator.max_models,
            classification=estimator.task == "classification",
        )
        selected_results = {r.name: r for r in results}
        final_start = time.perf_counter()
        for selected_index, name in enumerate(selected_names):
            spec = next(s for s in specs if s.name == name)
            base = self._make_pipeline(spec, estimator.random_state, estimator.n_jobs)
            if estimator.tune_hyperparameters and spec.search_space:
                inner_count = self._safe_inner_cv_budget(y, budget.inner_cv)
                if inner_count is None:
                    fitted = base.fit(X, y)
                    warnings_list.append(
                        f"Hyperparameter tuning was skipped for {name!r} because the selected data cannot support inner CV."
                    )
                else:
                    inner_cv = build_cv(
                        inner_count, classification=estimator.task == "classification", y=y,
                        random_state=child_seed(estimator.random_state, 500 + selected_index),
                    )
                    fitted = tune_pipeline(
                        base, spec.search_space, X=X, y=y, cv=inner_cv, scoring=estimator.scoring_,
                        random_state=child_seed(estimator.random_state, 800 + selected_index),
                        budget=budget, n_jobs=estimator._safe_search_jobs,
                    )
            else:
                fitted = base.fit(X, y)
            selected_results[name].fitted_estimator = fitted

        ensemble_start = time.perf_counter()
        ensemble = self._build_ensemble(selected_results, selected_names, oof_predictions, y)
        timings = {
            "model_evaluation": final_start - started,
            "final_fitting": ensemble_start - final_start,
            "ensemble_optimization": time.perf_counter() - ensemble_start,
            "total": time.perf_counter() - started,
        }
        return TrainingResult(results, selected_names, oof_predictions, ensemble, timings, warnings_list, repr(cv))

    def _safe_inner_cv_budget(self, y: np.ndarray, requested: int) -> int | None:
        if requested < 2:
            return None
        if self.estimator.task == "classification":
            min_count = int(np.min(np.unique(y, return_counts=True)[1]))
            return min(requested, min_count) if min_count >= 2 else None
        return requested if requested <= len(y) else None

    def _make_pipeline(self, spec: ModelSpec, random_state: int | None, n_jobs: int):
        model = spec.factory(random_state, max(1, min(n_jobs, 4)))
        if self.estimator.preprocessing is not None:
            preprocessor = clone(self.estimator.preprocessing)
        elif not self.estimator.auto_preprocess:
            preprocessor = FunctionTransformer(validate=False)
        else:
            preprocessor = SchemaPreprocessor(
                scale_numeric=(self.estimator.scale_features and spec.family == "linear"),
                impute_missing=self.estimator.impute_missing,
                encode_categorical=self.estimator.encode_categorical,
                remove_low_variance=self.estimator.remove_low_variance,
                variance_threshold=self.estimator.variance_threshold,
            )
        if spec.dense:
            preprocessor = Pipeline([
                ("base", preprocessor),
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ])
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def _evaluate_candidate(self, spec, X, y, cv, budget, *, groups, index):
        splits = list(cv.split(X, y, groups) if groups is not None else cv.split(X, y))
        if self.estimator.task == "classification":
            oof = np.zeros((len(y), len(self.estimator.classes_)), dtype=float)
        else:
            oof = np.zeros(len(y), dtype=float)
        fold_scores: list[float] = []
        failures: list[str] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            try:
                pipeline = self._make_pipeline(spec, child_seed(self.estimator.random_state, index * 100 + fold_idx), 1)
                train_X = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
                valid_X = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
                if self.estimator.tune_hyperparameters and spec.search_space:
                    inner_count = self._safe_inner_cv_budget(y[train_idx], budget.inner_cv)
                    if inner_count is None:
                        pipeline.fit(train_X, y[train_idx])
                    else:
                        inner = build_cv(
                            inner_count, classification=self.estimator.task == "classification",
                            y=y[train_idx], random_state=child_seed(self.estimator.random_state, 10000 + index * 100 + fold_idx),
                        )
                        pipeline = tune_pipeline(
                            pipeline, spec.search_space, X=train_X, y=y[train_idx], cv=inner,
                            scoring=self.estimator.scoring_,
                            random_state=child_seed(self.estimator.random_state, 20000 + index * 100 + fold_idx),
                            budget=budget, n_jobs=1,
                        )
                else:
                    pipeline.fit(train_X, y[train_idx])

                if self.estimator.task == "classification":
                    pred = pipeline.predict_proba(valid_X)
                    classes = np.asarray(getattr(pipeline, "classes_", self.estimator.classes_))
                    aligned = np.zeros((len(valid_idx), len(self.estimator.classes_)), dtype=float)
                    for local_idx, cls in enumerate(classes):
                        matches = np.flatnonzero(self.estimator.classes_ == cls)
                        if len(matches) != 1:
                            raise BestatMLEnsembleError(f"Model {spec.name!r} returned unexpected class {cls!r}.")
                        aligned[:, int(matches[0])] = pred[:, local_idx]
                    oof[valid_idx] = aligned
                    score_pred = aligned
                else:
                    pred = pipeline.predict(valid_X)
                    oof[valid_idx] = pred
                    score_pred = pred
                score = score_predictions(
                    y[valid_idx], score_pred,
                    classification=self.estimator.task == "classification",
                    classes=self.estimator.classes_, scoring=self.estimator.scoring_,
                )
                if not np.isfinite(score):
                    raise BestatMLTrainingError("cross-validation score was not finite")
                fold_scores.append(float(score))
            except Exception as exc:
                failures.append(f"fold {fold_idx}: {type(exc).__name__}: {exc}")
        if len(fold_scores) != len(splits):
            detail = "; ".join(failures) or "one or more folds failed"
            raise BestatMLTrainingError(f"{spec.name} could not produce complete OOF predictions: {detail}")
        return CandidateResult(
            spec.name, spec.family, float(np.mean(fold_scores)), oof,
            self._make_pipeline(spec, self.estimator.random_state, self.estimator.n_jobs),
            tuple(fold_scores), tuple(failures),
        )

    def _build_ensemble(self, selected, names, oof, y):
        if len(names) == 1:
            return EnsembleResult("best_model", None, None)
        available = [name for name in names if name in selected]
        if not available:
            raise BestatMLEnsembleError("BestatML[EnsembleError]: No selected models are available.")
        if self.estimator.ensemble_method != "auto":
            return self._explicit_ensemble(self.estimator.ensemble_method, available, oof, y)
        candidates = [("best_model", max(selected[n].score for n in available), EnsembleResult("best_model", None, None))]
        if self.estimator.use_weighted_blending:
            try:
                weights = optimize_weights([oof[n] for n in available], y, classification=self.estimator.task == "classification", classes=self.estimator.classes_, scoring=self.estimator.scoring_)
                pred = make_blend([oof[n] for n in available], weights, classification=self.estimator.task == "classification")
                candidates.append(("weighted_blending", score_predictions(y, pred, classification=self.estimator.task == "classification", classes=self.estimator.classes_, scoring=self.estimator.scoring_), EnsembleResult("weighted_blending", weights, None)))
            except Exception as exc:
                self.logger.debug("Weighted blending candidate failed: %s", exc, exc_info=True)
        if self.estimator.use_stacking:
            meta = self._fit_meta(available, oof, y)
            if meta is not None:
                try:
                    pred = self._meta_predict(meta, [oof[n] for n in available])
                    candidates.append(("stacking", score_predictions(y, pred, classification=self.estimator.task == "classification", classes=self.estimator.classes_, scoring=self.estimator.scoring_), EnsembleResult("stacking", None, meta)))
                except Exception as exc:
                    self.logger.debug("Stacking scoring failed: %s", exc, exc_info=True)
        if self.estimator.use_voting:
            try:
                pred = np.mean([oof[n] for n in available], axis=0)
                candidates.append(("voting", score_predictions(y, pred, classification=self.estimator.task == "classification", classes=self.estimator.classes_, scoring=self.estimator.scoring_), EnsembleResult("voting", np.full(len(available), 1.0 / len(available)), None)))
            except Exception as exc:
                self.logger.debug("Voting candidate failed: %s", exc, exc_info=True)
        return max(candidates, key=lambda x: x[1])[2]

    def _explicit_ensemble(self, method, names, oof, y):
        if method == "best_model":
            return EnsembleResult(method, None, None)
        if method == "weighted_blending":
            weights = optimize_weights([oof[n] for n in names], y, classification=self.estimator.task == "classification", classes=self.estimator.classes_, scoring=self.estimator.scoring_)
            return EnsembleResult(method, weights, None)
        if method == "voting":
            return EnsembleResult(method, np.full(len(names), 1.0 / len(names)), None)
        if method == "stacking":
            meta = self._fit_meta(names, oof, y)
            if meta is not None:
                return EnsembleResult(method, None, meta)
            weights = optimize_weights([oof[n] for n in names], y, classification=self.estimator.task == "classification", classes=self.estimator.classes_, scoring=self.estimator.scoring_)
            self.logger.warning("Stacking failed; falling back to weighted blending.")
            return EnsembleResult("weighted_blending", weights, None)
        raise BestatMLEnsembleError(f"BestatML[EnsembleError]: Unknown ensemble_method={method!r}.")

    def _fit_meta(self, names, oof, y):
        from sklearn.linear_model import LogisticRegression, Ridge
        meta_X = self._meta_matrix([oof[n] for n in names])
        model = clone(self.estimator.meta_learner) if self.estimator.meta_learner is not None else (
            LogisticRegression(max_iter=1000, C=1.0, random_state=self.estimator.random_state)
            if self.estimator.task == "classification" else Ridge(alpha=1.0)
        )
        try:
            model.fit(meta_X, y)
            return model
        except Exception as exc:
            self.logger.warning("Meta learner failed: %s", exc)
            return None

    @staticmethod
    def _meta_matrix(predictions):
        return np.hstack([np.asarray(p) if np.asarray(p).ndim == 2 else np.asarray(p).reshape(-1, 1) for p in predictions])

    def _meta_predict(self, meta, predictions):
        meta_X = self._meta_matrix(predictions)
        return meta.predict_proba(meta_X) if self.estimator.task == "classification" else meta.predict(meta_X)
