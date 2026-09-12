from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV

from bestatml.core.configuration import SearchBudget


def tune_pipeline(pipeline: Any, search_space: dict[str, list], *, X: Any, y: Any, cv: Any, scoring: str, random_state: int | None, budget: SearchBudget, n_jobs: int):
    if not search_space:
        pipeline.fit(X, y)
        return pipeline
    search = RandomizedSearchCV(
        pipeline,
        search_space,
        n_iter=min(budget.n_iter, max(1, _space_size(search_space))),
        scoring=scoring,
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_


def _space_size(space: dict[str, list]) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
        if size > 10_000:
            return 10_000
    return size
