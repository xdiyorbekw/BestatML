from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBudget:
    n_iter: int
    inner_cv: int


BUDGETS = {
    "lightweight": SearchBudget(n_iter=1, inner_cv=3),
    "balanced": SearchBudget(n_iter=2, inner_cv=3),
    "aggressive": SearchBudget(n_iter=5, inner_cv=4),
}


def resolve_budget(name: str) -> SearchBudget:
    try:
        return BUDGETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown search_budget={name!r}; choose from {', '.join(BUDGETS)}."
        ) from exc
