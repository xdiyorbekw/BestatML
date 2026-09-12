from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from bestatml.exceptions import BestatMLConfigurationError, BestatMLValidationError


def build_cv(cv: int | Any, *, classification: bool, y: np.ndarray, random_state: int | None):
    if hasattr(cv, "split"):
        return cv
    if not isinstance(cv, int) or cv < 2:
        raise BestatMLConfigurationError(
            "BestatML[ConfigurationError]: cv must be an integer >= 2 or a sklearn-compatible CV splitter."
        )
    if classification:
        counts = np.unique(y, return_counts=True)[1]
        min_count = int(np.min(counts))
        if min_count < cv:
            raise BestatMLValidationError(
                "BestatML[ValidationError]: Stratified cross-validation cannot use the requested folds.\n\n"
                f"Smallest class contains:\n    {min_count} samples\nRequested folds:\n    {cv}\n\n"
                f"How to fix:\n    Reduce cv to {min_count} or provide a compatible custom splitter."
            )
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    if len(y) < cv:
        raise BestatMLValidationError(
            "BestatML[ValidationError]: K-fold cross-validation requires at least as many samples as folds.\n\n"
            f"Samples:\n    {len(y)}\nRequested folds:\n    {cv}\n\nHow to fix:\n    Reduce cv or provide a custom splitter."
        )
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)
