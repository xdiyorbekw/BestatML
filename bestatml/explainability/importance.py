from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_feature_importance(estimator: Any, X: Any, y: Any, *, scoring: str, n_repeats: int, random_state: int | None) -> pd.DataFrame:
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    if isinstance(X, pd.DataFrame):
        names = list(map(str, X.columns))
    else:
        names = [f"feature_{i}" for i in range(result.importances_mean.shape[0])]
    out = pd.DataFrame({
        "feature": names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    })
    return out.sort_values("importance", ascending=False, ignore_index=True)
