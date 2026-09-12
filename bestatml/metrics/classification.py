from __future__ import annotations

import numpy as np


def default_classification_scoring(y: np.ndarray) -> str:
    n_classes = len(np.unique(y))
    return "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
