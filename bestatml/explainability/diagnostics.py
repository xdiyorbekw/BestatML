from __future__ import annotations

from typing import Any


def build_summary(model: Any) -> dict[str, Any]:
    return dict(model.training_summary_)
