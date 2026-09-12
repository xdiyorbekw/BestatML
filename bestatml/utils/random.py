from __future__ import annotations

from typing import Any


def child_seed(random_state: int | None, offset: int = 0) -> int | None:
    if random_state is None:
        return None
    return int((random_state + 1009 * (offset + 1)) % (2**32 - 1))


def seed_params(params: dict[str, Any], random_state: int | None) -> dict[str, Any]:
    if random_state is None:
        return params
    out = dict(params)
    if "random_state" in out and out["random_state"] is None:
        out["random_state"] = random_state
    return out
