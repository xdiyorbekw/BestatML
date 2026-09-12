from __future__ import annotations

import logging
from collections.abc import Iterator

from bestatml.utils.logging import get_logger


def progress_iter(items: list, *, enabled: bool, verbose: int, label: str) -> Iterator:
    logger = get_logger()
    total = len(items)
    for idx, item in enumerate(items, start=1):
        if enabled or verbose >= 2:
            logger.info("[%d/%d] %s: %s", idx, total, label, getattr(item, "name", item))
        yield item
