"""Shared deterministic vote-count utilities for full-scale experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def sorted_counter_items(counter: Counter[Any]) -> list[tuple[Any, int]]:
    """Sort counts by descending frequency with deterministic value tie-breaking."""
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))


def majority_value(values: Iterable[Any]) -> Any:
    """Return the deterministic majority value from a non-empty iterable."""
    counts = Counter(values)
    if not counts:
        raise ValueError("Cannot compute a majority from an empty collection")
    return sorted_counter_items(counts)[0][0]


def dpa_certified_radius(winner_votes: int, competitor_votes: int) -> int:
    """Return the conservative DPA radius where a changed tie is uncertified."""
    return max(0, (winner_votes - competitor_votes - 1) // 2)
