"""Signals beside the score that guard the pairs against reward hacking.

``motion_index`` is dimensionless and needs no metric scale: the largest displacement of each
object from its first tracked position divided by its own median extent, the maximum over
objects. A still video gives 0. A summed per-frame path would grow with tracking jitter on a
still object; the displacement from the start is bounded by the jitter amplitude instead, and it
still counts a ball that bounces back toward where it started.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from physverify.datasets.local_video import QueryEntry
from physverify.schema.states import ObjectStateTable

MIN_FRACTION: Final = 0.8
MIN_POINTS: Final = 2


def _displacement(positions: np.ndarray) -> float | None:
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    if len(finite) < MIN_POINTS:
        return None
    return float(np.max(np.linalg.norm(finite - finite[0], axis=1)))


def _extent(sizes: np.ndarray) -> float | None:
    finite = sizes[np.isfinite(sizes)]
    finite = finite[finite > 0]
    return float(np.median(finite)) if finite.size else None


def motion_index(states: ObjectStateTable) -> float | None:
    """max over objects of displacement / median extent; None when no object has both."""
    values = []
    for n in range(states.n_objects):
        length = _displacement(states.position[:, n, :])
        extent = _extent(states.size[:, n, :])
        if length is not None and extent is not None:
            values.append(length / extent)
    return max(values) if values else None


def objects_ok(
    states: ObjectStateTable,
    queries: Sequence[str | QueryEntry],
    min_fraction: float = MIN_FRACTION,
) -> bool | None:
    """Each query with an ``expected_count`` is visible that many times in >= ``min_fraction`` of
    frames (counted by label); None when no query declares a count."""
    expected = {
        q.phrase: q.expected_count
        for q in queries
        if isinstance(q, QueryEntry) and q.expected_count
    }
    if not expected:
        return None
    if states.n_frames == 0:
        return False
    for label, count in expected.items():
        columns = np.flatnonzero(states.labels == label)
        per_frame = (
            states.visible[:, columns].sum(axis=1) if columns.size else np.zeros(states.n_frames)
        )
        if np.mean(per_frame >= count) < min_fraction:
            return False
    return True
