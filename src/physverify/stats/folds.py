"""Cross-validation folds that keep each group whole."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def grouped_folds(groups: Sequence[str], k: int, seed: int) -> tuple[frozenset[str], ...]:
    """At most ``k`` disjoint sets of whole groups (scenes), shuffled by ``seed``."""
    unique = sorted(set(groups))
    if not unique:
        return ()
    order = np.random.default_rng(seed).permutation(len(unique))
    n_folds = min(k, len(unique))
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    for position, index in enumerate(order):
        folds[position % n_folds].add(unique[int(index)])
    return tuple(frozenset(fold) for fold in folds)
