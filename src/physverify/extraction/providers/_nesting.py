"""Tracks whose mask lies inside another track's mask: a part of an object or a second
detection of it, which the lift would place as a body interpenetrating the first."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from physverify.schema.perception import MaskTrack

FloatArray = NDArray[np.float64]


def containment(masks: NDArray[np.bool_]) -> FloatArray:
    """(N, N) median over the frames where both masks are non-empty of |A_i & A_j| / |A_i|;
    NaN on the diagonal and for pairs never visible together."""
    areas = masks.sum(axis=(2, 3)).astype(np.float64)
    n = masks.shape[1]
    found = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            both = (areas[:, i] > 0) & (areas[:, j] > 0)
            if i == j or not both.any():
                continue
            shared = (masks[both, i] & masks[both, j]).sum(axis=(1, 2))
            found[i, j] = float(np.median(shared / areas[both, i]))
    return found


def outer_tracks(masks: NDArray[np.bool_], max_overlap: float) -> list[int]:
    """Indices kept, largest median area first: a track is dropped when more than
    ``max_overlap`` of it lies inside a kept one, so of two duplicates the larger stays."""
    areas = masks.sum(axis=(2, 3)).astype(np.float64)
    visible = np.where(areas > 0, areas, np.nan)
    median_area = np.array([_nanmedian(visible[:, n]) for n in range(masks.shape[1])])
    inside = containment(masks)
    kept: list[int] = []
    for n in np.argsort(-median_area, kind="stable").tolist():
        if not any(inside[n, k] > max_overlap for k in kept):
            kept.append(n)
    return sorted(kept)


def without_nested(track: MaskTrack, max_overlap: float) -> MaskTrack:
    """``track`` without the tracks ``outer_tracks`` drops, in their original order."""
    kept = outer_tracks(track.masks, max_overlap)
    if len(kept) == track.masks.shape[1]:
        return track
    return replace(
        track,
        object_ids=track.object_ids[kept],
        labels=track.labels[kept],
        masks=track.masks[:, kept],
        scores=track.scores[:, kept],
    )


def _nanmedian(values: FloatArray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else 0.0
