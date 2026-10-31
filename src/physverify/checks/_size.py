"""Shared pieces of the R4_size implementations: whole rows, their lengths and the first-order
sigma of a size ratio."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from physverify.physics.geometry import characteristic_lengths
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def whole_rows(in_frame: BoolArray, occlusion: FloatArray, occluded_at: float) -> BoolArray:
    """In frame and not known to be occluded (NaN >= x is False, so NaN keeps the row)."""
    return in_frame & ~(occlusion >= occluded_at)


def whole_lengths(states: ObjectStateTable, whole: BoolArray) -> FloatArray:
    """(T, N) lengths of every object, NaN on rows where ``whole`` is False."""
    return np.where(whole, characteristic_lengths(states.size), math.nan)


def size_sigma(ratio: float, extents: FloatArray, extents_sigma: FloatArray) -> float:
    """First order: the ratio L_t / median(L) carries ``length_rel_sigma`` twice (independent
    errors of both lengths)."""
    return ratio * math.sqrt(2.0) * length_rel_sigma(extents, extents_sigma)


def length_rel_sigma(extents: FloatArray, extents_sigma: FloatArray) -> float:
    """sigma_L / L = sqrt(sum (sigma_e / e)^2) / 3 for L = (e1 e2 e3)^(1/3)."""
    return math.sqrt(float(np.sum((extents_sigma / extents) ** 2))) / 3.0


def relative_sigma(
    states: ObjectStateTable, whole: FloatArray, index: int, row: int, ratio: float
) -> float:
    """First order, as ``size_sigma``, for r = L_i / ref: the relative sigma of the geometric
    mean of n references is sqrt(mean_j (sigma_Lj / L_j)^2 / n), over the references the
    magnitude used at that row (``whole``: the lengths of ``whole_lengths``)."""
    own = length_rel_sigma(states.size[row, index], states.size_sigma[row, index])
    refs = [
        length_rel_sigma(states.size[row, j], states.size_sigma[row, j])
        for j in range(states.n_objects)
        if j != index and math.isfinite(float(whole[row, j])) and whole[row, j] > 0.0
    ]
    reference = math.sqrt(float(np.mean(np.square(refs))) / len(refs))
    return ratio * math.sqrt(2.0) * math.hypot(own, reference)
