"""R4_size size constancy per object."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._params import OcclusionParams
from physverify.checks._results import abstain, first_max, unscored
from physverify.physics.geometry import characteristic_lengths
from physverify.physics.measures.object_integrity import size_change_ratio
from physverify.physics.measures.relative_size import relative_lengths, relative_size_change
from physverify.physics.tracks import native_span
from physverify.schema.checks import AbstainReason, CheckResult

FloatArray = NDArray[np.float64]


SizeReference = Literal["self", "others"]


class SizeParams(OcclusionParams):
    """``reference``: "others" (v2) measures the size relative to the other
    bodies of each frame; "self" (v1) the size alone."""

    min_size_frames: int = Field(default=3, ge=2)
    reference: SizeReference = "others"


class SizeConstancyCheck:
    """R4_size [rel_size_change]: max_t |r_t / median(r) - 1| with r_t = L_t / ref_t.

    ``reference`` "others" (v2): ref_t is the geometric mean of the lengths of the other objects
    whole in the same row, each relative to its own median (``relative_lengths``), so a
    depth-scale error common to the row cancels and a reference leaving the image does not make
    a step; rows without any such object do not count, and a clip with one object abstains
    (``details.reference`` "none").
    With two objects a change of either shows in both, so the subject is ambiguous. "self" (v1):
    ref_t = 1.

    Rows out of frame or with a known occlusion >= ``occluded_at`` are left out (for the object
    and for its references): their extents describe the visible part, not the object. Unknown
    (NaN) occlusion keeps the row, since perceived tables never carry occlusion; a truncated
    perceived mask (at the image border or behind another object) therefore still enters the
    measure. ABSTAIN when fewer than ``min_size_frames`` rows with known extents (and a
    reference) remain.
    """

    check_id: ClassVar[str] = "R4_size"
    requires: ClassVar[frozenset[str]] = frozenset()
    magnitude_unit: ClassVar[str] = "rel_size_change"
    Params: ClassVar[type[BaseModel]] = SizeParams

    def __init__(self, params: BaseModel) -> None:
        self.params = SizeParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        return [self._object(inputs, i) for i in range(inputs.states.n_objects)]

    def _object(self, inputs: CheckInputs, index: int) -> CheckResult:
        if self.params.reference == "others":
            return self._relative(inputs, index)
        states = inputs.states
        subject = (str(states.object_ids[index]),)
        size = states.size[:, index]
        lengths = characteristic_lengths(size)
        whole = self._whole_rows(states.in_frame[:, index], states.occlusion[:, index])
        known = np.flatnonzero(whole & np.isfinite(lengths))
        excluded = int(np.count_nonzero(~whole))
        if known.size < self.params.min_size_frames:
            details = {
                "size_rows": int(known.size),
                "excluded_rows": excluded,
                "min_size_frames": self.params.min_size_frames,
            }
            return abstain(type(self), inputs, subject, AbstainReason.TOO_FEW_FRAMES, None, details)
        ratios = lengths[known] / np.median(lengths[known])
        pick = first_max(np.abs(ratios - 1.0))
        row = int(known[pick])
        sigma = _size_sigma(float(ratios[pick]), size[row], states.size_sigma[row, index])
        span = native_span(states.frame_index, row, row)
        details = {"row": row, "size_rows": int(known.size), "excluded_rows": excluded}
        magnitude = size_change_ratio(size[known])
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)

    def _whole_rows(self, in_frame: NDArray[np.bool_], occlusion: FloatArray) -> NDArray[np.bool_]:
        """In frame and not known to be occluded (NaN >= x is False, so NaN keeps the row)."""
        return in_frame & ~(occlusion >= self.params.occluded_at)

    def _whole_lengths(self, inputs: CheckInputs) -> FloatArray:
        """(T, N) lengths of every object, NaN on rows where it is not whole."""
        states = inputs.states
        lengths = characteristic_lengths(states.size)
        whole = self._whole_rows(states.in_frame, states.occlusion)
        return np.where(whole, lengths, math.nan)

    def _relative(self, inputs: CheckInputs, index: int) -> CheckResult:
        states = inputs.states
        subject = (str(states.object_ids[index]),)
        lengths = self._whole_lengths(inputs)
        own, others = lengths[:, index], np.delete(lengths, index, axis=1)
        referenced = np.any(np.isfinite(others), axis=1)
        ratios = relative_lengths(own, others)
        known = np.flatnonzero(np.isfinite(ratios))
        details: dict[str, object] = {
            "reference": "others" if bool(referenced.any()) else "none",
            "size_rows": int(known.size),
            "excluded_rows": int(np.count_nonzero(~np.isfinite(own))),
            "unreferenced_rows": int(np.count_nonzero(np.isfinite(own) & ~referenced)),
        }
        if known.size < self.params.min_size_frames:
            details["min_size_frames"] = self.params.min_size_frames
            return abstain(type(self), inputs, subject, AbstainReason.TOO_FEW_FRAMES, None, details)
        scaled = ratios[known] / np.median(ratios[known])
        pick = first_max(np.abs(scaled - 1.0))
        row = int(known[pick])
        sigma = _relative_sigma(inputs, lengths, index, row, float(scaled[pick]))
        span = native_span(states.frame_index, row, row)
        magnitude = relative_size_change(own, others)
        details["row"] = row
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)


def _size_sigma(ratio: float, extents: FloatArray, extents_sigma: FloatArray) -> float:
    """First order: the ratio L_t / median(L) carries ``_length_rel_sigma`` twice (independent
    errors of both lengths)."""
    return ratio * math.sqrt(2.0) * _length_rel_sigma(extents, extents_sigma)


def _length_rel_sigma(extents: FloatArray, extents_sigma: FloatArray) -> float:
    """sigma_L / L = sqrt(sum (sigma_e / e)^2) / 3 for L = (e1 e2 e3)^(1/3)."""
    return math.sqrt(float(np.sum((extents_sigma / extents) ** 2))) / 3.0


def _relative_sigma(
    inputs: CheckInputs, whole: FloatArray, index: int, row: int, ratio: float
) -> float:
    """First order, as ``_size_sigma``, for r = L_i / ref: the relative sigma of the geometric
    mean of n references is sqrt(mean_j (sigma_Lj / L_j)^2 / n), over the references the
    magnitude used at that row (``whole``: the lengths of ``_whole_lengths``)."""
    states = inputs.states
    own = _length_rel_sigma(states.size[row, index], states.size_sigma[row, index])
    refs = [
        _length_rel_sigma(states.size[row, j], states.size_sigma[row, j])
        for j in range(states.n_objects)
        if j != index and math.isfinite(float(whole[row, j])) and whole[row, j] > 0.0
    ]
    reference = math.sqrt(float(np.mean(np.square(refs))) / len(refs))
    return ratio * math.sqrt(2.0) * math.hypot(own, reference)


PLUGINS = (SizeConstancyCheck,)
