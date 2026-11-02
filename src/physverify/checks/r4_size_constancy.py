"""R4_size size constancy per object."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import OcclusionParams
from physverify.checks._peaks import first_max
from physverify.checks._size import relative_sigma, size_sigma, whole_lengths, whole_rows
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
    version: ClassVar[str] = "1"
    requires: ClassVar[frozenset[str]] = frozenset()
    time_dependent: ClassVar[bool] = False
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
        whole = whole_rows(
            states.in_frame[:, index], states.occlusion[:, index], self.params.occluded_at
        )
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
        sigma = size_sigma(float(ratios[pick]), size[row], states.size_sigma[row, index])
        span = native_span(states.frame_index, row, row)
        details = {"row": row, "size_rows": int(known.size), "excluded_rows": excluded}
        magnitude = size_change_ratio(size[known])
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)

    def _whole_lengths(self, inputs: CheckInputs) -> FloatArray:
        states = inputs.states
        return whole_lengths(
            states, whole_rows(states.in_frame, states.occlusion, self.params.occluded_at)
        )

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
        sigma = relative_sigma(inputs.states, lengths, index, row, float(scaled[pick]))
        span = native_span(states.frame_index, row, row)
        magnitude = relative_size_change(own, others)
        details["row"] = row
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)


PLUGINS = (SizeConstancyCheck,)
