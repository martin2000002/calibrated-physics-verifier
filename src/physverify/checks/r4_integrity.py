"""R4 object integrity per object: permanence (R4_missing), constant size (R4_size) and the v1
spatio-temporal continuity (``ContinuityCheck``, out of the registry since it4: R4_jump is
``r4_jump_noise.NoiseContinuityCheck``). ADR 0006 section 3.4 and its it4 amendment."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._common import CommonParams, OcclusionParams, abstain, first_max, unscored
from physverify.physics.finite_diff import third_difference, third_difference_sigma
from physverify.physics.geometry import characteristic_length, characteristic_lengths
from physverify.physics.measures.r4_integrity import jump_ratio, missing_frames, size_change_ratio
from physverify.physics.measures.r4_size_relative import relative_lengths, relative_size_change
from physverify.physics.tracks import (
    impact_stencil_mask,
    median_extents,
    native_span,
    row_step,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.contacts import ContactTable

FloatArray = NDArray[np.float64]


class PermanenceParams(OcclusionParams):
    # "unknown": sigma NaN unless the positions are exact (before 2026-10-20). "edges": the
    # estimated counting sigma of ``count_sigma``, whose level the D9 factor sets.
    sigma_model: Literal["unknown", "edges"] = "edges"


SizeReference = Literal["self", "others"]


class SizeParams(OcclusionParams):
    """``reference``: "others" (v2, plan-it4 change B) measures the size relative to the other
    bodies of each frame; "self" (v1) the size alone."""

    min_size_frames: int = Field(default=3, ge=2)
    reference: SizeReference = "others"


class PermanenceCheck:
    """R4_missing [frames]: native frames in which the object is inside the image but unseen and
    not explained by occlusion. Rows are converted to native frames (x row_step, D14.6)."""

    check_id: ClassVar[str] = "R4_missing"
    requires: ClassVar[frozenset[str]] = frozenset()
    magnitude_unit: ClassVar[str] = "frames"
    Params: ClassVar[type[BaseModel]] = PermanenceParams

    def __init__(self, params: BaseModel) -> None:
        self.params = PermanenceParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        return [self._object(inputs, i) for i in range(inputs.states.n_objects)]

    def _object(self, inputs: CheckInputs, index: int) -> CheckResult:
        states = inputs.states
        fi = states.frame_index
        subject = (str(states.object_ids[index]),)
        shown, inside = states.visible[:, index], states.in_frame[:, index]
        occlusion = states.occlusion[:, index]
        unresolved = np.flatnonzero(inside & ~shown & np.isnan(occlusion))
        if unresolved.size:
            hidden = native_span(fi, int(unresolved[0]), int(unresolved[-1]))
            details = {"unresolved_rows": int(unresolved.size)}
            return abstain(type(self), inputs, subject, AbstainReason.OCCLUDED, hidden, details)
        at = self.params.occluded_at
        n_rows = missing_frames(shown, inside, occlusion, at)
        rows = [
            k
            for k in range(fi.shape[0])
            if missing_frames(shown[k : k + 1], inside[k : k + 1], occlusion[k : k + 1], at)
        ]
        span = native_span(fi, rows[0], rows[-1]) if rows else None
        step = row_step(fi)
        sigma, estimated = self._sigma(states.position_sigma[:, index], rows, step)
        details = {"missing_rows": n_rows, **({"sigma_estimated": True} if estimated else {})}
        return unscored(type(self), inputs, subject, n_rows * step, sigma, span, details)

    def _sigma(self, position_sigma: FloatArray, rows: list[int], step: int) -> tuple[float, bool]:
        """(sigma, estimated): 0 for exact positions, else the edge model or, before, unknown."""
        if bool(np.all(position_sigma == 0.0)):
            return 0.0, False
        if self.params.sigma_model == "edges":
            return count_sigma(rows, step), True
        return math.nan, False


def count_sigma(rows: list[int], step: int) -> float:
    """Estimated sigma of a missing-frame count: step * sqrt(2 * max(runs, 1)).

    Each run of counted rows has two edges whose row is uncertain by one sampling step (a mask
    that fades under ``min_area_px`` a row early or late). A shape only: D9 sets its level."""
    runs = sum(1 for k, row in enumerate(rows) if k == 0 or row != rows[k - 1] + 1)
    return step * math.sqrt(2.0 * max(runs, 1))


class SizeConstancyCheck:
    """R4_size [rel_size_change]: max_t |r_t / median(r) - 1| with r_t = L_t / ref_t.

    ``reference`` "others" (v2): ref_t is the geometric mean of the lengths of the other objects
    whole in the same row, so a depth-scale error common to the row cancels; rows without any
    such object do not count, and a clip with one object abstains (``details.reference`` "none").
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
        sigma = _relative_sigma(inputs, index, row, float(scaled[pick]))
        span = native_span(states.frame_index, row, row)
        magnitude = relative_size_change(own, others)
        details["row"] = row
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)


class ContinuityCheck:
    """R4_jump v1 (ADR 0006 section 3.4 as preregistered; out of the registry since it4, kept for
    the v1 arm of the final evaluation, judgement J4 of plan-it4).

    R4_jump [jump_over_size]: max ||p[k+2] - 3 p[k+1] + 3 p[k] - p[k-1]|| / (2 L) over
    consecutive rows, skipping the rows whose stencil spans an impact (D13, D14.6: the smooth
    part grows with (row_step * dt)^3 and vanishes under constant acceleration, a teleport does
    not).

    Like every maximum over noisy rows, the value is biased upward: with position noise sigma_p
    a clean track gives about sqrt(20) sigma_p / (2 L) per row and more at the maximum, growing
    with the number of rows. ``sigma`` is the per-row sigma at the peak, not the spread of the
    maximum. The bias is not corrected here: tau is a conformal quantile of this same maximum
    over clean clips (``calibration.thresholds.fit_tau``), so the null distribution already
    contains it."""

    check_id: ClassVar[str] = "R4_jump"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    magnitude_unit: ClassVar[str] = "jump_over_size"
    Params: ClassVar[type[BaseModel]] = CommonParams

    def __init__(self, params: BaseModel) -> None:
        self.params = CommonParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        if inputs.contacts is None:
            raise ValueError(f"{self.check_id} requires contacts")
        contacts = inputs.contacts
        return [self._object(inputs, contacts, i) for i in range(inputs.states.n_objects)]

    def _object(self, inputs: CheckInputs, contacts: ContactTable, index: int) -> CheckResult:
        states = inputs.states
        object_id = str(states.object_ids[index])
        position = states.position[:, index]
        extents = median_extents(states, index)
        exclude = impact_stencil_mask(object_id, contacts, states.frame_index)
        magnitude = jump_ratio(position, extents, exclude)
        details = {"excluded_rows": int(np.count_nonzero(exclude))}
        if not math.isfinite(magnitude):
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, (object_id,), reason, None, details)
        per_row = np.array(
            [
                jump_ratio(position[k - 1 : k + 3], extents, exclude[k - 1 : k + 3])
                for k in range(1, position.shape[0] - 2)
            ]
        )
        k = 1 + first_max(per_row)
        sigma = _jump_sigma(
            position[k - 1 : k + 3], states.position_sigma[k, index], characteristic_length(extents)
        )
        span = native_span(states.frame_index, k - 1, k + 2)
        return unscored(
            type(self), inputs, (object_id,), magnitude, sigma, span, {**details, "row": k}
        )


def _size_sigma(ratio: float, extents: FloatArray, extents_sigma: FloatArray) -> float:
    """First order: the ratio L_t / median(L) carries ``_length_rel_sigma`` twice (independent
    errors of both lengths)."""
    return ratio * math.sqrt(2.0) * _length_rel_sigma(extents, extents_sigma)


def _length_rel_sigma(extents: FloatArray, extents_sigma: FloatArray) -> float:
    """sigma_L / L = sqrt(sum (sigma_e / e)^2) / 3 for L = (e1 e2 e3)^(1/3)."""
    return math.sqrt(float(np.sum((extents_sigma / extents) ** 2))) / 3.0


def _relative_sigma(inputs: CheckInputs, index: int, row: int, ratio: float) -> float:
    """First order, as ``_size_sigma``, for r = L_i / ref: the relative sigma of the geometric
    mean of n references is sqrt(mean_j (sigma_Lj / L_j)^2 / n)."""
    states = inputs.states
    own = _length_rel_sigma(states.size[row, index], states.size_sigma[row, index])
    lengths = characteristic_lengths(states.size[row])
    refs = [
        _length_rel_sigma(states.size[row, j], states.size_sigma[row, j])
        for j in range(states.n_objects)
        if j != index and math.isfinite(float(lengths[j]))
    ]
    reference = math.sqrt(float(np.mean(np.square(refs))) / len(refs))
    return ratio * math.sqrt(2.0) * math.hypot(own, reference)


def _jump_sigma(window: FloatArray, position_sigma: FloatArray, length: float) -> float:
    """sqrt(20) sigma_p / (2 L) = sqrt(5) sigma_p / L with sigma_p the position sigma along the
    jump direction."""
    jump = third_difference(window)[0]
    norm = float(np.linalg.norm(jump))
    if norm > 0.0:
        along = float(np.linalg.norm(jump / norm * position_sigma))
    else:
        along = math.sqrt(float(np.mean(position_sigma**2)))
    return third_difference_sigma(along, 1.0) / (2.0 * length)


PLUGINS = (PermanenceCheck, SizeConstancyCheck)
