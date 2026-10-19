"""R4 object integrity per object: permanence (R4_missing), constant size (R4_size) and
spatio-temporal continuity (R4_jump). ADR 0006 section 3.4."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._common import CommonParams, OcclusionParams, abstain, first_max, unscored
from physverify.physics.finite_diff import third_difference, third_difference_sigma
from physverify.physics.geometry import characteristic_length, characteristic_lengths
from physverify.physics.measures.r4_integrity import jump_ratio, missing_frames, size_change_ratio
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
    pass


class SizeParams(OcclusionParams):
    min_size_frames: int = Field(default=3, ge=2)


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
        exact = bool(np.all(states.position_sigma[:, index] == 0.0))
        magnitude = n_rows * row_step(fi)
        details = {"missing_rows": n_rows}
        return unscored(
            type(self), inputs, subject, magnitude, 0.0 if exact else math.nan, span, details
        )


class SizeConstancyCheck:
    """R4_size [rel_size_change]: max_t |L_t / median(L) - 1| of the object's extents.

    Rows out of frame or with a known occlusion >= ``occluded_at`` are left out: their extents
    describe the visible part, not the object. Unknown (NaN) occlusion keeps the row, since
    perceived tables never carry occlusion; a truncated perceived mask (at the image border or
    behind another object) therefore still enters the measure. ABSTAIN only when fewer than
    ``min_size_frames`` rows with known extents remain.
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


class ContinuityCheck:
    """R4_jump [jump_over_size]: max ||p[k+2] - 3 p[k+1] + 3 p[k] - p[k-1]|| / (2 L) over
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
    """First order: sigma_L / L = sqrt(sum (sigma_e / e)^2) / 3 for L = (e1 e2 e3)^(1/3); the
    ratio L_t / median(L) carries it twice (independent errors of both lengths)."""
    rel_length = math.sqrt(float(np.sum((extents_sigma / extents) ** 2))) / 3.0
    return ratio * math.sqrt(2.0) * rel_length


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


PLUGINS = (PermanenceCheck, SizeConstancyCheck, ContinuityCheck)
