"""The first R4_jump statistic, ``jump_over_size``, kept out of the registry so the tests and
goldens of the v1 arm can still run it."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.checks._base import CheckInputs
from physverify.checks._common import CommonParams, abstain, first_max, unscored
from physverify.physics.finite_diff import third_difference, third_difference_sigma
from physverify.physics.geometry import characteristic_length
from physverify.physics.measures.r4_integrity import jump_ratio
from physverify.physics.tracks import impact_stencil_mask, median_extents, native_span
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.contacts import ContactTable

FloatArray = NDArray[np.float64]


class ContinuityCheck:
    """R4_jump v1, replaced in the registry by ``r4_jump_noise.NoiseContinuityCheck``.

    R4_jump [jump_over_size]: max ||p[k+2] - 3 p[k+1] + 3 p[k] - p[k-1]|| / (2 L) over
    consecutive rows, skipping the rows whose stencil spans an impact (the smooth
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
