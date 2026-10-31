"""R4_missing object permanence: unexplained missing time per object."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import OcclusionParams
from physverify.physics.measures.object_integrity import missing_frames
from physverify.physics.row_time import seconds_per_row
from physverify.physics.tracks import FloatArray, native_span, row_step
from physverify.schema.checks import AbstainReason, CheckResult


class PermanenceParams(OcclusionParams):
    # "unknown": sigma NaN unless the positions are exact (the original rule). "edges": the
    # estimated counting sigma of ``count_sigma``, whose level the per-check sigma recalibration
    # sets.
    sigma_model: Literal["unknown", "edges"] = "edges"


class PermanenceCheck:
    """R4_missing [seconds]: the real time during which the object is inside the image but unseen
    and not explained by occlusion, missing rows times the seconds per row (``row_time``), so a
    clip at another frame rate or row step reads on the same scale; at the declared frame rate,
    which for PISA is an estimate. The native frames it stands for are in ``details``
    (``missing_frames``), the unit the perturbations are injected in."""

    check_id: ClassVar[str] = "R4_missing"
    requires: ClassVar[frozenset[str]] = frozenset()
    time_dependent: ClassVar[bool] = True
    magnitude_unit: ClassVar[str] = "seconds"
    perturbation_unit: ClassVar[str] = "frames"
    perturbation_estimate: ClassVar[str] = "missing_frames"
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
        row_seconds = seconds_per_row(fi, inputs.clip.seconds_per_frame)
        sigma, estimated = self._sigma(states.position_sigma[:, index], rows, row_seconds)
        details = {
            "missing_rows": n_rows,
            "missing_frames": n_rows * row_step(fi),
            **({"sigma_estimated": True} if estimated else {}),
        }
        return unscored(type(self), inputs, subject, n_rows * row_seconds, sigma, span, details)

    def _sigma(
        self, position_sigma: FloatArray, rows: list[int], row_seconds: float
    ) -> tuple[float, bool]:
        """(sigma, estimated): 0 for exact positions, else the edge model or, before, unknown."""
        if bool(np.all(position_sigma == 0.0)):
            return 0.0, False
        if self.params.sigma_model == "edges":
            return count_sigma(rows, row_seconds), True
        return math.nan, False


def count_sigma(rows: list[int], row_seconds: float) -> float:
    """Estimated sigma of a missing time: row_seconds * sqrt(2 * max(runs, 1)).

    Each run of counted rows has two edges whose row is uncertain by one sampling step (a mask
    that fades under ``min_area_px`` a row early or late). A shape only: the per-check sigma
    recalibration sets its level."""
    runs = sum(1 for k, row in enumerate(rows) if k == 0 or row != rows[k - 1] + 1)
    return row_seconds * math.sqrt(2.0 * max(runs, 1))


PLUGINS = (PermanenceCheck,)
