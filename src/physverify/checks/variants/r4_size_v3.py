"""R4_size v3: a persistent change of the size relative to the other bodies, against its running
median, with time windows in seconds and rows at the image border left out."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import OcclusionParams
from physverify.checks._size import relative_sigma, whole_lengths, whole_rows
from physverify.physics.measures.detrended_size import persistent_change
from physverify.physics.measures.relative_size import relative_lengths
from physverify.physics.row_time import odd_rows_in, reference_seconds, rows_in
from physverify.physics.tracks import native_span
from physverify.schema.checks import AbstainReason, CheckResult

FloatArray = NDArray[np.float64]

WINDOW_S = reference_seconds(17)
"""The 17-row window of the 24 fps synthetic clips: wider than twice the longest r4_resize bump
(8 rows), so the running median does not follow a bump. Set on simulator data (Kubric dev)."""
PERSISTENCE_S = reference_seconds(3)
"""The shortest r4_resize bump (3 rows at 24 fps). Set on simulator data (Kubric dev)."""


class DetrendedSizeParams(OcclusionParams):
    """Windows in seconds, turned into rows with the clip's real frame rate and row step."""

    reference: Literal["others"] = "others"
    window_s: float = Field(default=WINDOW_S, gt=0.0)
    persistence_s: float = Field(default=PERSISTENCE_S, gt=0.0)


class DetrendedSizeCheck:
    """R4_size v3 [rel_size_change]: m = max_t min_{s in [t, t+P)} |r_s / runmed_W(r)_s - 1|
    over runs of P rows of one sign, with r the size relative to the other bodies of the row
    (``relative_lengths``, as v2), W = ``window_s`` and P = ``persistence_s`` in rows.

    Rows are left out for the object and as a reference when out of frame, known occluded
    (>= ``occluded_at``) or cut by the image border (``border_cut``, flagged by the lift from the
    mask). Perceived tables carry no occlusion and their visibility cannot see an untracked
    occluder, so a mask cut by such an object still enters; only the border cut is caught.
    ABSTAIN (TOO_FEW_FRAMES) without any run of P known rows; a clip with one object has no
    reference and abstains likewise. The windows are seconds of the declared frame rate, so an
    fps that is declared and not measured (PISA) shifts W and P by its own error."""

    check_id: ClassVar[str] = "R4_size"
    variant: ClassVar[str] = "v3"
    version: ClassVar[str] = "1"
    requires: ClassVar[frozenset[str]] = frozenset()
    time_dependent: ClassVar[bool] = False
    magnitude_unit: ClassVar[str] = "rel_size_change"
    Params: ClassVar[type[BaseModel]] = DetrendedSizeParams

    def __init__(self, params: BaseModel) -> None:
        self.params = DetrendedSizeParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        states = inputs.states
        whole = whole_rows(states.in_frame, states.occlusion, self.params.occluded_at)
        if states.border_cut is not None:
            whole &= ~states.border_cut
        lengths = whole_lengths(states, whole)
        row_seconds = inputs.row_seconds
        window = odd_rows_in(self.params.window_s, row_seconds)
        persistence = rows_in(self.params.persistence_s, row_seconds)
        return [
            self._object(inputs, lengths, index, window, persistence)
            for index in range(states.n_objects)
        ]

    def _object(
        self, inputs: CheckInputs, lengths: FloatArray, index: int, window: int, persistence: int
    ) -> CheckResult:
        states = inputs.states
        subject = (str(states.object_ids[index]),)
        own, others = lengths[:, index], np.delete(lengths, index, axis=1)
        ratios = relative_lengths(own, others)
        details: dict[str, object] = {
            "window_rows": window,
            "persistence_rows": persistence,
            "size_rows": int(np.count_nonzero(np.isfinite(ratios))),
            "excluded_rows": int(np.count_nonzero(~np.isfinite(own))),
        }
        change = persistent_change(ratios, window, persistence)
        if change is None:
            return abstain(type(self), inputs, subject, AbstainReason.TOO_FEW_FRAMES, None, details)
        ratio = 1.0 + change.deviation
        sigma = relative_sigma(states, lengths, index, change.row, ratio)
        last = change.first_row + persistence - 1
        span = native_span(states.frame_index, change.first_row, last)
        details |= {"row": change.row, "deviation": change.deviation}
        return unscored(type(self), inputs, subject, change.magnitude, sigma, span, details)


PLUGINS = (DetrendedSizeCheck,)
