"""R1 learned: R1 on object pairs, each re-scored by a trained artifact from the pair's track
features and the image features of its contact zone (``_learned``); pairs with the ground are
left out. The magnitude is an estimate in R1's unit, with the artifact's validation residual
rms as sigma."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.checks._base import CheckInputs
from physverify.checks._params import as_params
from physverify.checks.r1_interpenetration import InterpenetrationCheck, InterpenetrationParams
from physverify.checks.variants._learned import LearnedFields, LearnedMixin, SubjectFeatures
from physverify.physics.tracks import body_length
from physverify.schema.checks import CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.states import ObjectStateTable

TRACK_COLUMNS = (
    "base_magnitude",
    "base_sigma",
    "base_abstained",
    "gap_over_size",
    "occlusion_a",
    "occlusion_b",
    "size_ratio",
    "orientation_assumed",
)


class LearnedInterpenetrationParams(InterpenetrationParams, LearnedFields):
    pass


class LearnedInterpenetrationCheck(LearnedMixin, InterpenetrationCheck):
    variant: ClassVar[str] = "learned"
    version: ClassVar[str] = "2"
    Params: ClassVar[type[BaseModel]] = LearnedInterpenetrationParams
    TRACK_COLUMNS: ClassVar[tuple[str, ...]] = TRACK_COLUMNS

    def __init__(self, params: BaseModel) -> None:
        super().__init__(params)
        self.learned_params = as_params(LearnedInterpenetrationParams, params)

    def subjects(self, inputs: CheckInputs) -> list[SubjectFeatures]:
        return [
            SubjectFeatures(
                base, pair_track(inputs.states, base), [(base.subject[0], base.subject[1])]
            )
            for base in InterpenetrationCheck.evaluate(self, inputs)
            if base.subject[1] != ENV_ID
        ]


def pair_track(states: ObjectStateTable, base: CheckResult) -> np.ndarray:
    """Values in ``TRACK_COLUMNS`` order; NaN where the default result or the track has none."""
    a, b = (states.object_index(o) for o in base.subject)
    lengths = [body_length(states, a), body_length(states, b)]
    distance = base.details.get("distance")
    shorter = min(lengths)
    gap = math.nan if distance is None or not math.isfinite(shorter) else distance / shorter
    with np.errstate(all="ignore"):
        occlusion = [float(np.nanmean(states.occlusion[:, i])) for i in (a, b)]
        ratio = shorter / max(lengths)
    abstained = base.magnitude is None
    return np.array(
        [
            math.nan if abstained else base.magnitude,
            base.sigma,
            float(abstained),
            gap,
            *occlusion,
            ratio,
            float(bool(base.details.get("orientation_assumed", False))),
        ],
        dtype=np.float64,
    )


PLUGINS = (LearnedInterpenetrationCheck,)
