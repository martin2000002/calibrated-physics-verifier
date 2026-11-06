"""R2 learned: R2's resting groups, each re-scored by a trained artifact from the group's track
features and the image features of the contact zones of its pairs (``_learned``). The magnitude
is an estimated overhang over size (the unit R2's injected edit is measured in), with the
artifact's validation residual rms as sigma."""

from __future__ import annotations

import math
from itertools import combinations
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.checks._base import CheckInputs
from physverify.checks._params import as_params
from physverify.checks.r2_static_equilibrium import EquilibriumCheck, EquilibriumParams
from physverify.checks.variants._learned import LearnedFields, LearnedMixin, SubjectFeatures
from physverify.schema.checks import CheckResult
from physverify.schema.states import ObjectStateTable

TRACK_COLUMNS = (
    "base_magnitude",
    "base_sigma",
    "base_abstained",
    "max_overhang",
    "bodies",
    "excluded_contacts",
    "occlusion_max",
)


class LearnedEquilibriumParams(EquilibriumParams, LearnedFields):
    pass


class LearnedEquilibriumCheck(LearnedMixin, EquilibriumCheck):
    variant: ClassVar[str] = "learned"
    version: ClassVar[str] = "2"
    magnitude_unit: ClassVar[str] = "overhang_over_size"
    replaced_unit: ClassVar[str] = EquilibriumCheck.magnitude_unit
    Params: ClassVar[type[BaseModel]] = LearnedEquilibriumParams
    TRACK_COLUMNS: ClassVar[tuple[str, ...]] = TRACK_COLUMNS

    def __init__(self, params: BaseModel) -> None:
        super().__init__(params)
        self.learned_params = as_params(LearnedEquilibriumParams, params)

    def subjects(self, inputs: CheckInputs) -> list[SubjectFeatures]:
        return [
            SubjectFeatures(
                base, group_track(inputs.states, base), list(combinations(base.subject, 2))
            )
            for base in EquilibriumCheck.evaluate(self, inputs)
        ]


def group_track(states: ObjectStateTable, base: CheckResult) -> np.ndarray:
    """Values in ``TRACK_COLUMNS`` order; NaN where the default result or the track has none."""
    known = set(states.object_ids.astype(str).tolist())
    bodies = [states.object_index(o) for o in base.subject if o in known]
    overhangs = [v for v in (base.details.get("overhang") or {}).values() if v is not None]
    with np.errstate(all="ignore"):
        occlusion = float(np.nanmax(states.occlusion[:, bodies])) if bodies else math.nan
    abstained = base.magnitude is None
    return np.array(
        [
            math.nan if abstained else base.magnitude,
            base.sigma,
            float(abstained),
            max(overhangs) if overhangs else math.nan,
            float(len(base.subject)),
            float(len(base.details.get("excluded_contacts", []))),
            occlusion,
        ],
        dtype=np.float64,
    )


PLUGINS = (LearnedEquilibriumCheck,)
