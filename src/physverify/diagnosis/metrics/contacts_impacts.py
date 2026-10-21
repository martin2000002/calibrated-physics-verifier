"""CONTACTS: impacts alone, matched one to one by time (it3 change C; not in the profile).

``contacts/events`` matches every record by temporal IoU, so one long predicted rest against a
true rest cut into pieces counts once and impacts drown among rests. Here only IMPACT records
take part: a predicted and a true impact of the same pair match when the gap between their
frame intervals is at most MAX_GAP_FRAMES native frames (Hungarian assignment on the gap).
Recall and precision come overall, for the environment and for object pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.cases import (
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
)
from physverify.diagnosis.metrics.contacts_events import NOT_A_CANDIDATE, same_pair
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import hungarian
from physverify.schema.clip import ClipManifest
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage

MAX_GAP_FRAMES = 1.0
"""A predicted impact within one native frame of a true one, as in the AC-4 of plan-it3."""


def frame_gap(a: ContactRecord, b: ContactRecord) -> float:
    """Native frames between two closed intervals (0 when they overlap)."""
    return max(0.0, max(a.t_start, b.t_start) - min(a.t_end, b.t_end))


def midpoint(record: ContactRecord) -> float:
    return 0.5 * (record.t_start + record.t_end)


def impacts(table: ContactTable) -> tuple[ContactRecord, ...]:
    return tuple(r for r in table.records if r.kind is ContactKind.IMPACT)


def closeness(pred: Sequence[ContactRecord], truth: Sequence[ContactRecord]) -> FloatArray:
    """MAX_GAP_FRAMES - gap for impacts of the same pair, NOT_A_CANDIDATE otherwise."""
    out = np.full((len(pred), len(truth)), NOT_A_CANDIDATE)
    for i, p in enumerate(pred):
        for j, g in enumerate(truth):
            if same_pair(p, g):
                out[i, j] = MAX_GAP_FRAMES - frame_gap(p, g)
    return out


def with_env(records: Sequence[ContactRecord], env: bool) -> int:
    return sum((ENV_ID in (r.object_a, r.object_b)) == env for r in records)


@dataclass(frozen=True)
class ContactImpactMetric:
    stage: ClassVar[Stage] = Stage.CONTACTS
    name: ClassVar[str] = "impacts"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> dict[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        if not isinstance(case.pred, ContactTable) or not isinstance(case.truth, ContactTable):
            raise TypeError("contact metric compares two ContactTable")
        pred, truth = impacts(case.pred), impacts(case.truth)
        matches = hungarian(closeness(pred, truth), 0.0)
        matched = [truth[j] for _, j in matches]
        errors = [midpoint(pred[i]) - midpoint(truth[j]) for i, j in matches]
        ratios: dict[str, tuple[float, float]] = {
            "recall": (float(len(matches)), float(len(truth))),
            "precision": (float(len(matches)), float(len(pred))),
        }
        for label, env in (("env", True), ("pair", False)):
            hits = float(with_env(matched, env))
            ratios[f"recall_{label}"] = (hits, float(with_env(truth, env)))
            ratios[f"precision_{label}"] = (hits, float(with_env(pred, env)))
        return MetricOutcome(
            samples=MappingProxyType({"t_err_frames": np.array(errors, dtype=np.float64)}),
            ratios=MappingProxyType(ratios),
        )


PLUGINS = (ContactImpactMetric,)
