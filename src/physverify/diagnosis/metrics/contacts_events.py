"""CONTACTS: event matching by temporal IoU, then normal, gap and point errors per match."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.cases import (
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._states import object_length
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import hungarian
from physverify.schema.clip import ClipManifest
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

NOT_A_CANDIDATE = -1.0


def interval_iou(a: ContactRecord, b: ContactRecord) -> float:
    """tIoU of [t_start, t_end] in native frames, each padded to whole frames (+1), so that
    point-like impacts (t_start == t_end) still overlap."""
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start) + 1.0)
    union = (a.t_end - a.t_start + 1.0) + (b.t_end - b.t_start + 1.0) - inter
    return inter / union


def same_pair(a: ContactRecord, b: ContactRecord) -> bool:
    return {a.object_a, a.object_b} == {b.object_a, b.object_b}


def tiou_matrix(pred: tuple[ContactRecord, ...], truth: tuple[ContactRecord, ...]) -> FloatArray:
    """tIoU between candidates (same unordered pair); ``NOT_A_CANDIDATE`` otherwise."""
    out = np.full((len(pred), len(truth)), NOT_A_CANDIDATE)
    for i, p in enumerate(pred):
        for j, g in enumerate(truth):
            if same_pair(p, g):
                out[i, j] = interval_iou(p, g)
    return out


def truth_normal_as_pred(pred: ContactRecord, truth: ContactRecord) -> FloatArray:
    """The truth normal points from b to a; flipped when the pair order is inverted."""
    normal: FloatArray = truth.normal if pred.object_a == truth.object_a else -truth.normal
    return normal


def angle_deg(u: FloatArray, v: FloatArray) -> float:
    """Angle via atan2(|u x v|, u . v): exact near 0, where arccos loses half the digits."""
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def object_lengths(states: ObjectStateTable) -> dict[str, float]:
    return {
        str(object_id): object_length(states.size[:, n])
        for n, object_id in enumerate(states.object_ids.tolist())
    }


def pair_length(record: ContactRecord, lengths: Mapping[str, float]) -> float:
    """L_min = min(L_a, L_b); the environment has no length of its own."""
    ids = [i for i in (record.object_a, record.object_b) if i != ENV_ID]
    values = [lengths.get(i, math.nan) for i in ids]
    return min(values) if values and all(math.isfinite(v) for v in values) else math.nan


def geometry_errors(pred: ContactRecord, truth: ContactRecord, length: float) -> dict[str, float]:
    out = {"normal_err_deg": angle_deg(pred.normal, truth_normal_as_pred(pred, truth))}
    if not (math.isfinite(length) and length > 0):
        return out
    offset = pred.points.mean(axis=0) - truth.points.mean(axis=0)
    out["point_rel"] = float(np.linalg.norm(offset)) / length
    if pred.point_distances is not None and truth.point_distances is not None:
        gap = float(np.median(pred.point_distances) - np.median(truth.point_distances))
        out["gap_rel"] = gap / length
    return out


@dataclass(frozen=True)
class ContactEventMetric:
    """``normal_err_deg`` and the set rates need no context; ``gap_rel`` and ``point_rel`` need
    context LIFT (true sizes) and are NaN without it."""

    stage: ClassVar[Stage] = Stage.CONTACTS
    name: ClassVar[str] = "events"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, ContactTable) or not isinstance(truth, ContactTable):
            raise TypeError("contact metric compares two ContactTable")
        matches = hungarian(tiou_matrix(pred.records, truth.records), case.options.contact_min_tiou)
        states = stage_output(case.context, Stage.LIFT, ObjectStateTable)
        lengths = object_lengths(states) if states is not None else {}
        columns: dict[str, list[float]] = {"normal_err_deg": [], "gap_rel": [], "point_rel": []}
        same_kind = 0
        for i, j in matches:
            p, g = pred.records[i], truth.records[j]
            same_kind += int(p.kind is g.kind)
            for key, value in geometry_errors(p, g, pair_length(g, lengths)).items():
                columns[key].append(value)
        m, n_pred, n_truth = (
            float(len(matches)),
            float(len(pred.records)),
            float(len(truth.records)),
        )
        return MetricOutcome(
            samples=MappingProxyType(
                {k: np.array(v, dtype=np.float64) for k, v in columns.items()}
            ),
            ratios=MappingProxyType(
                {
                    "precision": (m, n_pred),
                    "recall": (m, n_truth),
                    "kind_agreement": (float(same_kind), m),
                    "contact_f1": (2.0 * m, n_pred + n_truth),
                }
            ),
        )


PLUGINS = (ContactEventMetric,)
