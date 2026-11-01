"""LIFT: error of the horizontal offset between a body and the body it rests on, in the upper
body's length (what R2 footprint compares against its supports' outline)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.checks.variants.r2_footprint import SUPPORT_MAX_TILT_DEG
from physverify.diagnosis.measurement_cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._state_pairing import StatePairs, object_length, paired_states
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import conditional_scale_fit
from physverify.physics.footprint import support_order
from physverify.physics.geometry import project_to_plane
from physverify.physics.tracks import contact_rows
from physverify.schema.clip import ClipManifest
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.error_profile import SUPPORT_OFFSET_REL
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class LiftSupportOffsetMetric:
    """``support_offset_rel``: per horizontal axis, (s (p_i - p_j) - (p*_i - p*_j)) / L*_i over
    the rows where a true RESTING record holds body i on body j (normal within the R2 footprint
    tilt), after the conditional scale fit s. Needs context CONTACTS (true); empty without it."""

    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "support_offset"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, ObjectStateTable) or not isinstance(truth, ObjectStateTable):
            raise TypeError("lift metric compares two ObjectStateTable")
        contacts = stage_output(case.context, Stage.CONTACTS, ContactTable)
        fit = conditional_scale_fit(pred, truth)
        if contacts is None or fit is None:
            return _outcome(np.zeros(0), (0.0, 0.0))
        return support_offsets(pred, truth, contacts, fit.meters_per_unit)


def support_offsets(
    pred: ObjectStateTable, truth: ObjectStateTable, contacts: ContactTable, scale: float
) -> MetricOutcome:
    pairs = paired_states(pred, truth)
    up = -np.asarray(truth.gravity_dir, dtype=np.float64)
    truth_ids = truth.object_ids[pairs.truth_cols].tolist()
    found: list[FloatArray] = []
    held, measured = 0, 0
    for record in contacts.records:
        if record.kind is not ContactKind.RESTING or record.object_b == ENV_ID:
            continue
        order = support_order(
            record.object_a, record.object_b, record.normal, up, SUPPORT_MAX_TILT_DEG
        )
        if order is None or not set(order) <= set(truth_ids):
            continue
        upper, lower = (truth_ids.index(name) for name in order)
        rows = contact_rows(record, truth.frame_index[pairs.truth_rows])
        held += int(np.count_nonzero(rows))
        offsets = _offsets(pred, truth, pairs, (upper, lower), rows, scale, up)
        measured += offsets.shape[0]
        found.append(offsets.ravel())
    values = np.concatenate(found) if found else np.zeros(0)
    return _outcome(values, (float(measured), float(held)))


def _offsets(
    pred: ObjectStateTable,
    truth: ObjectStateTable,
    pairs: StatePairs,
    bodies: tuple[int, int],
    rows: BoolArray,
    scale: float,
    up: FloatArray,
) -> FloatArray:
    """(n, 2) horizontal offset errors over the upper body's true length, on usable rows."""
    upper, lower = bodies
    p, g = pairs.pred(pred.position), pairs.truth(truth.position)
    visible = pairs.visible(pred, truth)
    error = scale * (p[:, upper] - p[:, lower]) - (g[:, upper] - g[:, lower])
    usable = rows & visible[:, upper] & visible[:, lower] & np.isfinite(error).all(axis=-1)
    length = object_length(pairs.truth(truth.size)[:, upper])
    if not usable.any() or not np.isfinite(length) or length <= 0:
        return np.zeros((0, 2))
    planar: FloatArray = project_to_plane(error[usable], up) / length
    return planar


def _outcome(values: FloatArray, coverage: tuple[float, float]) -> MetricOutcome:
    return MetricOutcome(
        samples=MappingProxyType({SUPPORT_OFFSET_REL: values}),
        ratios=MappingProxyType({"coverage": coverage}),
    )


PLUGINS = (LiftSupportOffsetMetric,)
