"""LIFT: error of a resting body's gap to its scene plane, in its length (what
``contacts/kinematic_scene`` gates environment rests with)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._state_pairing import (
    IntArray,
    StatePairs,
    object_length,
    paired_states,
)
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import conditional_scale_fit
from physverify.extraction.providers._proximity import YAW_AXIS
from physverify.physics.signed_distance import lowest_height
from physverify.physics.tracks import contact_rows, place_body
from physverify.schema.clip import ClipManifest
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.error_profile import SUPPORT_GAP_REL
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class LiftSupportGapMetric:
    """``support_gap_rel``: (s g_i - g*_i) / L*_i over the rows where a true RESTING record holds
    body i on the environment, g_i = lowest_i - min over j != i of H_j, H_j the median lowest
    height of body j over its own true environment rest rows, each table along its own up,
    after the conditional scale fit s. Needs context CONTACTS (true); empty without it."""

    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "support_gap"
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
        return support_gaps(pred, truth, contacts, fit.meters_per_unit)


def support_gaps(
    pred: ObjectStateTable, truth: ObjectStateTable, contacts: ContactTable, scale: float
) -> MetricOutcome:
    pairs = paired_states(pred, truth)
    rest = env_rest_rows(truth, pairs, contacts)
    held = float(np.count_nonzero(rest))
    lowest_pred = _lowest(pred, pairs.pred_rows, pairs.pred_cols)
    lowest_true = _lowest(truth, pairs.truth_rows, pairs.truth_cols)
    usable = rest & np.isfinite(lowest_pred) & np.isfinite(lowest_true)
    gap_pred, gap_true = _gaps(lowest_pred, usable), _gaps(lowest_true, usable)
    lengths = np.array(
        [object_length(pairs.truth(truth.size)[:, k]) for k in range(pairs.truth_cols.size)]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        error = (scale * gap_pred - gap_true) / lengths[None, :]
    found = usable & np.isfinite(error)
    values: FloatArray = error[found]
    return _outcome(values, (float(values.size), held))


def env_rest_rows(truth: ObjectStateTable, pairs: StatePairs, contacts: ContactTable) -> BoolArray:
    """(T, K) paired rows where a true RESTING record holds the paired object on the
    environment."""
    frames = truth.frame_index[pairs.truth_rows]
    ids = truth.object_ids[pairs.truth_cols].tolist()
    rest = np.zeros((frames.size, len(ids)), dtype=np.bool_)
    for record in contacts.records:
        env = record.kind is ContactKind.RESTING and record.object_b == ENV_ID
        if env and record.object_a in ids:
            rest[:, ids.index(record.object_a)] |= contact_rows(record, frames)
    return rest


def _lowest(states: ObjectStateTable, rows: IntArray, cols: IntArray) -> FloatArray:
    """(T, K) lowest height of each paired body along the table's up; NaN when unplaced."""
    gravity = np.asarray(states.gravity_dir, dtype=np.float64)
    out = np.full((rows.size, cols.size), math.nan)
    norm = float(np.linalg.norm(gravity))
    if not (bool(np.all(np.isfinite(gravity))) and norm > 0.0):
        return out
    up = -gravity / norm
    for r, row in enumerate(rows.tolist()):
        for k, col in enumerate(cols.tolist()):
            placed = place_body(states, row, col, np.asarray(YAW_AXIS))
            if placed is not None:
                out[r, k] = lowest_height(placed.solid, up)
    return out


def _gaps(lowest: FloatArray, usable: BoolArray) -> FloatArray:
    """lowest_i - min over j != i of H_j, H_j the median of body j over its usable rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        heights = np.nanmedian(np.where(usable, lowest, np.nan), axis=0)
    floors = np.array([_floor(heights, k) for k in range(heights.size)])
    gaps: FloatArray = lowest - floors[None, :]
    return gaps


def _floor(heights: FloatArray, k: int) -> float:
    others = np.delete(heights, k)
    others = others[np.isfinite(others)]
    return float(others.min()) if others.size else math.nan


def _outcome(values: FloatArray, coverage: tuple[float, float]) -> MetricOutcome:
    return MetricOutcome(
        samples=MappingProxyType({SUPPORT_GAP_REL: values}),
        ratios=MappingProxyType({"coverage": coverage}),
    )


PLUGINS = (LiftSupportGapMetric,)
