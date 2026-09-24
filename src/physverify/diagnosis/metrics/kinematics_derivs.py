"""KINEMATICS: velocity and acceleration errors, normalized by v_ref and by g in table units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._states import (
    StatePairs,
    finite_rows,
    object_length,
    paired_states,
    sigma_scores,
)
from physverify.diagnosis.normalizers import gravity_in_units, reference_speed
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.outputs import StageOutput
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


def reference_speeds(truth: ObjectStateTable, pairs: StatePairs, visible: BoolArray) -> FloatArray:
    """v_ref per paired object: max(RMS_t ||v_gt||, L_gt/T) over its visible frames."""
    velocity, sizes = pairs.truth(truth.velocity), pairs.truth(truth.size)
    duration = float(truth.t[-1] - truth.t[0])
    return np.array(
        [
            reference_speed(velocity[visible[:, n], n], object_length(sizes[:, n]), duration)
            for n in range(velocity.shape[1])
        ],
        dtype=np.float64,
    )


def impact_frames(
    contacts: ContactTable,
    times: FloatArray,
    object_ids: list[str],
    clip: ClipManifest,
    half: float,
) -> BoolArray:
    """(T, N): sample within ``half`` seconds of a ground-truth IMPACT involving the object.

    Contact bounds are fractional native frames; the tables' time base is frame * seconds."""
    near = np.zeros((times.size, len(object_ids)), dtype=bool)
    for record in contacts.records:
        if record.kind is not ContactKind.IMPACT:
            continue
        start = record.t_start * clip.seconds_per_frame - half
        end = record.t_end * clip.seconds_per_frame + half
        rows = (times >= start) & (times <= end)
        for n, object_id in enumerate(object_ids):
            if object_id in (record.object_a, record.object_b):
                near[:, n] |= rows
    return near


def velocity_samples(
    pred: ObjectStateTable, truth: ObjectStateTable, pairs: StatePairs, visible: BoolArray
) -> dict[str, FloatArray]:
    residual = pairs.pred(pred.velocity) - pairs.truth(truth.velocity)
    ok = visible & finite_rows(residual)
    v_ref = reference_speeds(truth, pairs, visible)
    usable = np.isfinite(v_ref) & (v_ref > 0)
    scaled = residual / np.where(usable, v_ref, np.nan)[None, :, None]
    return {
        "vel_rel": scaled[ok & usable[None, :]].ravel(),
        "vel_z": sigma_scores(residual, pairs.pred(pred.velocity_sigma), ok),
    }


def true_gravity(case: MetricCase) -> float:
    """g0/s_true in table units from context SCALE; NaN without it."""
    scale = stage_output(case.context, Stage.SCALE, ScaleEstimate)
    if scale is None or scale.meters_per_unit is None:
        return float("nan")
    return gravity_in_units(scale.meters_per_unit)


def impact_mask(case: MetricCase, truth: ObjectStateTable, pairs: StatePairs) -> BoolArray | None:
    contacts = stage_output(case.context, Stage.CONTACTS, ContactTable)
    if contacts is None:
        return None
    ids = [str(truth.object_ids[j]) for j in pairs.truth_cols]
    times = truth.t[pairs.truth_rows]
    return impact_frames(contacts, times, ids, case.clip, case.options.impact_half_window_s)


def acceleration_samples(
    residual: FloatArray, ok: BoolArray, g_units: float, near: BoolArray | None
) -> dict[str, FloatArray]:
    keys = ("acc_rel", "acc_rel_impact", "acc_rel_free")
    out: dict[str, FloatArray] = {key: np.zeros(0) for key in keys}
    if not np.isfinite(g_units):
        return out
    scaled = residual / g_units
    out["acc_rel"] = scaled[ok].ravel()
    if near is not None:
        out["acc_rel_impact"] = scaled[ok & near].ravel()
        out["acc_rel_free"] = scaled[ok & ~near].ravel()
    return out


@dataclass(frozen=True)
class DerivativeMetric:
    """``vel_rel`` and ``vel_z`` need no context. ``acc_rel`` needs context SCALE (true scale);
    ``acc_rel_impact``/``acc_rel_free`` also need context CONTACTS (true contacts). Without them
    those keys are NaN. ``acc_z`` needs no context."""

    stage: ClassVar[Stage] = Stage.KINEMATICS
    name: ClassVar[str] = "derivatives"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, ObjectStateTable) or not isinstance(truth, ObjectStateTable):
            raise TypeError("derivative metric compares two ObjectStateTable")
        pairs = paired_states(pred, truth)
        visible = pairs.visible(pred, truth)
        samples = velocity_samples(pred, truth, pairs, visible)
        a_res = pairs.pred(pred.acceleration) - pairs.truth(truth.acceleration)
        ok_a = visible & finite_rows(a_res)
        samples["acc_z"] = sigma_scores(a_res, pairs.pred(pred.acceleration_sigma), ok_a)
        near = impact_mask(case, truth, pairs)
        samples.update(acceleration_samples(a_res, ok_a, true_gravity(case), near))
        return MetricOutcome(samples=MappingProxyType(samples))


PLUGINS = (DerivativeMetric,)
