"""kinematics/derivatives: velocity and acceleration errors, split around impacts."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import pytest
from metric_cases import metric_case
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.kinematics_derivatives import DerivativeMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.constants import ENV_ID, STANDARD_GRAVITY
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.outputs import StageOutput
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.1,
)
FRAMES = 12
LENGTH = 0.1
S_TRUE = 0.5
G_UNITS = STANDARD_GRAVITY / S_TRUE
IMPACT_FRAME = 9.0
NEAR = np.arange(FRAMES) >= 7
"""Frames within 0.1 s of native frame 9 at 24 fps: t in [0.275, 0.475] -> frames 7..11."""
TOL = 1e-9


def vectors(value: tuple[float, float, float]) -> NDArray[np.float64]:
    return np.tile(np.array(value, dtype=np.float64), (FRAMES, 1, 1))


def truth(speed: float = 1.0) -> ObjectStateTable:
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.RELATIVE,
        source_key="gt:clip",
        t=np.arange(FRAMES, dtype=np.float64) * manifest().seconds_per_frame,
        frame_index=np.arange(FRAMES, dtype=np.int64),
        object_ids=["ball"],
        labels=["ball"],
        primitive=["sphere"],
    )
    return dataclasses.replace(
        blank,
        velocity=vectors((speed, 0.0, 0.0)),
        acceleration=vectors((0.0, 0.0, -G_UNITS)),
        size=vectors((LENGTH, LENGTH, LENGTH)),
    )


def predicted(
    velocity_bias: float, acc_error: NDArray[np.float64], sigma: float = math.nan
) -> ObjectStateTable:
    """``acc_error`` (T,) is added to the vertical acceleration of the truth."""
    base = truth()
    acceleration = base.acceleration.copy()
    acceleration[:, 0, 2] += acc_error
    return dataclasses.replace(
        base,
        velocity=base.velocity + np.array([velocity_bias, 0.0, 0.0]),
        acceleration=acceleration,
        velocity_sigma=np.full((FRAMES, 1, 3), sigma),
        acceleration_sigma=np.full((FRAMES, 1, 3), sigma),
    )


def scale() -> ScaleEstimate:
    return ScaleEstimate(
        meters_per_unit=S_TRUE, sigma=0.0, source=ScaleSource.GROUND_TRUTH, lift_key="gt:clip"
    )


def contacts(kind: ContactKind) -> ContactTable:
    record = ContactRecord(
        object_a="ball",
        object_b=ENV_ID,
        t_start=IMPACT_FRAME,
        t_end=IMPACT_FRAME,
        kind=kind,
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.zeros((1, 3)),
        manifold=ManifoldKind.POINT,
    )
    return ContactTable(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.RELATIVE,
        source_key="gt:clip",
        records=(record,),
    )


def measure(
    pred: ObjectStateTable,
    truth_table: ObjectStateTable,
    context: Mapping[Stage, StageOutput] | None = None,
) -> MetricOutcome:
    case = metric_case(pred, truth_table, options=OPTIONS, context=context)
    return DerivativeMetric(OPTIONS).samples(case)


def test_velocity_bias_in_reference_speeds() -> None:
    bias = 0.2
    outcome = measure(predicted(bias, np.zeros(FRAMES), sigma=0.5), truth())
    vel = outcome.samples["vel_rel"].reshape(FRAMES, 3)
    np.testing.assert_allclose(vel[:, 0], bias, atol=TOL)
    np.testing.assert_allclose(vel[:, 1:], 0.0, atol=TOL)
    np.testing.assert_allclose(outcome.samples["vel_z"].reshape(FRAMES, 3)[:, 0], bias / 0.5)


def test_slow_objects_are_normalized_by_length_over_duration() -> None:
    bias, speed = 0.05, 0.01
    slow = truth(speed)
    pred = dataclasses.replace(slow, velocity=slow.velocity + np.array([bias, 0.0, 0.0]))
    v_ref = LENGTH / float(slow.t[-1] - slow.t[0])
    vel = measure(pred, slow).samples["vel_rel"].reshape(FRAMES, 3)
    np.testing.assert_allclose(vel[:, 0], bias / v_ref, atol=TOL)


def test_acceleration_error_in_units_of_g_needs_the_true_scale() -> None:
    error = 0.5 * G_UNITS
    pred = predicted(0.0, np.full(FRAMES, error), sigma=2.0)
    outcome = measure(pred, truth(), {Stage.SCALE: scale()})
    acc = outcome.samples["acc_rel"].reshape(FRAMES, 3)
    np.testing.assert_allclose(acc[:, 2], 0.5, atol=TOL)
    np.testing.assert_allclose(acc[:, :2], 0.0, atol=TOL)
    np.testing.assert_allclose(outcome.samples["acc_z"].reshape(FRAMES, 3)[:, 2], error / 2.0)
    assert outcome.samples["acc_rel_impact"].size == 0


def test_impact_frames_are_split_from_free_flight() -> None:
    error = np.where(NEAR, 2.0 * G_UNITS, 0.1 * G_UNITS)
    context: dict[Stage, StageOutput] = {
        Stage.SCALE: scale(),
        Stage.CONTACTS: contacts(ContactKind.IMPACT),
    }
    outcome = measure(predicted(0.0, error), truth(), context)
    impact = outcome.samples["acc_rel_impact"].reshape(-1, 3)
    free = outcome.samples["acc_rel_free"].reshape(-1, 3)
    assert impact.shape[0] == int(NEAR.sum())
    assert free.shape[0] == FRAMES - int(NEAR.sum())
    np.testing.assert_allclose(impact[:, 2], 2.0, atol=TOL)
    np.testing.assert_allclose(free[:, 2], 0.1, atol=TOL)


def test_resting_contacts_are_not_impacts() -> None:
    context: dict[Stage, StageOutput] = {
        Stage.SCALE: scale(),
        Stage.CONTACTS: contacts(ContactKind.RESTING),
    }
    outcome = measure(predicted(0.0, np.zeros(FRAMES)), truth(), context)
    assert outcome.samples["acc_rel_impact"].size == 0
    assert outcome.samples["acc_rel_free"].size == 3 * FRAMES


def test_without_context_acceleration_is_nan() -> None:
    summary = DerivativeMetric(OPTIONS).compute(
        predicted(0.1, np.ones(FRAMES)), truth(), manifest()
    )
    assert math.isnan(summary["acc_rel.mean"])
    assert math.isnan(summary["acc_rel_impact.mean"])
    assert summary["vel_rel.n"] == 3.0 * FRAMES
    assert summary["vel_z.n"] == 0.0


def test_wrong_output_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        measure(scale(), truth())  # type: ignore[arg-type]
