"""lift/support_gap: error of a resting body's gap to the lowest other resting body, in its
length."""

from __future__ import annotations

import dataclasses

import numpy as np
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case
from numpy.typing import NDArray

from physverify.diagnosis.metrics.lift_support_gap import LiftSupportGapMetric
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.error_profile import SUPPORT_GAP_REL
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

FRAMES = 6
S0 = 2.5
SIDE = 0.2
IDS = ["a", "b", "c"]
CENTRES = np.array([[0.0, 0.0, SIDE / 2], [1.0, 0.0, SIDE / 2], [0.0, 0.0, 1.5 * SIDE]])
UP = (0.0, 0.0, 1.0)


def table(unit: LengthUnit, position: NDArray[np.float64]) -> ObjectStateTable:
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=unit,
        source_key="gt:clip" if unit is LengthUnit.METER else "lift:clip",
        t=np.arange(FRAMES) / 24.0,
        frame_index=np.arange(FRAMES, dtype=np.int64),
        object_ids=IDS,
        labels=["box"] * 3,
        primitive=["box"] * 3,
    )
    size = SIDE if unit is LengthUnit.METER else SIDE / S0
    return dataclasses.replace(
        blank,
        position=np.broadcast_to(position, (FRAMES, 3, 3)).copy(),
        size=np.full((FRAMES, 3, 3), size),
        gravity_dir=np.array([0.0, 0.0, -1.0]),
    )


def resting(a: str, b: str = ENV_ID) -> ContactRecord:
    return ContactRecord(
        object_a=a,
        object_b=b,
        t_start=0.0,
        t_end=float(FRAMES - 1),
        kind=ContactKind.RESTING,
        normal=np.array(UP),
        points=np.zeros((1, 3)),
        manifold=ManifoldKind.POINT,
    )


def measure(shift: NDArray[np.float64], *records: ContactRecord) -> NDArray[np.float64]:
    pred = table(LengthUnit.RELATIVE, (CENTRES + shift) / S0)
    truth = table(LengthUnit.METER, CENTRES)
    context = None
    if records:
        contacts = ContactTable(
            clip_id="clip",
            frame=CoordFrame.WORLD_ZUP,
            unit=LengthUnit.METER,
            source_key="gt:clip",
            records=records,
        )
        context = {Stage.CONTACTS: contacts}
    outcome = LiftSupportGapMetric(OPTIONS).samples(metric_case(pred, truth, context=context))
    values: NDArray[np.float64] = outcome.samples[SUPPORT_GAP_REL]
    return values


FLOOR = (resting("a"), resting("b"), resting("c", "a"))


def test_an_exact_lift_has_no_gap_error_and_only_floor_rests_are_sampled() -> None:
    values = measure(np.zeros((3, 3)), *FLOOR)
    assert values.shape == (2 * FRAMES,)
    np.testing.assert_allclose(values, 0.0, atol=1e-9)


def test_a_lifted_floor_body_is_its_length_fraction_and_moves_the_other_one() -> None:
    lift = 0.01
    values = measure(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, lift], [0.0] * 3]), *FLOOR)
    expected = [[-lift / SIDE, lift / SIDE]] * FRAMES
    np.testing.assert_allclose(values.reshape(FRAMES, 2), expected, rtol=1e-2)


def test_a_horizontal_shift_is_no_gap_error() -> None:
    values = measure(np.array([[0.03, 0.02, 0.0], [0.0] * 3, [0.0] * 3]), *FLOOR)
    np.testing.assert_allclose(values, 0.0, atol=1e-9)


def test_a_single_floor_body_has_no_plane_and_no_sample() -> None:
    assert measure(np.zeros((3, 3)), resting("a"), resting("c", "a")).size == 0


def test_without_true_contacts_there_is_no_sample() -> None:
    assert measure(np.zeros((3, 3))).size == 0
