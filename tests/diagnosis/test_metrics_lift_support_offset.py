"""lift/support_offset: horizontal offset error between a body and its support, in its length."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case
from numpy.typing import NDArray

from physverify.diagnosis.metrics.lift_support_offset import LiftSupportOffsetMetric
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.error_profile import SUPPORT_OFFSET_REL
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

FRAMES = 6
S0 = 2.5
TOP = 0.2
UP = (0.0, 0.0, 1.0)
CENTRES = np.array([[0.0, 0.0, 5.1], [0.05, 0.0, 5.3]])
SIZES = np.array([[0.4, 0.4, 0.2], [TOP, TOP, TOP]])


def table(unit: LengthUnit, position: NDArray[np.float64]) -> ObjectStateTable:
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=unit,
        source_key="gt:clip" if unit is LengthUnit.METER else "lift:clip",
        t=np.arange(FRAMES) / 24.0,
        frame_index=np.arange(FRAMES, dtype=np.int64),
        object_ids=["base", "top"],
        labels=["box", "box"],
        primitive=["box", "box"],
    )
    return dataclasses.replace(
        blank,
        position=np.broadcast_to(position, (FRAMES, 2, 3)).copy(),
        size=np.broadcast_to(SIZES, (FRAMES, 2, 3)).copy(),
        gravity_dir=np.array([0.0, 0.0, -1.0]),
    )


def resting(
    a: str = "top", b: str = "base", normal: tuple[float, float, float] = UP
) -> ContactRecord:
    return ContactRecord(
        object_a=a,
        object_b=b,
        t_start=0.0,
        t_end=float(FRAMES - 1),
        kind=ContactKind.RESTING,
        normal=np.array(normal, dtype=np.float64),
        points=np.zeros((1, 3)),
        manifold=ManifoldKind.POINT,
    )


def contacts(*records: ContactRecord) -> ContactTable:
    return ContactTable(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key="gt:clip",
        records=records,
    )


def measure(shift: tuple[float, float, float], *records: ContactRecord) -> NDArray[np.float64]:
    pred = table(LengthUnit.RELATIVE, (CENTRES + np.array([[0.0] * 3, shift])) / S0)
    truth = table(LengthUnit.METER, CENTRES)
    context = {Stage.CONTACTS: contacts(*records)} if records else None
    outcome = LiftSupportOffsetMetric(OPTIONS).samples(metric_case(pred, truth, context=context))
    values: NDArray[np.float64] = outcome.samples[SUPPORT_OFFSET_REL]
    return values


def test_an_exact_lift_has_no_offset_error() -> None:
    values = measure((0.0, 0.0, 0.0), resting())
    assert values.shape == (2 * FRAMES,)
    np.testing.assert_allclose(values, 0.0, atol=1e-12)


@pytest.mark.parametrize("record", [resting(), resting("base", "top", (0.0, 0.0, -1.0))])
def test_a_horizontal_shift_is_its_length_fraction_either_record_order(
    record: ContactRecord,
) -> None:
    values = measure((0.0, 0.01, 0.0), record).reshape(FRAMES, 2)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 0.01 / TOP, rtol=1e-3)


def test_a_vertical_shift_is_no_offset() -> None:
    values = measure((0.0, 0.0, 0.002), resting())
    np.testing.assert_allclose(values, 0.0, atol=1e-3)


def test_floor_and_steep_contacts_are_not_supports() -> None:
    steep = (math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0)))
    assert measure((0.0, 0.01, 0.0), resting("top", ENV_ID), resting(normal=steep)).size == 0


def test_without_true_contacts_there_is_no_sample() -> None:
    assert measure((0.0, 0.01, 0.0)).size == 0
