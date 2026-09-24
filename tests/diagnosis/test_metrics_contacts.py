"""contacts/events: tIoU matching, normal, point and gap errors (ADR 0007 §7)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics.contacts_events import ContactEventMetric, interval_iou
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
UP = (0.0, 0.0, 1.0)
BALL_LENGTH, BOX_LENGTH = 0.2, 0.4
TOL = 1e-9


def record(
    a: str = "ball",
    b: str = "box",
    span: tuple[float, float] = (4.0, 4.0),
    kind: ContactKind = ContactKind.IMPACT,
    normal: tuple[float, float, float] = UP,
    point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance: float | None = None,
) -> ContactRecord:
    return ContactRecord(
        object_a=a,
        object_b=b,
        t_start=span[0],
        t_end=span[1],
        kind=kind,
        normal=np.array(normal, dtype=np.float64),
        points=np.array([point], dtype=np.float64),
        manifold=ManifoldKind.POINT,
        point_distances=None if distance is None else np.array([distance]),
    )


def table(*records: ContactRecord) -> ContactTable:
    return ContactTable(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key="gt:clip",
        records=records,
    )


def sizes() -> ObjectStateTable:
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key="gt:clip",
        t=np.arange(3) / 24.0,
        frame_index=np.arange(3, dtype=np.int64),
        object_ids=["ball", "box"],
        labels=["ball", "box"],
        primitive=["sphere", "box"],
    )
    size: NDArray[np.float64] = np.empty((3, 2, 3))
    size[:, 0] = BALL_LENGTH
    size[:, 1] = BOX_LENGTH
    return dataclasses.replace(blank, size=size)


def measure(
    pred: ContactTable, truth: ContactTable, context: Mapping[Stage, StageOutput] | None = None
) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=truth,
        context=MappingProxyType(dict(context or {})),
        gt=None,
        options=OPTIONS,
        plan=None,
    )
    return ContactEventMetric(OPTIONS).samples(case)


def test_identical_events_match_with_no_error() -> None:
    truth = table(
        record(span=(3.0, 3.0)), record(b=ENV_ID, span=(8.0, 11.0), kind=ContactKind.RESTING)
    )
    outcome = measure(truth, truth)
    np.testing.assert_allclose(outcome.samples["normal_err_deg"], [0.0, 0.0], atol=TOL)
    assert outcome.ratios["precision"] == (2.0, 2.0)
    assert outcome.ratios["recall"] == (2.0, 2.0)
    assert outcome.ratios["kind_agreement"] == (2.0, 2.0)
    assert outcome.ratios["contact_f1"] == (4.0, 4.0)


def test_inverted_pair_with_inverted_normal_is_the_same_contact() -> None:
    down = (0.0, 0.0, -1.0)
    outcome = measure(table(record(a="box", b="ball", normal=down)), table(record()))
    np.testing.assert_allclose(outcome.samples["normal_err_deg"], [0.0], atol=TOL)


def test_tilted_normal_gives_its_angle() -> None:
    tilt = math.radians(10.0)
    pred = table(record(normal=(math.sin(tilt), 0.0, math.cos(tilt))))
    outcome = measure(pred, table(record()))
    np.testing.assert_allclose(outcome.samples["normal_err_deg"], [10.0], atol=TOL)


def test_events_below_the_tiou_threshold_do_not_match() -> None:
    outcome = measure(table(record(span=(0.0, 1.0))), table(record(span=(5.0, 9.0))))
    assert outcome.samples["normal_err_deg"].size == 0
    assert outcome.ratios["precision"] == (0.0, 1.0)
    assert outcome.ratios["recall"] == (0.0, 1.0)
    assert outcome.ratios["contact_f1"] == (0.0, 2.0)


def test_other_pair_is_not_a_candidate() -> None:
    outcome = measure(table(record(b=ENV_ID)), table(record()))
    assert outcome.ratios["recall"] == (0.0, 1.0)


def test_kind_disagreement_is_counted_among_matches() -> None:
    pred = table(record(kind=ContactKind.RESTING))
    outcome = measure(pred, table(record()))
    assert outcome.ratios["kind_agreement"] == (0.0, 1.0)


def test_gap_and_point_errors_in_the_smaller_object_length() -> None:
    pred = table(record(point=(0.02, 0.0, 0.0), distance=-0.01))
    truth = table(record(distance=0.0))
    outcome = measure(pred, truth, {Stage.LIFT: sizes()})
    np.testing.assert_allclose(outcome.samples["gap_rel"], [-0.01 / BALL_LENGTH], atol=TOL)
    np.testing.assert_allclose(outcome.samples["point_rel"], [0.02 / BALL_LENGTH], atol=TOL)


def test_environment_contacts_use_the_object_length() -> None:
    pred = table(record(b=ENV_ID, point=(0.0, 0.04, 0.0), distance=0.0))
    truth = table(record(b=ENV_ID, distance=0.0))
    outcome = measure(pred, truth, {Stage.LIFT: sizes()})
    np.testing.assert_allclose(outcome.samples["point_rel"], [0.04 / BALL_LENGTH], atol=TOL)
    np.testing.assert_allclose(outcome.samples["gap_rel"], [0.0], atol=TOL)


def test_without_lift_context_gap_and_point_are_nan() -> None:
    pred = table(record(point=(0.02, 0.0, 0.0), distance=-0.01))
    summary = ContactEventMetric(OPTIONS).compute(pred, table(record(distance=0.0)), manifest())
    assert math.isnan(summary["gap_rel.mean"])
    assert math.isnan(summary["point_rel.mean"])
    assert summary["normal_err_deg.mean"] == pytest.approx(0.0, abs=TOL)


def test_interval_iou_pads_point_events_to_whole_frames() -> None:
    assert interval_iou(record(span=(4.0, 4.0)), record(span=(4.0, 4.0))) == 1.0
    assert interval_iou(record(span=(4.0, 4.0)), record(span=(5.0, 5.0))) == 0.0
    assert interval_iou(record(span=(0.0, 3.0)), record(span=(2.0, 5.0))) == pytest.approx(1 / 3)
