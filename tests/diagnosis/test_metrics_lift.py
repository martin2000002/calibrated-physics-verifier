"""lift/position: errors after the conditional scale fit, in object lengths."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics._state_pairing import object_length
from physverify.diagnosis.metrics.lift_position import LiftPositionMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
FRAMES = 6
S0 = 2.5
LENGTH = 0.2
TOL = 1e-9


def table(unit: LengthUnit, object_id: str = "ball") -> ObjectStateTable:
    return ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=unit,
        source_key="gt:clip" if unit is LengthUnit.METER else "lift:clip",
        t=np.arange(FRAMES) / 24.0,
        frame_index=np.arange(FRAMES, dtype=np.int64),
        object_ids=[object_id],
        labels=["ball"],
        primitive=["sphere"],
    )


def true_positions() -> NDArray[np.float64]:
    t = np.arange(FRAMES, dtype=np.float64)
    return np.stack([0.1 * t, 0.2 + 0.0 * t, 5.0 + 0.05 * t], axis=-1)[:, None, :]


def truth(position: NDArray[np.float64] | None = None) -> ObjectStateTable:
    return dataclasses.replace(
        table(LengthUnit.METER),
        position=true_positions() if position is None else position,
        size=np.full((FRAMES, 1, 3), LENGTH),
    )


def lifted(
    position: NDArray[np.float64],
    size: NDArray[np.float64] | None = None,
    sigma: float = np.nan,
    object_id: str = "ball",
) -> ObjectStateTable:
    """A LIFT table in relative units: meters / S0."""
    return dataclasses.replace(
        table(LengthUnit.RELATIVE, object_id),
        position=position / S0,
        size=(np.full((FRAMES, 1, 3), LENGTH) if size is None else size) / S0,
        position_sigma=np.full((FRAMES, 1, 3), sigma),
    )


def measure(pred: ObjectStateTable, truth_table: ObjectStateTable) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=truth_table,
        context=MappingProxyType({}),
        gt=None,
        options=OPTIONS,
        plan=None,
    )
    return LiftPositionMetric(OPTIONS).samples(case)


def test_exact_lift_has_no_error_after_the_scale_fit() -> None:
    outcome = measure(lifted(true_positions()), truth())
    for key in ("centroid_rel", "size_rel", "position_err_m"):
        np.testing.assert_allclose(outcome.samples[key], 0.0, atol=TOL)
    assert outcome.samples["centroid_rel"].size == 3 * FRAMES
    assert outcome.ratios["coverage"] == (float(FRAMES), float(FRAMES))


def test_perpendicular_offset_in_object_lengths() -> None:
    depth, offset = 5.0, 0.3
    fixed = np.tile([0.0, 0.0, depth], (FRAMES, 1, 1))
    outcome = measure(lifted(fixed + np.array([offset, 0.0, 0.0])), truth(fixed))
    fitted = depth**2 / (depth**2 + offset**2)
    centroid = outcome.samples["centroid_rel"].reshape(FRAMES, 3)
    np.testing.assert_allclose(centroid[:, 0], fitted * offset / LENGTH, atol=TOL)
    np.testing.assert_allclose(centroid[:, 1], 0.0, atol=TOL)
    np.testing.assert_allclose(centroid[:, 2], (fitted - 1.0) * depth / LENGTH, atol=TOL)


def test_twenty_percent_large_size() -> None:
    size = np.full((FRAMES, 1, 3), 1.2 * LENGTH)
    outcome = measure(lifted(true_positions(), size), truth())
    np.testing.assert_allclose(outcome.samples["size_rel"], 0.2, atol=TOL)
    assert outcome.samples["size_rel"].size == 3 * FRAMES


def test_coverage_counts_truth_samples_the_lift_missed() -> None:
    position = true_positions().copy()
    position[0] = np.nan
    outcome = measure(lifted(position), truth())
    assert outcome.ratios["coverage"] == (float(FRAMES - 1), float(FRAMES))
    assert outcome.samples["position_err_m"].size == FRAMES - 1


def test_position_z_uses_the_declared_sigma() -> None:
    depth, offset, sigma = 5.0, 0.3, 0.05
    fixed = np.tile([0.0, 0.0, depth], (FRAMES, 1, 1))
    outcome = measure(lifted(fixed + np.array([offset, 0.0, 0.0]), sigma=sigma), truth(fixed))
    fitted = depth**2 / (depth**2 + offset**2)
    s = S0 * fitted
    expected_x = (offset / S0) / sigma
    expected_z = (depth / S0 - depth / s) / sigma
    z = outcome.samples["position_z"].reshape(FRAMES, 3)
    np.testing.assert_allclose(z[:, 0], expected_x, atol=1e-9)
    np.testing.assert_allclose(z[:, 2], expected_z, atol=1e-9)


def test_no_position_z_without_a_sigma() -> None:
    outcome = measure(lifted(true_positions()), truth())
    assert outcome.samples["position_z"].size == 0


def test_no_common_object_gives_empty_samples() -> None:
    outcome = measure(lifted(true_positions(), object_id="other"), truth())
    assert all(values.size == 0 for values in outcome.samples.values())
    assert outcome.ratios["coverage"] == (0.0, float(FRAMES))


def test_object_length_ignores_missing_sizes() -> None:
    sizes = np.array([[0.1, 0.2, 0.4], [np.nan, 1.0, 1.0], [0.1, 0.2, 0.4]])
    assert object_length(sizes) == pytest.approx(0.2)
    assert np.isnan(object_length(np.full((2, 3), np.nan)))


def test_compute_summarizes_without_context() -> None:
    summary = LiftPositionMetric(OPTIONS).compute(lifted(true_positions()), truth(), manifest())
    assert summary["centroid_rel.mean"] == pytest.approx(0.0, abs=TOL)
    assert summary["coverage"] == 1.0
