"""depth/object_parts: the split of the per-object depth residual (ADR 0014 §2.1)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics.depth_object import ObjectDepthMetric, object_depths
from physverify.diagnosis.metrics.depth_object_parts import QUANTITIES, ObjectDepthPartsMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.providers.lift_backproject import Params as LiftParams
from physverify.schema.clip import FrameSampling
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.stages import Stage

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
FRAMES, SIZE = 4, 20
FLOAT32_TOL = 1e-5


def sampling() -> FrameSampling:
    return FrameSampling(
        frame_indices=np.arange(FRAMES, dtype=np.int64),
        width=SIZE,
        height=SIZE,
        native_width=SIZE,
        native_height=SIZE,
        seconds_per_frame=1.0 / 24.0,
        step=1,
        max_side=None,
    )


def object_masks() -> NDArray[np.bool_]:
    masks = np.zeros((FRAMES, 2, SIZE, SIZE), dtype=bool)
    masks[:, 0, 2:9, 2:9] = True
    masks[:, 1, 10:18, 10:18] = True
    return masks


def track() -> MaskTrack:
    return MaskTrack(
        sampling=sampling(),
        object_ids=np.array(["ball", "box"]),
        labels=np.array(["ball", "box"]),
        masks=object_masks(),
        scores=np.ones((FRAMES, 2)),
    )


def true_z() -> NDArray[np.float64]:
    t, rows, cols = np.mgrid[:FRAMES, :SIZE, :SIZE]
    z: NDArray[np.float64] = 2.0 + 0.1 * cols + 0.05 * rows + 0.3 * t
    return z


def depth_map(values: NDArray[np.float64], kind: DepthKind = DepthKind.METRIC_Z) -> DepthMap:
    return DepthMap(
        sampling=sampling(),
        depth=values.astype(np.float32),
        kind=kind,
        valid=np.ones(values.shape, dtype=bool),
    )


def case(pred: DepthMap, context: Mapping[Stage, StageOutput]) -> MetricCase:
    return MetricCase(
        clip=manifest(),
        pred=pred,
        truth=depth_map(true_z()),
        context=MappingProxyType(dict(context)),
        gt=None,
        options=OPTIONS,
        plan=None,
    )


def measure(pred: DepthMap) -> MetricOutcome:
    return ObjectDepthPartsMetric(OPTIONS).samples(case(pred, {Stage.TRACK: track()}))


def rms(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def test_between_and_within_add_up_to_object_rel() -> None:
    rng = np.random.default_rng(3)
    factor = np.ones(true_z().shape)
    for t in range(FRAMES):
        for n in range(2):
            factor[t][object_masks()[t, n]] = 1.0 + rng.normal(0.0, 0.1)
    pred = depth_map(1.3 * factor * true_z())
    parts = measure(pred).samples
    whole = ObjectDepthMetric(OPTIONS).samples(case(pred, {Stage.TRACK: track()})).samples
    between, within = parts["depth_rel_between"], parts["depth_rel_within"]
    assert between.size == within.size == whole["depth_rel"].size == FRAMES * 2
    assert rms(whole["depth_rel"]) ** 2 == pytest.approx(
        rms(between) ** 2 + rms(within) ** 2, abs=1e-12
    )
    np.testing.assert_allclose(between + within, whole["depth_rel"], atol=1e-12)


def test_global_scale_leaves_every_part_at_zero() -> None:
    outcome = measure(depth_map(1.25 * true_z(), DepthKind.RELATIVE_DEPTH))
    for name in QUANTITIES:
        np.testing.assert_allclose(outcome.samples[name], 0.0, atol=FLOAT32_TOL, err_msg=name)
    assert outcome.samples["depth_absrel_objects"].size == FRAMES
    assert outcome.samples["depth_d2_rel"].size == (FRAMES - 2) * 2


def test_constant_object_bias_is_all_between() -> None:
    eps = 0.1
    factor = np.full(true_z().shape, 1.25)
    factor[object_masks()[:, 0]] *= 1.0 + eps
    samples = measure(depth_map(factor * true_z())).samples
    between = samples["depth_rel_between"].reshape(FRAMES, 2)
    np.testing.assert_allclose(between[:, 0], np.sqrt(1 + eps) - 1, atol=1e-6)
    np.testing.assert_allclose(between[:, 1], 1 / np.sqrt(1 + eps) - 1, atol=1e-6)
    np.testing.assert_allclose(samples["depth_rel_within"], 0.0, atol=1e-6)
    np.testing.assert_allclose(samples["depth_d2_rel"], 0.0, atol=FLOAT32_TOL)


def test_one_frame_jump_gives_its_second_difference() -> None:
    eps = 0.2
    factor = np.ones(true_z().shape)
    factor[1][object_masks()[1, 0]] = 1.0 + eps
    samples = measure(depth_map(factor * true_z())).samples
    z_ball = object_depths(depth_map(true_z()), track(), np.eye(3), LiftParams())[:, 0]
    jump = eps * z_ball[1] / z_ball.mean()
    d2 = samples["depth_d2_rel"].reshape(FRAMES - 2, 2)
    np.testing.assert_allclose(d2[:, 0], [-2.0 * jump, jump], atol=FLOAT32_TOL)
    np.testing.assert_allclose(d2[:, 1], 0.0, atol=FLOAT32_TOL)


def test_oracle_shift_removes_an_affine_offset_zero_shift_keeps() -> None:
    pred = depth_map(1.0 / true_z() + 0.2, DepthKind.AFFINE_DISPARITY)
    samples = measure(pred).samples
    whole = ObjectDepthMetric(OPTIONS).samples(case(pred, {Stage.TRACK: track()})).samples
    assert rms(whole["depth_rel"]) > 0.01
    np.testing.assert_allclose(samples["depth_rel_oracle_shift"], 0.0, atol=FLOAT32_TOL)


def test_inverted_depth_order_leaves_the_oracle_bound_undefined() -> None:
    samples = measure(depth_map(1.0 - 1.0 / true_z(), DepthKind.AFFINE_DISPARITY)).samples
    assert samples["depth_rel_oracle_shift"].size == 0
    assert samples["depth_rel_between"].size == FRAMES * 2


def test_without_track_every_part_is_empty() -> None:
    outcome = ObjectDepthPartsMetric(OPTIONS).samples(case(depth_map(true_z()), {}))
    assert set(outcome.samples) == set(QUANTITIES)
    assert all(values.size == 0 for values in outcome.samples.values())


def test_wrong_output_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        ObjectDepthPartsMetric(OPTIONS).compute(sampling(), depth_map(true_z()), manifest())  # type: ignore[arg-type]
