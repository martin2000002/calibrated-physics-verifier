"""track/mask: J, F and the centroid error on analytic discs (ADR 0007 §7)."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics.track_mask import (
    MaskMetric,
    boundary_f,
    boundary_tolerance_px,
    region_similarity,
)
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import MaskTrack

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
SIZE = 48
FRAMES = 3
RADIUS = 6


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


def disc(cx: int, cy: int) -> NDArray[np.bool_]:
    rows, cols = np.mgrid[:SIZE, :SIZE]
    inside: NDArray[np.bool_] = (cols - cx) ** 2 + (rows - cy) ** 2 <= RADIUS**2
    return inside


def track(ids: list[str], discs: list[NDArray[np.bool_]]) -> MaskTrack:
    per_frame = np.stack(discs)
    masks = np.repeat(per_frame[None], FRAMES, axis=0)
    return MaskTrack(
        sampling=sampling(),
        object_ids=np.array(ids, dtype=str),
        labels=np.array(ids, dtype=str),
        masks=masks,
        scores=np.ones((FRAMES, len(ids))),
    )


def measure(pred: MaskTrack, truth: MaskTrack) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=truth,
        context=MappingProxyType({}),
        gt=None,
        options=OPTIONS,
        plan=None,
    )
    return MaskMetric(OPTIONS).samples(case)


def test_identical_masks_score_one_with_zero_centroid_error() -> None:
    truth = track(["a", "b"], [disc(12, 12), disc(34, 30)])
    outcome = measure(track(["p0", "p1"], [disc(12, 12), disc(34, 30)]), truth)
    np.testing.assert_allclose(outcome.samples["mask_iou"], np.ones(2 * FRAMES))
    np.testing.assert_allclose(outcome.samples["boundary_f"], np.ones(2 * FRAMES))
    np.testing.assert_allclose(outcome.samples["j_and_f"], np.ones(2 * FRAMES))
    np.testing.assert_allclose(outcome.samples["centroid_px_rel"], np.zeros(4 * FRAMES))
    assert outcome.ratios["object_recall"] == (2.0, 2.0)


def test_shifted_disc_gives_the_shift_in_equivalent_radii() -> None:
    shift = 2
    truth_disc = disc(20, 20)
    outcome = measure(track(["p"], [disc(20 + shift, 20)]), track(["a"], [truth_disc]))
    r_eq = math.sqrt(float(truth_disc.sum()) / math.pi)
    centroid = outcome.samples["centroid_px_rel"]
    np.testing.assert_allclose(centroid[0::2], np.full(FRAMES, shift / r_eq), atol=1e-9)
    np.testing.assert_allclose(centroid[1::2], np.zeros(FRAMES), atol=1e-9)
    assert np.all(outcome.samples["mask_iou"] < 1.0)
    assert np.all((outcome.samples["boundary_f"] > 0.0) & (outcome.samples["boundary_f"] < 1.0))


def test_unmatched_truth_object_scores_zero() -> None:
    truth = track(["a", "b"], [disc(12, 12), disc(34, 30)])
    pred = track(["p0", "p1"], [disc(12, 12), disc(40, 8)])
    outcome = measure(pred, truth)
    np.testing.assert_allclose(outcome.samples["mask_iou"], [1.0] * FRAMES + [0.0] * FRAMES)
    np.testing.assert_allclose(outcome.samples["boundary_f"], [1.0] * FRAMES + [0.0] * FRAMES)
    assert outcome.samples["centroid_px_rel"].size == 2 * FRAMES
    assert outcome.ratios["object_recall"] == (1.0, 2.0)


def test_frames_without_the_truth_object_are_skipped() -> None:
    truth = track(["a"], [disc(20, 20)])
    hidden = truth.masks.copy()
    hidden[0] = False
    truth = MaskTrack(
        sampling=truth.sampling,
        object_ids=truth.object_ids,
        labels=truth.labels,
        masks=hidden,
        scores=truth.scores,
    )
    outcome = measure(track(["p"], [disc(20, 20)]), truth)
    assert outcome.samples["mask_iou"].size == FRAMES - 1


def test_mismatched_mask_sizes_are_rejected() -> None:
    small = MaskTrack(
        sampling=FrameSampling(
            frame_indices=np.arange(FRAMES, dtype=np.int64),
            width=SIZE // 2,
            height=SIZE // 2,
            native_width=SIZE,
            native_height=SIZE,
            seconds_per_frame=1.0 / 24.0,
            step=2,
            max_side=SIZE // 2,
        ),
        object_ids=np.array(["p"]),
        labels=np.array(["p"]),
        masks=np.zeros((FRAMES, 1, SIZE // 2, SIZE // 2), dtype=bool),
        scores=np.ones((FRAMES, 1)),
    )
    with pytest.raises(ValueError, match="share frames and size"):
        measure(small, track(["a"], [disc(20, 20)]))


def test_davis_conventions_for_empty_masks() -> None:
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    assert region_similarity(empty, empty) == 1.0
    assert boundary_f(empty, empty, 1.0) == 1.0
    assert boundary_f(empty, disc(20, 20), 1.0) == 0.0


def test_boundary_tolerance_is_a_fraction_of_the_diagonal() -> None:
    assert boundary_tolerance_px(96, 72, 0.008) == 1.0
    assert boundary_tolerance_px(1920, 1080, 0.008) == float(round(0.008 * math.hypot(1920, 1080)))


def test_compute_summarizes_without_context() -> None:
    truth = track(["a"], [disc(20, 20)])
    summary = MaskMetric(OPTIONS).compute(truth, truth, manifest())
    assert summary["j_and_f.mean"] == pytest.approx(1.0)
    assert summary["object_recall"] == 1.0
