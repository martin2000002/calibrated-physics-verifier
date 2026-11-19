"""The affine background flow recovers a known camera motion; its key and cache."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from physverify.app.plugins import PLUGIN_KINDS
from physverify.core.errors import ConfigError
from physverify.extraction.camera_motion._base import (
    CAMERA_MOTION_ESTIMATORS,
    CameraMotionEstimator,
    build_estimator,
)
from physverify.extraction.camera_motion.affine_flow import AffineFlowEstimator, Params
from physverify.extraction.camera_motion.run import (
    CameraMotionStore,
    camera_motion_key,
    ensure_camera_motion,
)
from physverify.schema.camera_motion import CameraMotion
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import MaskTrack

SIDE = 160
SPF = 1 / 24
GRID = np.mgrid[2:SIDE:4, 2:SIDE:4]
Frames = NDArray[np.uint8]
Affine = NDArray[np.float64]


class ArrayVideo:
    """A ``VideoSource`` over frames already at ``SIDE`` pixels."""

    def __init__(self, frames: Frames) -> None:
        self._frames = frames

    @property
    def real_fps(self) -> float:
        return 1 / SPF

    @property
    def n_frames(self) -> int:
        return int(self._frames.shape[0])

    @property
    def native_size(self) -> tuple[int, int]:
        return (SIDE, SIDE)

    def frames(self, start: int, stop: int, step: int = 1, max_side: int | None = None) -> Frames:
        assert max_side == SIDE
        return self._frames[start:stop:step]


def texture(seed: int = 0) -> Frames:
    noise = np.random.default_rng(seed).random((SIDE, SIDE)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    grey = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def warped(frame: Frames, affine: Affine) -> Frames:
    return np.asarray(
        cv2.warpAffine(
            frame, affine, (SIDE, SIDE), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        ),
        dtype=np.uint8,
    )


def track(n_frames: int, objects: NDArray[np.bool_] | None = None) -> MaskTrack:
    masks = np.zeros((n_frames, 1, SIDE, SIDE), dtype=bool) if objects is None else objects
    sampling = FrameSampling(
        frame_indices=np.arange(n_frames, dtype=np.int64),
        width=SIDE,
        height=SIDE,
        native_width=SIDE,
        native_height=SIDE,
        seconds_per_frame=SPF,
        step=1,
        max_side=SIDE,
    )
    return MaskTrack(
        sampling=sampling,
        object_ids=np.array(["o"], dtype=str),
        labels=np.array(["o"], dtype=str),
        masks=masks,
        scores=np.ones((n_frames, 1)),
    )


def expected_px(inverse: Affine) -> float:
    """Median length of the displacement the warp moves the grid points by (backward map)."""
    forward = cv2.invertAffineTransform(inverse)
    points = np.stack([GRID[1].ravel(), GRID[0].ravel()], axis=1).astype(np.float64)
    moved = points @ forward[:, :2].T + forward[:, 2]
    return float(np.median(np.hypot(*(moved - points).T)))


def estimate(frames: Frames, objects: NDArray[np.bool_] | None = None) -> CameraMotion:
    return AffineFlowEstimator(Params()).estimate(ArrayVideo(frames), track(len(frames), objects))


def rotation_zoom(degrees: float, scale: float) -> Affine:
    return np.asarray(
        cv2.getRotationMatrix2D((SIDE / 2, SIDE / 2), degrees, scale), dtype=np.float64
    )


@pytest.mark.parametrize(
    "inverse",
    [
        np.array([[1.0, 0.0, 1.5], [0.0, 1.0, -0.7]]),
        rotation_zoom(1.0, 1.0),
        rotation_zoom(0.0, 1.02),
        rotation_zoom(-0.6, 0.99),
    ],
    ids=["translation", "rotation", "zoom", "rotation_zoom"],
)
def test_a_known_affine_camera_motion_is_recovered_within_a_tenth_of_a_pixel(
    inverse: Affine,
) -> None:
    base = texture()
    motion = estimate(np.stack([base, warped(base, inverse)]))
    shift_px = float(motion.pair_shift[0]) * math.hypot(SIDE, SIDE)
    assert abs(shift_px - expected_px(inverse)) <= 0.1
    assert motion.measured and motion.pair_seconds.tolist() == pytest.approx([SPF])


def test_a_still_camera_measures_zero() -> None:
    base = texture()
    motion = estimate(np.stack([base, base, base]))
    assert np.allclose(motion.pair_shift, 0.0, atol=1e-6)
    assert motion.background_fraction.tolist() == [1.0, 1.0]


def test_an_object_moving_inside_its_mask_is_not_camera_motion() -> None:
    base, patch = texture(), texture(seed=1)[:40, :40]
    before, after = base.copy(), base.copy()
    before[60:100, 40:80] = patch
    after[60:100, 50:90] = patch
    objects = np.zeros((2, 1, SIDE, SIDE), dtype=bool)
    objects[:, 0, 60:100, 40:90] = True
    motion = estimate(np.stack([before, after]), objects)
    assert float(motion.pair_shift[0]) * math.hypot(SIDE, SIDE) <= 0.05


def test_a_frame_without_enough_background_gives_nan() -> None:
    base = texture()
    objects = np.ones((2, 1, SIDE, SIDE), dtype=bool)
    motion = estimate(np.stack([base, base]), objects)
    assert math.isnan(float(motion.pair_shift[0]))
    assert motion.clip_statistics(per_second=True) is None


def test_per_second_divides_each_pair_by_its_seconds() -> None:
    motion = CameraMotion(
        "affine_flow", np.array([0.01, 0.02]), np.array([0.5, 0.25]), np.ones(2), True
    )
    assert motion.pair_values(per_second=True).tolist() == pytest.approx([0.02, 0.08])
    assert motion.clip_statistics(per_second=False) == pytest.approx(
        (float(np.percentile([0.01, 0.02], 90)), 0.02)
    )


def estimator() -> CameraMotionEstimator:
    return build_estimator("affine_flow", {}, PLUGIN_KINDS[CAMERA_MOTION_ESTIMATORS.name]())


def test_the_estimator_is_registered_and_its_params_are_validated() -> None:
    assert isinstance(estimator(), AffineFlowEstimator)
    with pytest.raises(ConfigError, match="unknown"):
        build_estimator("nope", {}, PLUGIN_KINDS[CAMERA_MOTION_ESTIMATORS.name]())
    with pytest.raises(ConfigError, match="invalid params"):
        build_estimator(
            "affine_flow", {"max_side": 0}, PLUGIN_KINDS[CAMERA_MOTION_ESTIMATORS.name]()
        )


def test_the_key_follows_the_track_key_the_estimator_and_the_dataset() -> None:
    base = camera_motion_key("t", estimator(), "d")
    assert camera_motion_key("t", estimator(), "d") == base
    assert camera_motion_key("u", estimator(), "d") != base
    assert camera_motion_key("t", estimator(), "e") != base
    other = build_estimator(
        "affine_flow", {"max_side": 128}, PLUGIN_KINDS[CAMERA_MOTION_ESTIMATORS.name]()
    )
    assert camera_motion_key("t", other, "d") != base


def test_the_motion_is_computed_once_and_then_loaded(tmp_path: Path) -> None:
    store = CameraMotionStore(tmp_path)
    base = texture()
    frames = np.stack([base, warped(base, rotation_zoom(0.0, 1.02))])
    calls: list[int] = []

    def inputs() -> tuple[ArrayVideo, MaskTrack]:
        calls.append(1)
        return ArrayVideo(frames), track(2)

    first = ensure_camera_motion(estimator(), store, "d", "c", "t", inputs)
    second = ensure_camera_motion(estimator(), store, "d", "c", "t", inputs)
    assert len(calls) == 1 and first == second
    key = camera_motion_key("t", estimator(), "d")
    assert sorted(p.name for p in store.directory("d", "c", "affine_flow", key).iterdir()) == [
        "key.txt",
        "output.json",
    ]


def test_a_failed_estimate_is_unmeasured_and_not_cached(tmp_path: Path) -> None:
    store = CameraMotionStore(tmp_path)

    def broken() -> tuple[ArrayVideo, MaskTrack]:
        raise FileNotFoundError("no video")

    motion = ensure_camera_motion(estimator(), store, "d", "c", "t", broken)
    assert not motion.measured and motion.clip_statistics(per_second=True) is None
    assert not any(tmp_path.rglob("output.json"))


def test_a_stale_key_file_is_a_miss(tmp_path: Path) -> None:
    store = CameraMotionStore(tmp_path)
    key = camera_motion_key("t", estimator(), "d")
    store.save("d", "c", key, CameraMotion.unmeasured("affine_flow"))
    assert store.load("d", "c", "affine_flow", key) is not None
    (store.directory("d", "c", "affine_flow", key) / "key.txt").write_text(
        "other", encoding="utf-8"
    )
    assert store.load("d", "c", "affine_flow", key) is None
