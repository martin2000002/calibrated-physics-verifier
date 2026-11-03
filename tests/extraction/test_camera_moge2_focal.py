"""camera/moge2_focal with a fake MoGe-2 and a fake GeoCalib injected through the loaders (no
torch, no weights)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from support import entry, make_context, profile_of
from synthetic_clip import SyntheticClip

from physverify.extraction._base import StageContext, WeightKind
from physverify.extraction.backprojection import focal_rel_sigma
from physverify.extraction.cache import profile_keys
from physverify.extraction.providers import camera_moge2_focal
from physverify.extraction.providers._assumed_fov import Params as AssumedParams
from physverify.extraction.providers._common import frame_indices
from physverify.extraction.providers._geocalib import Calibration, Calibrator
from physverify.extraction.providers._moge2 import FrameEstimate, PointMapModel
from physverify.extraction.providers.camera_moge2_focal import (
    MoGe2FocalCamera,
    clip_focal,
    clip_phi,
)
from physverify.schema.error_profile import FOCAL_REL, ErrorProfile
from physverify.schema.perception import CameraModel, IntrinsicsSource, focal_from_fov
from physverify.schema.stages import Stage

PLAN = {Stage.CAMERA: "moge2_focal"}
NORMAL_MAD = 0.6745
"""median(|z|) of a standard normal."""
UP_SIGMA_RAD = 0.01


class FakeMoGe:
    """Returns ``focal_rel`` frame after frame, across batches; records what it received."""

    def __init__(self, focal_rel: Sequence[float]) -> None:
        self.focal_rel = np.asarray(focal_rel, dtype=np.float64)
        self.frames: list[NDArray[np.uint8]] = []
        self.fov: list[float | None] = []

    def estimate(self, frames: NDArray[np.uint8], fov_x_deg: float | None) -> FrameEstimate:
        start = sum(f.shape[0] for f in self.frames)
        self.frames.append(frames)
        self.fov.append(fov_x_deg)
        n, h, w = frames.shape[:3]
        focal = np.resize(np.roll(self.focal_rel, -start), n)
        return FrameEstimate(np.ones((n, h, w), dtype=np.float32), focal)


class FlakyMoGe:
    """Every batch of more than one frame raises; single frames return ``focal_rel`` in call
    order, except the calls listed in ``broken``, which raise."""

    def __init__(self, focal_rel: Sequence[float], broken: set[int]) -> None:
        self.focal_rel = list(focal_rel)
        self.broken = broken
        self.single_calls = 0

    def estimate(self, frames: NDArray[np.uint8], fov_x_deg: float | None) -> FrameEstimate:
        if frames.shape[0] > 1:
            raise RuntimeError("batch failed")
        k = self.single_calls
        self.single_calls += 1
        if k in self.broken:
            raise RuntimeError("frame failed")
        n, h, w = frames.shape[:3]
        return FrameEstimate(np.ones((n, h, w), dtype=np.float32), np.array([self.focal_rel[k]]))


class FakeCalibrator:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float | None]] = []

    def calibrate(self, frames: NDArray[np.uint8], focal_prior_px: float | None) -> Calibration:
        self.calls.append((frames.shape[0], focal_prior_px))
        n = frames.shape[0]
        up = np.tile([0.0, -1.0, 0.0], (n, 1))
        return Calibration(999.0, 1.0, up, np.full(n, UP_SIGMA_RAD))


def refusing(ctx: StageContext) -> Any:
    raise AssertionError("this model must not load")


def faked(
    moge: Callable[[StageContext], PointMapModel], calib: Callable[[StageContext], Calibrator]
) -> type[MoGe2FocalCamera]:
    class Faked(MoGe2FocalCamera):
        def moge_loader(self) -> Callable[[StageContext], PointMapModel]:
            return moge

        def loader(self) -> Callable[[StageContext], Calibrator]:
            return calib

    return Faked


@pytest.fixture(autouse=True)
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_moge2_focal, "set_torch_threads", lambda threads: None)


def run(
    clip: SyntheticClip,
    moge: PointMapModel,
    calib: Callable[[StageContext], Calibrator] = refusing,
    profile: ErrorProfile | None = None,
    **params: Any,
) -> CameraModel:
    provider = faked(lambda ctx: moge, calib)
    params.setdefault("gravity", "none")
    ctx = make_context(clip, plan=PLAN, profile=profile)
    out = provider(provider.Params(**params)).run(ctx)
    assert isinstance(out, CameraModel)
    return out


def lift_phi(camera: CameraModel) -> float:
    return focal_rel_sigma(camera.fov_deg, camera.fov_sigma_deg)


def measured(sigma: float, bias: float = 0.0) -> ErrorProfile:
    return profile_of(entry(Stage.CAMERA, "moge2_focal", FOCAL_REL, sigma, bias=bias))


def test_the_clip_focal_is_the_median_of_log_focal_with_a_mad_spread() -> None:
    found = clip_focal(np.array([0.5, 1.0, 2.0]))
    assert found is not None
    assert found.focal_rel == pytest.approx(1.0)
    assert found.spread_rel == pytest.approx(math.log(2.0) / NORMAL_MAD / math.sqrt(3))
    assert found.n_eff == 3
    even = clip_focal(np.array([0.8, 0.9, 1.0, 1.2]))
    assert even is not None
    assert even.focal_rel == pytest.approx(math.sqrt(0.9 * 1.0))


def test_frames_without_a_usable_focal_are_dropped_from_n_eff() -> None:
    found = clip_focal(np.array([math.nan, 0.9, math.inf, 0.0, -1.0, 0.9]))
    assert found is not None
    assert (found.focal_rel, found.spread_rel, found.n_eff) == (pytest.approx(0.9), 0.0, 2)
    assert clip_focal(np.array([math.nan, 0.0])) is None


@pytest.mark.parametrize(
    ("profile", "spread", "floor", "expected"),
    [(0.09, 0.02, 0.0, 0.09), (0.03, 0.07, 0.0, 0.07), (0.03, 0.02, 0.05, 0.05)],
)
def test_phi_is_the_largest_of_profile_spread_and_floor(
    profile: float, spread: float, floor: float, expected: float
) -> None:
    assert clip_phi(profile, spread, floor) == expected


def test_the_focal_comes_from_the_selected_frames_at_free_fov(fixture_clip: SyntheticClip) -> None:
    moge = FakeMoGe([0.8, 0.9, 1.0])
    camera = run(fixture_clip, moge, n_frames=3, batch_size=2)
    ctx = make_context(fixture_clip, plan=PLAN)
    s = ctx.sampling
    frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
    np.testing.assert_array_equal(
        np.concatenate(moge.frames), frames[frame_indices(frames.shape[0], 3)]
    )
    assert moge.fov == [None, None]
    assert camera.K[0, 0] == pytest.approx(0.9 * s.width)
    assert camera.source is IntrinsicsSource.ESTIMATED
    assert camera.world_from_camera is None


def test_geocalib_gets_the_same_frames_and_the_clip_focal_as_prior(
    fixture_clip: SyntheticClip,
) -> None:
    fake = FakeCalibrator()
    camera = run(fixture_clip, FakeMoGe([0.9]), lambda c: fake, gravity="geocalib", n_frames=4)
    assert fake.calls == [(4, pytest.approx(camera.K[0, 0]))]
    assert camera.world_from_camera is not None
    assert camera.gravity_sigma_deg == pytest.approx(math.degrees(UP_SIGMA_RAD))


def test_the_spread_reads_the_shared_robust_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_moge2_focal, "robust_scale", lambda values: 0.3)
    found = clip_focal(np.array([0.5, 1.0, 2.0, 4.0]))
    assert found is not None and found.spread_rel == pytest.approx(0.3 / 2.0)


def test_phi_is_the_measured_profile_entry(fixture_clip: SyntheticClip) -> None:
    camera = run(fixture_clip, FakeMoGe([0.9]), profile=measured(0.08, bias=0.02))
    assert lift_phi(camera) == pytest.approx(math.hypot(0.08, 0.02), rel=1e-9)


def test_a_wide_frame_spread_raises_phi_above_the_profile(fixture_clip: SyntheticClip) -> None:
    focal = [0.5, 1.0, 2.0]
    camera = run(fixture_clip, FakeMoGe(focal), profile=measured(0.01), n_frames=3)
    spread = math.log(2.0) / NORMAL_MAD / math.sqrt(3)
    assert lift_phi(camera) == pytest.approx(spread, rel=1e-9)


def test_an_unmeasured_focal_takes_the_assumed_cameras_sigma(fixture_clip: SyntheticClip) -> None:
    camera = run(fixture_clip, FakeMoGe([0.9]))
    assumed = AssumedParams()
    expected = focal_rel_sigma(assumed.fov_deg, assumed.fov_sigma_deg)
    assert lift_phi(camera) == pytest.approx(expected, rel=1e-9)


def test_with_no_usable_frame_the_camera_is_the_assumed_one(fixture_clip: SyntheticClip) -> None:
    calibrator = FakeCalibrator()
    camera = run(fixture_clip, FakeMoGe([math.nan]), lambda c: calibrator, gravity="geocalib")
    assumed = AssumedParams()
    focal = focal_from_fov(assumed.fov_deg, camera.sampling.width)
    assert camera.source is IntrinsicsSource.ASSUMED_FOV
    assert camera.K[0, 0] == pytest.approx(focal)
    assert camera.fov_sigma_deg == assumed.fov_sigma_deg
    assert calibrator.calls == [(8, pytest.approx(focal))]
    assert camera.world_from_camera is not None


def test_a_frame_whose_forward_raises_is_dropped(fixture_clip: SyntheticClip) -> None:
    moge = FlakyMoGe([0.8, 0.9, 1.0, 5.0], broken={3})
    camera = run(fixture_clip, moge, n_frames=4, batch_size=4)
    assert moge.single_calls == 4
    assert camera.K[0, 0] == pytest.approx(0.9 * camera.sampling.width)
    assert camera.source is IntrinsicsSource.ESTIMATED


def test_when_every_forward_raises_the_camera_is_the_assumed_one(
    fixture_clip: SyntheticClip,
) -> None:
    camera = run(fixture_clip, FlakyMoGe([0.9] * 4, broken={0, 1, 2, 3}), n_frames=4)
    assert camera.source is IntrinsicsSource.ASSUMED_FOV


def test_the_own_profile_entry_is_part_of_the_stage_key() -> None:
    assert profile_keys(MoGe2FocalCamera, PLAN) == ((Stage.CAMERA, "moge2_focal"),)


@pytest.mark.parametrize("bad", [math.nan, -0.1, math.inf])
def test_the_floor_and_the_unmeasured_sigma_are_numbers(bad: float) -> None:
    with pytest.raises(ValidationError, match="focal_sigma_floor_rel"):
        MoGe2FocalCamera.Params(focal_sigma_floor_rel=bad)
    with pytest.raises(ValidationError, match="unmeasured_focal_sigma_rel"):
        MoGe2FocalCamera.Params(unmeasured_focal_sigma_rel=bad)


def test_weights_are_moge2_and_geocalib_pinned() -> None:
    kinds = [spec.kind for spec in MoGe2FocalCamera.weights]
    assert kinds == [WeightKind.GIT, WeightKind.GIT, WeightKind.HF, WeightKind.GIT, WeightKind.URL]
    assert all(spec.revision for spec in MoGe2FocalCamera.weights)
