"""camera/from_depth: the depth stage's focal plus GeoCalib's gravity with that focal as prior,
with a fake calibrator injected through ``loader()`` (no torch, no weights)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from support import make_context
from synthetic_clip import SyntheticClip

from physverify.core.errors import MissingInputError
from physverify.extraction._base import StageContext
from physverify.extraction.gravity_pose import gravity_from_pose
from physverify.extraction.providers import camera_from_depth
from physverify.extraction.providers._geocalib import (
    Calibration,
    Calibrator,
    clip_gravity_sigma_deg,
)
from physverify.extraction.providers.camera_from_depth import DepthFocalCamera
from physverify.schema.perception import (
    CameraModel,
    DepthKind,
    DepthMap,
    IntrinsicsSource,
    horizontal_fov_deg,
)
from physverify.schema.stages import Stage

FOCAL_PX = 210.0
TILT_RAD = 0.3
UP_SIGMA_RAD = 0.01


class FakeCalibrator:
    """A camera tilted down by TILT_RAD; records the frames and the focal prior it received."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float | None]] = []

    def calibrate(self, frames: NDArray[np.uint8], focal_prior_px: float | None) -> Calibration:
        self.calls.append((frames.shape[0], focal_prior_px))
        n = frames.shape[0]
        up = np.array([0.0, -math.cos(TILT_RAD), -math.sin(TILT_RAD)])
        return Calibration(999.0, 1.0, np.tile(up, (n, 1)), np.full(n, UP_SIGMA_RAD))


def refusing_loader(ctx: StageContext) -> Calibrator:
    raise AssertionError("GeoCalib must not load when gravity is off")


def faked(load: Callable[[StageContext], Calibrator]) -> type[DepthFocalCamera]:
    class Faked(DepthFocalCamera):
        def loader(self) -> Callable[[StageContext], Calibrator]:
            return load

    return Faked


@pytest.fixture(autouse=True)
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_from_depth, "set_torch_threads", lambda threads: None)


def depth_context(clip: SyntheticClip, focal_px: float | None, sigma: float = math.nan) -> Any:
    sampling = make_context(clip).sampling
    shape = (sampling.n_frames, sampling.height, sampling.width)
    depth = DepthMap(
        sampling=sampling,
        depth=np.full(shape, 2.0, dtype=np.float32),
        kind=DepthKind.METRIC_Z,
        valid=np.ones(shape, dtype=bool),
        focal_px=focal_px,
        focal_sigma_px=sigma,
    )
    plan = {Stage.DEPTH: "moge2", Stage.CAMERA: "from_depth"}
    return make_context(clip, upstream={Stage.DEPTH: depth}, plan=plan)


def run(
    ctx: StageContext, load: Callable[[StageContext], Calibrator], **params: Any
) -> CameraModel:
    provider = faked(load)
    out = provider(provider.Params(**params)).run(ctx)
    assert isinstance(out, CameraModel)
    return out


def test_the_focal_is_the_depth_models(fixture_clip: SyntheticClip) -> None:
    fake = FakeCalibrator()
    ctx = depth_context(fixture_clip, FOCAL_PX)
    camera = run(ctx, lambda c: fake, n_frames=3)
    w = ctx.sampling.width
    assert camera.K[0, 0] == FOCAL_PX
    assert camera.fov_deg == pytest.approx(horizontal_fov_deg(FOCAL_PX, w))
    assert camera.source is IntrinsicsSource.ESTIMATED
    assert fake.calls == [(3, FOCAL_PX)]


def test_the_gravity_and_its_sigma_are_geocalibs(fixture_clip: SyntheticClip) -> None:
    camera = run(depth_context(fixture_clip, FOCAL_PX), lambda c: FakeCalibrator())
    assert camera.world_from_camera is not None
    np.testing.assert_allclose(
        gravity_from_pose(camera.world_from_camera),
        [0.0, math.cos(TILT_RAD), math.sin(TILT_RAD)],
        atol=1e-12,
    )
    assert camera.gravity_sigma_deg == pytest.approx(math.degrees(UP_SIGMA_RAD))


@pytest.mark.parametrize("sigma", [0.0, math.nan])
def test_a_gravity_sigma_geocalib_did_not_give_is_unknown(sigma: float) -> None:
    result = Calibration(FOCAL_PX, 0.0, np.zeros((3, 3)), np.full(3, sigma))
    assert math.isnan(clip_gravity_sigma_deg(result))


def test_the_gravity_sigma_is_the_median_frame_not_a_standard_error() -> None:
    result = Calibration(FOCAL_PX, 0.0, np.zeros((3, 3)), np.array([0.01, 0.02, 0.04]))
    assert clip_gravity_sigma_deg(result) == pytest.approx(math.degrees(0.02))


def test_without_gravity_geocalib_never_loads(fixture_clip: SyntheticClip) -> None:
    camera = run(depth_context(fixture_clip, FOCAL_PX), refusing_loader, gravity="none")
    assert camera.world_from_camera is None
    assert math.isnan(camera.gravity_sigma_deg)
    assert camera.K[0, 0] == FOCAL_PX


def test_a_depth_without_focal_is_an_error_not_a_fallback(fixture_clip: SyntheticClip) -> None:
    with pytest.raises(MissingInputError, match="depth/moge2 gives no focal"):
        run(depth_context(fixture_clip, None), refusing_loader)


def test_a_depth_at_another_width_is_refused(fixture_clip: SyntheticClip) -> None:
    ctx = depth_context(fixture_clip, FOCAL_PX)
    narrower = replace(ctx.sampling, width=ctx.sampling.width // 2)
    with pytest.raises(MissingInputError, match="width"):
        run(replace(ctx, sampling=narrower), refusing_loader, gravity="none")


def test_the_sigma_floor_applies_to_an_unknown_focal_sigma(fixture_clip: SyntheticClip) -> None:
    ctx = depth_context(fixture_clip, FOCAL_PX)
    unknown = run(ctx, refusing_loader, gravity="none")
    floored = run(ctx, refusing_loader, gravity="none", focal_sigma_floor_rel=0.05)
    assert math.isnan(unknown.fov_sigma_deg)
    fov = math.radians(floored.fov_deg)
    assert floored.fov_sigma_deg == pytest.approx(math.degrees(math.sin(fov) * 0.05))


def test_a_negative_floor_is_rejected() -> None:
    with pytest.raises(ValidationError, match="focal_sigma_floor_rel"):
        DepthFocalCamera.Params(focal_sigma_floor_rel=-0.1)
