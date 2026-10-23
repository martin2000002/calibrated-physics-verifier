"""camera/geocalib with a fake GeoCalib injected through ``loader()`` (no torch model, no
weights, no network). That the registry build imports nothing heavy is in test_registry.py."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from support import make_context
from synthetic_clip import SyntheticClip

from physverify.core.cache_keys import params_hash
from physverify.extraction._base import StageContext
from physverify.extraction.backprojection import focal_rel_sigma
from physverify.extraction.gravity_pose import LEVEL_CAMERA_GRAVITY, gravity_from_pose
from physverify.extraction.providers._assumed_fov import AssumedFovCamera
from physverify.extraction.providers._assumed_fov import Params as AssumedParams
from physverify.extraction.providers.camera_geocalib import (
    Calibration,
    Calibrator,
    GeoCalibCamera,
    TorchCalibrator,
    camera_model,
    clip_gravity,
    combine_gravity,
    fov_sigma_deg,
    frame_indices,
    per_frame,
    to_opencv_gravity,
)
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.synth.placement import camera_axes

TOL = 1e-12
FOCAL_PX = 180.0
FOCAL_SIGMA_PX = 9.0


def geocalib_up(roll: float, pitch: float) -> NDArray[np.float64]:
    """``Gravity.from_rp`` of geocalib/gravity.py (commit 1dedb29): GeoCalib's output vector."""
    return np.array(
        [-math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch), math.sin(pitch)]
    )


def true_gravity(tilt_down_deg: float) -> NDArray[np.float64]:
    """World down in the OpenCV axes of a camera looking ``tilt_down_deg`` below the horizon."""
    tilt = math.radians(tilt_down_deg)
    position = np.array([0.0, -4.0 * math.cos(tilt), 4.0 * math.sin(tilt)])
    out: NDArray[np.float64] = camera_axes(position, np.zeros(3)).T @ np.array([0.0, 0.0, -1.0])
    return out


class FakeCalibrator:
    """Returns fixed GeoCalib outputs and records what the provider passed."""

    def __init__(
        self, up: NDArray[np.float64], sigma: NDArray[np.float64], focal: float = FOCAL_PX
    ) -> None:
        self.up, self.sigma, self.focal = up, sigma, focal
        self.frames: list[NDArray[np.uint8]] = []
        self.priors: list[float | None] = []

    def calibrate(self, frames: NDArray[np.uint8], focal_prior_px: float | None) -> Calibration:
        self.frames.append(frames)
        self.priors.append(focal_prior_px)
        n = frames.shape[0]
        focal = self.focal if focal_prior_px is None else focal_prior_px
        sigma_f = FOCAL_SIGMA_PX if focal_prior_px is None else 0.0
        return Calibration(focal, sigma_f, np.tile(self.up, (n, 1)), np.full(n, self.sigma))


def faked(calibrator: Calibrator) -> type[GeoCalibCamera]:
    class Faked(GeoCalibCamera):
        def loader(self) -> Callable[[StageContext], Calibrator]:
            return lambda ctx: calibrator

    return Faked


def run(clip: SyntheticClip, calibrator: Calibrator, **params: Any) -> CameraModel:
    provider = faked(calibrator)
    out = provider(provider.Params(**params)).run(make_context(clip))
    assert isinstance(out, CameraModel)
    return out


def test_a_level_camera_gives_the_level_gravity() -> None:
    np.testing.assert_array_equal(
        to_opencv_gravity(geocalib_up(0.0, 0.0)[None])[0], LEVEL_CAMERA_GRAVITY
    )


@pytest.mark.parametrize("tilt_deg", [5.0, 15.0, 25.0, 35.0])
def test_a_camera_tilted_down_gives_cos_and_sin(tilt_deg: float) -> None:
    tilt = math.radians(tilt_deg)
    # GeoCalib's pitch is positive looking up (latitude = asin(ray . up)).
    gravity = to_opencv_gravity(geocalib_up(0.0, -tilt)[None])[0]
    np.testing.assert_allclose(gravity, [0.0, math.cos(tilt), math.sin(tilt)], atol=TOL)
    np.testing.assert_allclose(gravity, true_gravity(tilt_deg), atol=TOL)


def test_intrinsics_and_the_fov_sigma_invert_lifts_phi(fixture_clip: SyntheticClip) -> None:
    sampling = make_context(fixture_clip).sampling
    camera = camera_model(sampling, FOCAL_PX, FOCAL_SIGMA_PX, np.eye(4))
    w, h = sampling.width, sampling.height
    np.testing.assert_array_equal(
        camera.K, [[FOCAL_PX, 0.0, w / 2], [0.0, FOCAL_PX, h / 2], [0.0, 0.0, 1.0]]
    )
    assert camera.fov_deg == pytest.approx(math.degrees(2 * math.atan(w / (2 * FOCAL_PX))), abs=TOL)
    phi = focal_rel_sigma(camera.fov_deg, camera.fov_sigma_deg)
    assert phi == pytest.approx(FOCAL_SIGMA_PX / FOCAL_PX, rel=TOL)
    assert camera.source is IntrinsicsSource.ESTIMATED


def test_geocalib_gets_the_sampled_frames_at_processing_size(
    fixture_clip: SyntheticClip,
) -> None:
    fake = FakeCalibrator(geocalib_up(0.0, -0.4), np.array(0.01))
    camera = run(fixture_clip, fake)
    sampling = make_context(fixture_clip).sampling
    (frames,) = fake.frames
    assert frames.shape[1:] == (sampling.height, sampling.width, 3)
    assert frames.dtype == np.uint8
    assert fake.priors == [None]
    assert camera.K[0, 0] == FOCAL_PX


@pytest.mark.parametrize("sigma_f", [0.0, -1.0, math.nan, math.inf])
def test_a_missing_focal_sigma_is_unknown_never_zero(sigma_f: float) -> None:
    assert math.isnan(fov_sigma_deg(FOCAL_PX, sigma_f, 256))


def unit(v: Any) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def test_equal_frames_give_their_direction_and_a_vague_frame_barely_counts() -> None:
    g = unit([0.1, 0.9, 0.4])
    np.testing.assert_allclose(combine_gravity(np.tile(g, (4, 1)), np.full(4, 0.02)), g, atol=TOL)
    other = unit([0.0, 1.0, 0.0])
    frames = np.vstack([np.tile(g, (3, 1)), other])
    vague = combine_gravity(frames, np.array([0.02, 0.02, 0.02, 0.2]))
    sure = combine_gravity(frames, np.array([0.2, 0.2, 0.2, 0.02]))
    assert np.degrees(np.arccos(np.clip(vague @ g, -1, 1))) < 0.2
    assert np.degrees(np.arccos(np.clip(sure @ other, -1, 1))) < 2.0


@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan])
def test_t5b_a_non_positive_or_unknown_sigma_gives_equal_weights(bad: float) -> None:
    frames = np.vstack([unit([0.0, 1.0, 0.0]), unit([0.0, 1.0, 1.0])])
    expected = unit(frames.sum(axis=0))
    np.testing.assert_allclose(combine_gravity(frames, np.array([0.01, bad])), expected, atol=TOL)
    np.testing.assert_allclose(combine_gravity(frames, np.array([0.01])), expected, atol=TOL)


@pytest.mark.parametrize(("n_total", "n_frames"), [(24, 8), (24, 1), (5, 8), (8, 8), (100, 7)])
def test_frames_are_equally_spaced_and_short_clips_give_all(n_total: int, n_frames: int) -> None:
    rows = frame_indices(n_total, n_frames)
    assert rows.size == min(n_total, n_frames)
    assert len(set(rows.tolist())) == rows.size and rows[0] == 0
    if n_frames > 1 or n_total <= n_frames:
        assert rows[-1] == n_total - 1
    steps = np.diff(rows)
    assert steps.size == 0 or int(steps.max() - steps.min()) <= 1


def test_assumed_focal_is_assumed_fov_with_geocalibs_gravity(
    fixture_clip: SyntheticClip,
) -> None:
    fake = FakeCalibrator(geocalib_up(0.05, -0.5), np.array(0.01))
    camera = run(fixture_clip, fake, focal="assumed")
    assumed = AssumedFovCamera(AssumedParams()).run(make_context(fixture_clip))
    assert isinstance(assumed, CameraModel)
    assert camera.K.tobytes() == assumed.K.tobytes()
    assert (camera.fov_deg, camera.fov_sigma_deg) == (assumed.fov_deg, assumed.fov_sigma_deg)
    assert camera.source is IntrinsicsSource.ASSUMED_FOV
    assert fake.priors == [assumed.K[0, 0]]
    expected = to_opencv_gravity(geocalib_up(0.05, -0.5)[None])[0]
    assert camera.world_from_camera is not None
    np.testing.assert_allclose(gravity_from_pose(camera.world_from_camera), expected, atol=TOL)


def test_a_valid_camera_with_one_pose_and_a_key_per_parameter(
    fixture_clip: SyntheticClip,
) -> None:
    fake = FakeCalibrator(geocalib_up(0.0, -0.3), np.array(0.01))
    camera = run(fixture_clip, fake, n_frames=4)
    camera.validate()
    assert camera.world_from_camera is not None and camera.world_from_camera.shape == (4, 4)
    assert len(fake.frames[0]) == 4
    np.testing.assert_allclose(
        gravity_from_pose(camera.world_from_camera), true_gravity(math.degrees(0.3)), atol=TOL
    )
    hashes = {
        params_hash(GeoCalibCamera.Params(**p).model_dump(mode="json"))
        for p in ({}, {"n_frames": 4}, {"focal": "assumed"}, {"focal_sigma_floor_rel": 0.1})
    }
    assert len(hashes) == 4
    assert (GeoCalibCamera.version, GeoCalibCamera.requires, GeoCalibCamera.heavy) == (
        "1",
        (),
        True,
    )


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> NDArray[np.float32]:
        return self.values


class FakeGeoCalibModel:
    """What ``geocalib.GeoCalib.calibrate`` returns for B frames with shared intrinsics."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def calibrate(self, images: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"shape": tuple(images.shape), "max": float(images.max()), **kwargs})
        b = images.shape[0]
        up = np.tile(geocalib_up(0.0, -0.2), (b, 1))
        return {
            "camera": SimpleNamespace(f=FakeTensor(np.full((b, 2), FOCAL_PX))),
            "gravity": SimpleNamespace(vec3d=FakeTensor(up)),
            "focal_uncertainty": FakeTensor([FOCAL_SIGMA_PX]),
            "gravity_uncertainty": FakeTensor(np.full(b, 0.01)),
        }


def test_the_torch_adapter_reads_geocalibs_outputs() -> None:
    pytest.importorskip("torch")
    model = FakeGeoCalibModel()
    frames = np.full((3, 16, 20, 3), 255, dtype=np.uint8)
    result = TorchCalibrator(model, "cpu").calibrate(frames, None)
    (call,) = model.calls
    assert call["shape"] == (3, 3, 16, 20) and call["max"] == pytest.approx(1.0)
    assert call["shared_intrinsics"] is True and call["priors"] is None
    assert call["camera_model"] == "pinhole"
    assert result.focal_px == pytest.approx(FOCAL_PX)
    assert result.focal_sigma_px == pytest.approx(FOCAL_SIGMA_PX)
    assert result.up.shape == (3, 3) and result.up_sigma_rad.shape == (3,)
    TorchCalibrator(model, "cpu").calibrate(frames, 150.0)
    prior = model.calls[-1]["priors"]["focal"]
    assert tuple(prior.shape) == (3,) and float(prior.min()) == pytest.approx(150.0)
    assert model.calls[-1]["shared_intrinsics"] is False


def test_per_frame_broadcasts_one_value_and_refuses_a_mismatch() -> None:
    np.testing.assert_array_equal(per_frame(np.array([0.1]), 3), [0.1, 0.1, 0.1])
    np.testing.assert_array_equal(per_frame(np.array([0.1, 0.2]), 2), [0.1, 0.2])
    assert bool(np.isnan(per_frame(np.array([0.1, 0.2]), 3)).all())


PHI0 = 0.12


@pytest.mark.parametrize(
    ("sigma_f", "floor", "expected"),
    [
        (FOCAL_SIGMA_PX, PHI0, PHI0),  # floor above the model's sigma
        (FOCAL_SIGMA_PX, 0.01, FOCAL_SIGMA_PX / FOCAL_PX),  # model above the floor
        (0.0, PHI0, PHI0),  # invalid model sigma with a floor: the floor
        (math.nan, PHI0, PHI0),
        (0.0, math.nan, math.nan),  # no floor: the sigma stays unknown, as before
        (FOCAL_SIGMA_PX, math.nan, FOCAL_SIGMA_PX / FOCAL_PX),
    ],
)
def test_t4b_the_focal_sigma_floor(sigma_f: float, floor: float, expected: float) -> None:
    sigma_deg = fov_sigma_deg(FOCAL_PX, sigma_f, 256, floor)
    fov = math.degrees(2 * math.atan(256 / (2 * FOCAL_PX)))
    phi = focal_rel_sigma(fov, sigma_deg)
    if math.isnan(expected):
        assert math.isnan(phi)
    else:
        assert phi == pytest.approx(expected, rel=TOL)


def test_the_floor_is_nan_or_non_negative() -> None:
    assert math.isnan(GeoCalibCamera.Params().focal_sigma_floor_rel)  # type: ignore[attr-defined]
    for bad in (-0.1, math.inf):
        with pytest.raises(ValueError, match="floor"):
            GeoCalibCamera.Params(focal_sigma_floor_rel=bad)


def test_the_floor_reaches_the_camera_only_with_the_estimated_focal(
    fixture_clip: SyntheticClip,
) -> None:
    fake = FakeCalibrator(geocalib_up(0.0, -0.3), np.array(0.01))
    floored = run(fixture_clip, fake, focal_sigma_floor_rel=PHI0)
    assert focal_rel_sigma(floored.fov_deg, floored.fov_sigma_deg) == pytest.approx(PHI0, rel=TOL)
    assumed = run(fixture_clip, fake, focal="assumed", focal_sigma_floor_rel=PHI0)
    assert assumed.fov_sigma_deg == AssumedParams().fov_sigma_deg


class VaryingCalibrator(FakeCalibrator):
    """Different up vectors and sigma per frame, and other ones when a focal prior is given."""

    def __init__(self) -> None:
        super().__init__(np.zeros(3), np.array(0.0))
        self.results: list[Calibration] = []

    def calibrate(self, frames: NDArray[np.uint8], focal_prior_px: float | None) -> Calibration:
        self.priors.append(focal_prior_px)
        n = frames.shape[0]
        tilt = np.linspace(0.2, 0.5, n) + (0.1 if focal_prior_px is not None else 0.0)
        up = np.stack([geocalib_up(0.02 * k, -t) for k, t in enumerate(tilt)])
        result = Calibration(FOCAL_PX, FOCAL_SIGMA_PX, up, np.linspace(0.01, 0.05, n))
        self.results.append(result)
        return result


@pytest.mark.parametrize("focal", ["geocalib", "assumed"])
def test_the_pose_carries_the_gravity_the_gates_evaluate(
    fixture_clip: SyntheticClip, focal: str
) -> None:
    """P0's gates read clip_gravity of the call made in the adopted focal mode (the free one, or
    the one with the assumed focal as prior); the provider's pose carries exactly that."""
    fake = VaryingCalibrator()
    camera = run(fixture_clip, fake, focal=focal)
    (result,) = fake.results
    assert camera.world_from_camera is not None
    np.testing.assert_allclose(
        gravity_from_pose(camera.world_from_camera), clip_gravity(result), atol=TOL
    )
    assert (fake.priors[0] is None) == (focal == "geocalib")
