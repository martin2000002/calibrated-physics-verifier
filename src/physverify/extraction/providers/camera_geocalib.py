"""GeoCalib (Veicht et al., ECCV 2024): focal length and gravity direction estimated from the
clip's frames, without priors.

Code Apache-2.0 at commit 1dedb29 (after the v1.0 tag: it carries the camera.py interpolation
fix, so it is not exactly the paper's code); weights ``geocalib-pinhole`` of the v1.0 release,
CC-BY-4.0. GeoCalib's ``gravity`` vector is the camera's *up* direction in OpenCV axes: its
latitude field is asin(ray . v) with rays (x, y, 1) and image y down (perspective_fields.py),
and roll = pitch = 0 gives (0, -1, 0). The gravity is therefore -v.

Conventions, not measurements: one pose for the whole clip (the camera rotation is assumed
constant), translation 0 and the yaw fixed by ``pose_from_gravity``; only LIFT reads the pose.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.gravity_pose import pose_from_gravity
from physverify.extraction.providers._assumed_fov import AssumedFovCamera
from physverify.extraction.providers._assumed_fov import Params as AssumedParams
from physverify.extraction.providers._common import (
    LazyModel,
    ProviderParams,
    centred_intrinsics,
    expect_params,
    set_torch_threads,
)
from physverify.extraction.weights import local_path, vendored_import_path, weight_file
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.schema.stages import Stage

FloatArray = NDArray[np.float64]

CODE = WeightSpec(
    WeightKind.GIT,
    "https://github.com/cvg/GeoCalib.git",
    "1dedb29f94fc11b896c5f572005978aeebccd67f",
)
CHECKPOINT = WeightSpec(
    WeightKind.URL,
    "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
    "53aac4956f42a2b0a34939ab6da33373dc593f1b",  # commit of the v1.0 release tag
    "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac",
)
VENDORED_MODULE = "geocalib.extractor"
CAMERA_MODEL = "pinhole"

FocalMode = Literal["geocalib", "assumed"]


class Params(ProviderParams):
    n_frames: int = Field(default=8, ge=1)
    # "assumed": K, FOV and FOV sigma of camera/assumed_fov's defaults, and GeoCalib's gravity
    # with that focal as a prior, so gravity and focal agree.
    focal: FocalMode = "geocalib"
    # phi_0, the measured floor of the relative focal sigma: the 0.68
    # quantile of |focal_rel| on v1-precise-hdri-dev. NaN = no floor (GeoCalib's own sigma alone).
    focal_sigma_floor_rel: float = math.nan

    @field_validator("focal_sigma_floor_rel")
    @classmethod
    def _floor_nan_or_non_negative(cls, value: float) -> float:
        if not (math.isnan(value) or (math.isfinite(value) and value >= 0)):
            raise ValueError("focal_sigma_floor_rel must be NaN or finite >= 0")
        return value


@dataclass(frozen=True)
class Calibration:
    """GeoCalib's estimate for a batch of B frames, in processing pixels and radians."""

    focal_px: float
    focal_sigma_px: float
    up: FloatArray  # (B, 3) camera up in OpenCV axes
    up_sigma_rad: FloatArray  # (B,)


class Calibrator(Protocol):
    def calibrate(self, frames: NDArray[np.uint8], focal_prior_px: float | None) -> Calibration:
        """(B, H, W, 3) RGB frames; a focal prior fixes the focal and estimates only gravity."""
        ...


class GeoCalibCamera:
    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "geocalib"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, CHECKPOINT)
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[Calibrator] = LazyModel(self.loader())

    def loader(self) -> Callable[[StageContext], Calibrator]:
        return load_calibrator

    def run(self, ctx: StageContext) -> CameraModel:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        picked = frames[frame_indices(frames.shape[0], self.params.n_frames)]
        assumed = assumed_camera(ctx) if self.params.focal == "assumed" else None
        prior = None if assumed is None else float(assumed.K[0, 0])
        set_torch_threads(ctx.threads)
        result = self._model.get(ctx).calibrate(picked, prior)
        pose = pose_from_gravity(clip_gravity(result))
        if assumed is not None:
            return replace(assumed, world_from_camera=pose)
        floor = self.params.focal_sigma_floor_rel
        return camera_model(s, result.focal_px, result.focal_sigma_px, pose, floor)


def clip_gravity(result: Calibration) -> FloatArray:
    """The clip's gravity in OpenCV camera axes: what the pose carries and what P0 gates."""
    return combine_gravity(to_opencv_gravity(result.up), result.up_sigma_rad)


def frame_indices(n_total: int, n_frames: int) -> NDArray[np.int64]:
    """``n_frames`` equally spaced rows from first to last (every row of a shorter clip)."""
    if n_total <= n_frames:
        return np.arange(n_total, dtype=np.int64)
    return np.round(np.linspace(0.0, n_total - 1.0, n_frames)).astype(np.int64)


def to_opencv_gravity(up: FloatArray) -> FloatArray:
    """GeoCalib's up vectors (B, 3) to unit gravity directions in OpenCV camera axes."""
    vectors = np.asarray(up, dtype=np.float64)
    gravity: FloatArray = -vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    return gravity


def combine_gravity(gravity: FloatArray, sigma_rad: FloatArray) -> FloatArray:
    """Inverse-variance mean of unit directions, renormalised; equal weights when some sigma is
    not finite or not > 0 (GeoCalib returns 0 when its covariance fails)."""
    sigma = np.asarray(sigma_rad, dtype=np.float64)
    known = sigma.shape == (gravity.shape[0],) and bool(np.all(np.isfinite(sigma) & (sigma > 0)))
    weights = 1.0 / sigma**2 if known else np.ones(gravity.shape[0])
    mean = weights @ gravity
    combined: FloatArray = mean / np.linalg.norm(mean)
    return combined


def horizontal_fov_deg(focal_px: float, width: int) -> float:
    return math.degrees(2.0 * math.atan(width / (2.0 * focal_px)))


def focal_rel(focal_px: float, focal_sigma_px: float, floor_rel: float = math.nan) -> float:
    """phi = max(sigma_f / f, floor). Without a floor (NaN) an invalid model sigma (not finite or
    not > 0) is unknown, NaN, never 0; with a floor it counts as 0, so phi is the floor."""
    valid = math.isfinite(focal_sigma_px) and focal_sigma_px > 0
    model = focal_sigma_px / focal_px if valid else (0.0 if math.isfinite(floor_rel) else math.nan)
    return model if math.isnan(floor_rel) else max(model, floor_rel)


def fov_sigma_deg(
    focal_px: float, focal_sigma_px: float, width: int, floor_rel: float = math.nan
) -> float:
    """sigma_fov = sin(fov) phi, the inverse of LIFT's phi = sigma_fov / sin(fov)."""
    phi = focal_rel(focal_px, focal_sigma_px, floor_rel)
    fov = math.radians(horizontal_fov_deg(focal_px, width))
    return math.degrees(math.sin(fov) * phi)


def camera_model(
    sampling: FrameSampling,
    focal_px: float,
    focal_sigma_px: float,
    pose: FloatArray,
    floor_rel: float = math.nan,
) -> CameraModel:
    return CameraModel(
        sampling=sampling,
        K=centred_intrinsics(focal_px, sampling),
        source=IntrinsicsSource.ESTIMATED,
        fov_deg=horizontal_fov_deg(focal_px, sampling.width),
        fov_sigma_deg=fov_sigma_deg(focal_px, focal_sigma_px, sampling.width, floor_rel),
        world_from_camera=pose,
    )


def assumed_camera(ctx: StageContext) -> CameraModel:
    """camera/assumed_fov with its default FOV and sigma (the same values, not a copy)."""
    return AssumedFovCamera(AssumedParams()).run(ctx)


def load_calibrator(ctx: StageContext) -> Calibrator:
    """The vendored model, loaded from the pinned local checkpoint (never from the network)."""
    root = local_path(CODE, ctx.paths)
    with vendored_import_path(root):
        module = importlib.import_module(VENDORED_MODULE)
    model = module.GeoCalib(weights=str(weight_file(CHECKPOINT, ctx.paths)))
    return TorchCalibrator(model.to(ctx.device).eval(), ctx.device)


class TorchCalibrator:
    """``Calibrator`` over ``geocalib.GeoCalib.calibrate`` with intrinsics shared by the batch."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    def calibrate(self, frames: NDArray[np.uint8], focal_prior_px: float | None) -> Calibration:
        import torch

        images = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        images = images.to(self.device, dtype=torch.float32) / 255.0
        n = frames.shape[0]
        priors = None
        if focal_prior_px is not None:
            priors = {"focal": torch.full((n,), focal_prior_px, device=self.device)}
        # With the focal fixed there is nothing to share, and GeoCalib's shared system indexes a
        # focal gradient that a prior removes (lm_optimizer.py setup_system_shared_intrinsics).
        out = self.model.calibrate(
            images, camera_model=CAMERA_MODEL, priors=priors, shared_intrinsics=priors is None
        )
        return Calibration(
            focal_px=float(np.median(_numpy(out["camera"].f)[..., 0])),
            focal_sigma_px=float(np.median(_numpy(out["focal_uncertainty"]))),
            up=_numpy(out["gravity"].vec3d).reshape(n, 3),
            up_sigma_rad=per_frame(_numpy(out["gravity_uncertainty"]), n),
        )


def _numpy(tensor: Any) -> FloatArray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)


def per_frame(values: FloatArray, n: int) -> FloatArray:
    """One value per frame: as given, broadcast from one, else NaN (equal weights)."""
    flat = np.ravel(values)
    if flat.size == n:
        return flat
    return np.full(n, flat[0] if flat.size == 1 else math.nan)


PLUGINS = (GeoCalibCamera,)
