"""Camera with one focal per clip from MoGe-2, and GeoCalib's gravity with that focal as prior.

MoGe-2 runs here only for its focal, on ``n_frames`` evenly spaced frames with a free field of
view; the depth stage keeps its own provider. The clip focal is exp(median ln f) over the frame
focals. A frame whose forward raises, or whose focal is not finite or not > 0, is dropped; with
no frame left the camera is camera/assumed_fov's as configured (``assumed``: its focal, FOV sigma
and source) and GeoCalib takes that focal as its prior, so a failed clip is recorded as assumed,
never as estimated.

The focal's relative sigma phi is the larger of this provider's own measured ``focal_rel`` (the
CAMERA entry of the error profile; without one ``unmeasured_focal_sigma_rel``, by default
camera/assumed_fov's phi), the spread of the frame focals, and ``focal_sigma_floor_rel``.

It is the precise profile's camera (``extraction.profiles.precise.plan``); elsewhere it is
selected with ``--set extraction.plan.camera=moge2_focal``. It has no provider table: its defaults
are those of ``Params``, overridden at ``extraction.providers.camera.moge2_focal.*``; the loader
rejects ``extraction.profiles.*`` keys in ``--set``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.core.log import get_logger
from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers import _geocalib, _moge2
from physverify.extraction.providers._assumed_fov import Params as AssumedParams
from physverify.extraction.providers._common import (
    LazyModel,
    expect_params,
    frame_indices,
    set_torch_threads,
)
from physverify.extraction.providers._geocalib import (
    ASSUMED_FIELD,
    Calibrator,
    GravitySource,
    assumed_camera,
    camera_model,
    load_calibrator,
    unmeasured_phi,
    with_gravity,
)
from physverify.extraction.providers._moge2 import MAX_RESOLUTION_LEVEL, PointMapModel, predict
from physverify.physics.uncertainty import entry_rms, robust_scale
from physverify.schema.error_profile import FOCAL_REL
from physverify.schema.perception import CameraModel
from physverify.schema.stages import Stage

PROVIDER_KEY = "camera/moge2_focal"
FRAME_ERRORS: Final = (RuntimeError, ValueError)
"""What a failing MoGe-2 forward raises (torch raises RuntimeError, out of memory included)."""

log = get_logger("extraction.camera_moge2_focal")


class Params(ProviderParams):
    n_frames: int = Field(default=8, ge=1)
    resolution_level: int = Field(default=MAX_RESOLUTION_LEVEL, ge=0, le=MAX_RESOLUTION_LEVEL)
    batch_size: int = Field(default=4, ge=1)
    gravity: GravitySource = "geocalib"
    focal_sigma_floor_rel: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    unmeasured_focal_sigma_rel: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    """None: camera/assumed_fov's phi (``unmeasured_phi``)."""
    assumed: AssumedParams = Field(default_factory=AssumedParams)


@dataclass(frozen=True)
class ClipFocal:
    """The clip focal over the frame width, the standard error of ln f, and the frames used."""

    focal_rel: float
    spread_rel: float
    n_eff: int


def clip_focal(frame_focal_rel: NDArray[np.float64]) -> ClipFocal | None:
    """f_clip = exp(median ln f_k); spread = robust_scale(ln f_k) / sqrt(N_eff), N_eff = the
    frames with a finite focal > 0. None when no frame has one."""
    focal = np.asarray(frame_focal_rel, dtype=np.float64)
    logs = np.log(focal[np.isfinite(focal) & (focal > 0)])
    if logs.size == 0:
        return None
    centre = float(np.median(logs))
    return ClipFocal(math.exp(centre), robust_scale(logs) / math.sqrt(logs.size), int(logs.size))


def frame_focals(
    model: PointMapModel, frames: NDArray[np.uint8], batch_size: int
) -> NDArray[np.float64]:
    """Focal / width per frame at a free field of view. When a batch raises, the frames are
    retried one by one and a frame that still raises is NaN, so it drops out of the clip focal."""
    try:
        return predict(model, frames, None, batch_size).focal_rel
    except FRAME_ERRORS as exc:
        log.warning("%s: batched MoGe-2 failed (%s); retrying frame by frame", PROVIDER_KEY, exc)
    return np.array([_one_focal(model, frames[i : i + 1]) for i in range(frames.shape[0])])


def _one_focal(model: PointMapModel, frame: NDArray[np.uint8]) -> float:
    try:
        return float(model.estimate(frame, None).focal_rel[0])
    except FRAME_ERRORS as exc:
        log.warning("%s: MoGe-2 failed on a frame (%s); dropped", PROVIDER_KEY, exc)
        return math.nan


def clip_phi(profile_phi: float, spread_rel: float, floor_rel: float) -> float:
    """phi_clip = max(phi_profile, spread, floor). A small d ln f is d f / f, so the spread of
    ln f is a relative focal sigma. MoGe-2 gives no focal sigma, and with a focal prior GeoCalib
    estimates gravity only, so no model sigma enters phi."""
    return max(profile_phi, spread_rel, floor_rel)


class MoGe2FocalCamera:
    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "moge2_focal"
    # 3: the frame spread reads physics.uncertainty.robust_scale (MAD / 0.6745, was 1.4826 * MAD).
    version: ClassVar[str] = "3"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    reads_own_error: ClassVar[bool] = True
    weights: ClassVar[tuple[WeightSpec, ...]] = (
        *_moge2.WEIGHTS,
        _geocalib.CODE,
        _geocalib.CHECKPOINT,
    )
    Params: ClassVar[type[BaseModel]] = Params
    provider_params: ClassVar[Mapping[str, tuple[Stage, str]]] = ASSUMED_FIELD

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._moge: LazyModel[PointMapModel] = LazyModel(self.moge_loader())
        self._calibrator: LazyModel[Calibrator] = LazyModel(self.loader())

    def moge_loader(self) -> Callable[[StageContext], PointMapModel]:
        return functools.partial(
            _moge2.load_model,
            resolution_level=self.params.resolution_level,
            provider_key=PROVIDER_KEY,
        )

    def loader(self) -> Callable[[StageContext], Calibrator]:
        return load_calibrator

    def run(self, ctx: StageContext) -> CameraModel:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        picked = frames[frame_indices(frames.shape[0], self.params.n_frames)]
        set_torch_threads(ctx.threads)
        focal = clip_focal(frame_focals(self._moge.get(ctx), picked, self.params.batch_size))
        camera = (
            assumed_camera(ctx, self.params.assumed)
            if focal is None
            else self._estimated(ctx, focal)
        )
        if self.params.gravity == "none":
            return camera
        result = self._calibrator.get(ctx).calibrate(picked, float(camera.K[0, 0]))
        return with_gravity(camera, result)

    def _estimated(self, ctx: StageContext, focal: ClipFocal) -> CameraModel:
        measured = entry_rms(ctx.error_entry(Stage.CAMERA, FOCAL_REL))
        profile = self._unmeasured() if math.isnan(measured) else measured
        phi = clip_phi(profile, focal.spread_rel, self.params.focal_sigma_floor_rel)
        return camera_model(ctx.sampling, focal.focal_rel * ctx.sampling.width, phi, None)

    def _unmeasured(self) -> float:
        return unmeasured_phi(self.params.unmeasured_focal_sigma_rel, self.params.assumed)


PLUGINS = (MoGe2FocalCamera,)
