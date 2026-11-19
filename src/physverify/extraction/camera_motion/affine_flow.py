"""Global affine motion of the background between consecutive sampled frames.

Per pair: dense Farneback flow on the grey frames at ``max_side``; the background is what lies
outside the union of the tracked masks of both frames, dilated by ``dilation_frac`` of the image
diagonal. An affine map fitted with RANSAC to the flow at background points on a grid keeps the
global motion (translation, rotation, zoom) and drops incoherent motion (texture, flicker). The
pair's shift is the median length of the fitted affine displacement over those points, divided
by the image diagonal; NaN when at most ``min_background_frac`` of the image is background or
fewer than ``min_points`` grid points are.
"""

from __future__ import annotations

import math
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from physverify.datasets._base import VideoSource
from physverify.extraction.camera_motion._base import CameraMotionParams
from physverify.schema.camera_motion import CameraMotion
from physverify.schema.perception import MaskTrack

Grey = NDArray[np.uint8]
Mask = NDArray[np.bool_]


class Params(CameraMotionParams):
    """Fixed a priori, never fitted on calibration data."""

    max_side: int = Field(default=160, ge=16)
    dilation_frac: float = Field(default=0.02, ge=0.0)
    min_dilation_px: int = Field(default=2, ge=0)
    min_background_frac: float = Field(default=0.05, ge=0.0, lt=1.0)
    grid_step_px: int = Field(default=4, ge=1)
    min_points: int = Field(default=12, ge=3)
    flow_pyr_scale: float = Field(default=0.5, gt=0.0, lt=1.0)
    flow_levels: int = Field(default=4, ge=1)
    flow_winsize: int = Field(default=15, ge=3)
    flow_iterations: int = Field(default=3, ge=1)
    flow_poly_n: int = Field(default=5, ge=3)
    flow_poly_sigma: float = Field(default=1.2, gt=0.0)
    ransac_threshold_px: float = Field(default=0.5, gt=0.0)
    ransac_max_iters: int = Field(default=500, ge=1)
    ransac_refine_iters: int = Field(default=5, ge=0)


class AffineFlowEstimator:
    name: ClassVar[str] = "affine_flow"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[CameraMotionParams]] = Params

    def __init__(self, params: CameraMotionParams) -> None:
        assert isinstance(params, Params)
        self._params = params

    @property
    def params(self) -> CameraMotionParams:
        return self._params

    def estimate(self, video: VideoSource, track: MaskTrack) -> CameraMotion:
        s = track.sampling
        if not np.array_equal(s.frame_indices, np.arange(s.start, s.stop, s.step)):
            raise ValueError("camera motion needs a uniform track sampling")
        frames = video.frames(s.start, s.stop, s.step, self._params.max_side)
        if frames.shape[0] != track.masks.shape[0] or frames.shape[0] < 2:
            raise ValueError(f"{frames.shape[0]} frames for {track.masks.shape[0]} mask rows")
        grey = [np.asarray(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), dtype=np.uint8) for f in frames]
        objects = self._objects(track.masks.any(axis=1), grey[0].shape)
        pairs = [
            self._pair(grey[k - 1], grey[k], objects[k - 1] | objects[k])
            for k in range(1, len(grey))
        ]
        return CameraMotion(
            estimator=self.name,
            pair_shift=np.array([shift for shift, _ in pairs], dtype=np.float64),
            pair_seconds=np.diff(s.frame_indices).astype(np.float64) * s.seconds_per_frame,
            background_fraction=np.array([frac for _, frac in pairs], dtype=np.float64),
            measured=True,
        )

    def _objects(self, union: Mask, shape: tuple[int, ...]) -> list[Mask]:
        """Per row, the tracked masks at the frame size, dilated."""
        height, width = shape
        radius = max(
            self._params.min_dilation_px, round(self._params.dilation_frac * _diagonal(shape))
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        # OpenCV nearest-neighbour, the resize of the measurement the gate limits come from.
        return [
            cv2.dilate(
                cv2.resize(row.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST),
                kernel,
            ).astype(bool)
            for row in union
        ]

    def _pair(self, before: Grey, after: Grey, objects: Mask) -> tuple[float, float]:
        """The pair's shift (NaN without enough background) and its background fraction."""
        p = self._params
        background = ~objects
        step = p.grid_step_px
        rows, cols = np.mgrid[
            step // 2 : before.shape[0] : step, step // 2 : before.shape[1] : step
        ]
        on_grid = background[rows, cols]
        fraction = float(background.mean())
        if int(background.sum()) <= p.min_background_frac * background.size:
            return math.nan, fraction
        if int(on_grid.sum()) < p.min_points:
            return math.nan, fraction
        points = np.stack([cols[on_grid], rows[on_grid]], axis=1).astype(np.float32)
        return self._shift(before, after, points) / _diagonal(before.shape), fraction

    def _shift(self, before: Grey, after: Grey, points: NDArray[np.float32]) -> float:
        """Median length of the affine displacement fitted to the flow at ``points`` (x, y)."""
        p = self._params
        # flags 0: the zero array is only the output buffer, never an initial flow.
        flow = cv2.calcOpticalFlowFarneback(
            before,
            after,
            np.zeros((*before.shape, 2), dtype=np.float32),
            p.flow_pyr_scale,
            p.flow_levels,
            p.flow_winsize,
            p.flow_iterations,
            p.flow_poly_n,
            p.flow_poly_sigma,
            0,
        )
        cols, rows = points[:, 0].astype(np.intp), points[:, 1].astype(np.intp)
        affine, _ = cv2.estimateAffine2D(
            points,
            points + flow[rows, cols],
            method=cv2.RANSAC,
            ransacReprojThreshold=p.ransac_threshold_px,
            maxIters=p.ransac_max_iters,
            refineIters=p.ransac_refine_iters,
        )
        if affine is None:
            return math.nan
        field = points @ affine[:, :2].T + affine[:, 2] - points
        return float(np.median(np.hypot(field[:, 0], field[:, 1])))


def _diagonal(shape: tuple[int, ...]) -> float:
    return float(np.hypot(shape[0], shape[1]))


PLUGINS = (AffineFlowEstimator,)
