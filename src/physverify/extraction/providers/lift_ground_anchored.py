"""Back-projection anchored to the ground plane along each object's viewing ray."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.extraction._base import StageContext
from physverify.extraction.backprojection import camera_focal_phi, lift_depth
from physverify.extraction.floor_plane import FloorSampling, floor_points, plane_normal
from physverify.extraction.ground_anchor import (
    PIXEL_FLOOR_REL,
    SCALED_FIELDS,
    FloatArray,
    anchor_clip,
    anchored_fields,
    anchored_sigma,
    mask_angular_radius,
    normal_usable,
)
from physverify.extraction.lift_params import GroundAnchoredLiftParams, ground_anchor_params
from physverify.extraction.providers._backproject import SEED_HEX_DIGITS, BackprojectLift
from physverify.extraction.providers._common import expect_params
from physverify.extraction.stack_guard import stacked_rows
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import CENTROID_PX_REL
from physverify.schema.io import stable_hash
from physverify.schema.perception import CameraModel, DepthMap, MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind


class GroundAnchoredLift(BackprojectLift):
    """``lift/backproject`` whose objects are then moved along their rays: steady rows at the upper
    envelope of the implied camera height, not resting on another body, go onto the plane; the
    others by the object's bias.

    It extends ``run`` rather than ``_summaries``: the anchoring needs the clip's gravity and its
    sigma and every row of every object at once (h is a per-clip median), which the base computes
    after the per-row summaries."""

    name: ClassVar[str] = "ground_anchored"
    # 4: 1c8de44 changed the inherited footprint yaw trim; cached v3 lifts went stale.
    version: ClassVar[str] = "4"
    Params: ClassVar[type[BaseModel]] = GroundAnchoredLiftParams

    def __init__(self, params: BaseModel) -> None:
        anchored = expect_params(params, GroundAnchoredLiftParams)
        super().__init__(anchored)
        self.anchor = anchored

    def run(self, ctx: StageContext) -> ObjectStateTable:
        table = super().run(ctx)
        camera = ctx.require(Stage.CAMERA, CameraModel)
        track = self._track(ctx)
        down = self._normal(ctx, table, track, camera)
        normal_sigma = float(table.gravity_dir_sigma)
        params = ground_anchor_params(self.anchor)
        if not normal_usable(down, normal_sigma, params):
            return table
        visible = np.asarray(table.visible, dtype=bool)[..., None]
        position = np.where(visible, np.asarray(table.position, dtype=np.float64), np.nan)
        intrinsics = np.asarray(camera.K, dtype=np.float64)
        scale = _mask_scale(track, camera)
        angular_radius = mask_angular_radius(track.masks, float(intrinsics[1, 1]), float(scale[1]))
        result = anchor_clip(
            position,
            np.asarray(table.size, dtype=np.float64),
            np.asarray(table.quaternion, dtype=np.float64),
            np.asarray(table.primitive) == PrimitiveKind.SPHERE.value,
            down,
            normal_sigma,
            PIXEL_FLOOR_REL * angular_radius,
            self._stacked(track, position, intrinsics, down, scale),
            params,
        )
        if not math.isfinite(result.height):
            return table
        pixel = entry_rms(ctx.error_entry(Stage.TRACK, CENTROID_PX_REL)) * angular_radius
        phi = camera_focal_phi(camera)
        sigma_rel = anchored_sigma(result.gain, normal_sigma, phi, pixel)
        depth_rel = entry_rms(ctx.error_entry(Stage.DEPTH, "depth_rel"))
        fields = {name: getattr(table, name) for name in SCALED_FIELDS}
        scaled = anchored_fields(fields, result, sigma_rel, depth_rel, phi)
        return replace(
            table,
            position=scaled["position"],
            position_sigma=scaled["position_sigma"],
            size=scaled["size"],
            size_sigma=scaled["size_sigma"],
        )

    def _stacked(
        self,
        track: MaskTrack,
        position: FloatArray,
        intrinsics: FloatArray,
        down: FloatArray,
        scale: FloatArray,
    ) -> NDArray[np.bool_]:
        """Cells whose body sits on another tracked body in the image."""
        unit = down / float(np.linalg.norm(down)) if np.isfinite(down).all() else down
        return stacked_rows(
            track.masks, position, intrinsics, unit, scale, self.anchor.stack_reach_px
        )

    def _normal(
        self, ctx: StageContext, table: ObjectStateTable, track: MaskTrack, camera: CameraModel
    ) -> FloatArray:
        """Unit downward normal of the plane in camera axes (NaN when unknown)."""
        gravity = np.asarray(table.gravity_dir, dtype=np.float64)
        if self.anchor.normal_source == "gravity":
            return gravity
        z, valid, _ = lift_depth(ctx.require(Stage.DEPTH, DepthMap), track, self.anchor.shift)
        rng = np.random.default_rng(int(stable_hash(ctx.clip.clip_id)[:SEED_HEX_DIGITS], 16))
        sampling = FloorSampling()
        points = floor_points(z, valid, track.masks, np.asarray(camera.K), rng, sampling)
        normal = plane_normal(points, rng, sampling)
        return np.full(3, np.nan) if normal is None else normal


def _mask_scale(track: MaskTrack, camera: CameraModel) -> FloatArray:
    """Mask pixels per camera pixel along (x, y)."""
    height, width = track.masks.shape[2:]
    return np.array([width / camera.sampling.width, height / camera.sampling.height])


PLUGINS = (GroundAnchoredLift,)
