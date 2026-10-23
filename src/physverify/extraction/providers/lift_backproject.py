"""Back-projection of tracked masks through depth and intrinsics (ADR 0005, 4.1-4.4)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.pipeline import PENDING_SOURCE_KEY
from physverify.extraction.providers._backprojection import (
    LiftErrors,
    ObjectSummary,
    ShiftStrategy,
    SummaryOptions,
    closed_form_sigma,
    focal_rel_sigma,
    lift_depth,
    monte_carlo_sigma,
    object_position,
    object_summary,
)
from physverify.extraction.providers._common import ProviderParams, expect_params, lookup_label
from physverify.extraction.providers._gravity_pose import LEVEL_CAMERA_GRAVITY, gravity_from_pose
from physverify.extraction.providers._temporal_smoothing import EdgeMode, stabilize_summaries
from physverify.extraction.providers._visibility import gap_visibility
from physverify.physics.tracks import OCCLUDED_AT
from physverify.schema.io import stable_hash
from physverify.schema.perception import CameraModel, DepthMap, IntrinsicsSource, MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]
SEED_HEX_DIGITS = 16

GravityMode = Literal["auto", "camera_pose", "level_camera", "none"]
# "none" is version 1 (in_frame = visible, occlusion unknown); "interpolate" infers both inside
# interior mask gaps (ADR 0005 4.2 as amended 2026-10-20, ``_visibility.py``).
GapMode = Literal["none", "interpolate"]
# "none" is version 2; "median" stabilises depth and image extent along each visible run
# (ADR 0005 4.2 as amended for it2, ``_temporal.py``), off until the J6 gate activates it. Run
# ends: "keep" (the first and last two rows stay unfiltered) is the only eligible form; "shifted"
# fails the survival test J5 and exists only for J6's descriptive run (plan-it2, judgement 2 a).
TemporalFilter = Literal["none", "median"]


class Params(ProviderParams):
    shift: ShiftStrategy = ShiftStrategy.ZERO_SHIFT
    gravity: GravityMode = "auto"
    gravity_by_dataset: dict[str, GravityMode] = {}
    level_sigma_deg: float = Field(default=10.0, ge=0.0)
    pose_sigma_deg: float = Field(default=2.0, ge=0.0)
    min_area_px: int = Field(default=9, ge=1)
    erode_px: int = Field(default=1, ge=0)
    extent_percentiles: tuple[float, float] = (2.0, 98.0)
    center_offset_frac: dict[PrimitiveKind, float] = {
        PrimitiveKind.SPHERE: 0.354,
        PrimitiveKind.CYLINDER: 0.354,
        PrimitiveKind.BOX: 0.5,
        PrimitiveKind.UNKNOWN: 0.4,
    }
    primitive_by_label: dict[str, PrimitiveKind] = {
        "ball": PrimitiveKind.SPHERE,
        "sphere": PrimitiveKind.SPHERE,
        "cube": PrimitiveKind.BOX,
        "box": PrimitiveKind.BOX,
        "cylinder": PrimitiveKind.CYLINDER,
    }
    sigma_mode: Literal["closed_form", "monte_carlo"] = "closed_form"
    mc_draws: int = Field(default=500, ge=2)
    gap_visibility: GapMode = "interpolate"
    # Must equal the checks' ``occluded_at`` (resolve_plan refuses otherwise); a gap estimate
    # below it and above 0 is unknown.
    gap_occluded_at: float = Field(default=OCCLUDED_AT, gt=0.0, le=1.0)
    temporal_filter: TemporalFilter = "none"
    temporal_window: int = Field(default=5, ge=3)
    temporal_edges: EdgeMode = "keep"

    @field_validator("temporal_window")
    @classmethod
    def _odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("temporal_window must be odd")
        return value


class BackprojectLift:
    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "backproject"
    version: ClassVar[str] = "3"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.DEPTH, Stage.CAMERA)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.DEPTH)
    reads_own_error: ClassVar[bool] = True
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ObjectStateTable:
        track = ctx.require(Stage.TRACK, MaskTrack)
        depth = ctx.require(Stage.DEPTH, DepthMap)
        camera = ctx.require(Stage.CAMERA, CameraModel)
        z, valid, unit = lift_depth(depth, track, self.params.shift)
        primitives = [self._primitive(label) for label in track.labels]
        table = ObjectStateTable.blank(
            clip_id=ctx.clip.clip_id,
            frame=CoordFrame.CAMERA_OPENCV,
            unit=unit,
            source_key=PENDING_SOURCE_KEY,
            t=ctx.sampling.times(),
            frame_index=np.asarray(ctx.sampling.frame_indices, dtype=np.int64),
            object_ids=track.object_ids.tolist(),
            labels=track.labels.tolist(),
            primitive=[kind.value for kind in primitives],
        )
        fields = _mutable_fields(table)
        errors = _lift_errors(ctx, camera)
        rng = np.random.default_rng(int(stable_hash(ctx.clip.clip_id)[:SEED_HEX_DIGITS], 16))
        summaries = self._summaries(track, z, valid, camera.K)
        for t_index, row in enumerate(summaries):
            for n_index, (summary, primitive) in enumerate(zip(row, primitives, strict=True)):
                self._fill(fields, (t_index, n_index), summary, primitive, camera, errors, rng)
        fields["confidence"] = np.clip(track.scores.astype(np.float64), 0.0, 1.0)
        if self.params.gap_visibility == "interpolate":
            gaps = gap_visibility(track.masks, fields["visible"], self.params.gap_occluded_at)
            fields["in_frame"], fields["occlusion"] = gaps.in_frame, gaps.occlusion
        mode = self.params.gravity_by_dataset.get(ctx.clip.dataset, self.params.gravity)
        gravity, gravity_sigma = self._gravity(camera, mode)
        return replace(table, **fields, gravity_dir=gravity, gravity_dir_sigma=gravity_sigma)

    def _summaries(
        self, track: MaskTrack, z: FloatArray, valid: NDArray[np.bool_], intrinsics: FloatArray
    ) -> list[list[ObjectSummary | None]]:
        """Rows (T) of per-object summaries; with ``"median"`` every object's visible runs are
        stabilised before positions and sigma are computed from them."""
        n_t, n_obj = track.masks.shape[:2]
        rows = [
            [
                object_summary(track.masks[t, n], z[t], valid[t], intrinsics, self._options)
                for n in range(n_obj)
            ]
            for t in range(n_t)
        ]
        if self.params.temporal_filter == "none":
            return rows
        window, edges = self.params.temporal_window, self.params.temporal_edges
        columns = [
            stabilize_summaries([row[n] for row in rows], window, edges) for n in range(n_obj)
        ]
        return [[columns[n][t] for n in range(n_obj)] for t in range(n_t)]

    @property
    def _options(self) -> SummaryOptions:
        return SummaryOptions(
            erode_px=self.params.erode_px,
            extent_percentiles=self.params.extent_percentiles,
            min_area_px=self.params.min_area_px,
        )

    def _primitive(self, label: str) -> PrimitiveKind:
        return lookup_label(label, self.params.primitive_by_label) or PrimitiveKind.UNKNOWN

    def _fill(
        self,
        fields: dict[str, Any],
        cell: tuple[int, int],
        summary: ObjectSummary | None,
        primitive: PrimitiveKind,
        camera: CameraModel,
        errors: LiftErrors,
        rng: np.random.Generator,
    ) -> None:
        if summary is None:
            fields["visible"][cell] = False
            fields["in_frame"][cell] = False
            return
        offset = self.params.center_offset_frac[primitive]
        position = object_position(summary.centre, summary.extents, offset)
        if self.params.sigma_mode == "monte_carlo":
            pos_sigma, size_sigma = monte_carlo_sigma(
                summary, camera.K, offset, errors, rng, self.params.mc_draws
            )
        else:
            pos_sigma, size_sigma = closed_form_sigma(
                position, summary.extents, summary.r_eq_px, camera.K, errors
            )
        fields["position"][cell] = position
        fields["position_sigma"][cell] = pos_sigma
        fields["size"][cell] = summary.extents
        fields["size_sigma"][cell] = size_sigma

    def _gravity(self, camera: CameraModel, mode: GravityMode) -> tuple[FloatArray, float]:
        """Gravity direction in camera coordinates and its angular sigma in radians (4.3)."""
        if mode == "auto":
            mode = "camera_pose" if camera.world_from_camera is not None else "level_camera"
        if mode == "level_camera":
            return LEVEL_CAMERA_GRAVITY.copy(), math.radians(self.params.level_sigma_deg)
        if mode == "none" or camera.world_from_camera is None:
            return np.full(3, np.nan), math.nan
        exact = camera.source is IntrinsicsSource.GROUND_TRUTH
        sigma = 0.0 if exact else math.radians(self.params.pose_sigma_deg)
        return gravity_from_pose(camera.world_from_camera), sigma


def _mutable_fields(table: ObjectStateTable) -> dict[str, Any]:
    names = (
        "position",
        "position_sigma",
        "size",
        "size_sigma",
        "visible",
        "in_frame",
        "occlusion",
    )
    return {name: np.array(getattr(table, name), copy=True) for name in names}


def _focal_phi(camera: CameraModel) -> float:
    if camera.source is IntrinsicsSource.GROUND_TRUTH:
        return 0.0
    if not (math.isfinite(camera.fov_deg) and math.isfinite(camera.fov_sigma_deg)):
        return math.nan
    return focal_rel_sigma(camera.fov_deg, camera.fov_sigma_deg)


def _lift_errors(ctx: StageContext, camera: CameraModel) -> LiftErrors:
    return LiftErrors(
        depth_rel=ctx.error_entry(Stage.DEPTH, "depth_rel"),
        centroid_px_rel=ctx.error_entry(Stage.TRACK, "centroid_px_rel"),
        centroid_rel=ctx.error_entry(Stage.LIFT, "centroid_rel"),
        size_rel=ctx.error_entry(Stage.LIFT, "size_rel"),
        phi=_focal_phi(camera),
    )


PLUGINS = (BackprojectLift,)
