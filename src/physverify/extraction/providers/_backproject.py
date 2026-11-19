"""Back-projection of tracked masks through depth and intrinsics, shared by the lift providers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.config.plugin_context import PluginKind
from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.backprojection import (
    LiftErrors,
    ObjectSummary,
    SummaryOptions,
    camera_focal_phi,
    closed_form_sigma,
    lift_depth,
    monte_carlo_sigma,
    object_position,
    object_summary,
    place_object,
)
from physverify.extraction.centres._base import (
    CENTRE_STRATEGIES,
    CentreMethod,
    CentreStrategy,
    ImageCentres,
    build_centre,
)
from physverify.extraction.gravity_pose import LEVEL_CAMERA_GRAVITY, gravity_from_pose
from physverify.extraction.lift_geometry import (
    GroundFrame,
    ground_frame,
    orientation_sigma,
)
from physverify.extraction.lift_params import GravityMode, summary_options
from physverify.extraction.lift_params import LiftParams as Params
from physverify.extraction.mask_border import border_cut
from physverify.extraction.pinhole import ray_points
from physverify.extraction.pipeline import PENDING_SOURCE_KEY
from physverify.extraction.providers._common import expect_params
from physverify.extraction.providers._temporal_smoothing import stabilize_summaries
from physverify.extraction.providers._visibility import gap_visibility, nearer_cover
from physverify.schema.error_profile import (
    CENTROID_PX_REL,
    CENTROID_REL,
    DEPTH_REL,
    ORIENTATION_RAD,
    SIZE_REL,
)
from physverify.schema.io import stable_hash
from physverify.schema.perception import CameraModel, DepthMap, IntrinsicsSource, MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, ObjectStateTable, PrimitiveKind
from physverify.sizing.priors import lookup_label

FloatArray = NDArray[np.float64]
CentreRegistry = Mapping[str, type[CentreStrategy]]
SEED_HEX_DIGITS = 16


class BackprojectLift:
    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "backproject"
    # 6: 1c8de44 changed the footprint yaw trim (robust_scale) and cached v5 lifts went stale.
    # 7: the output gains nearer_cover.
    # 8: the image-plane centre comes from a configured strategy (params ``centre``).
    # 9: the output gains causal_centre where the strategy gives one.
    version: ClassVar[str] = "9"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.DEPTH, Stage.CAMERA)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.DEPTH)
    reads_own_error: ClassVar[bool] = True
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params
    plugin_reads: ClassVar[Mapping[str, PluginKind[Any]]] = MappingProxyType(
        {"centres": CENTRE_STRATEGIES}
    )

    def __init__(self, params: BaseModel, centres: CentreRegistry) -> None:
        self.params = expect_params(params, Params)
        self.centre = build_centre(self.params.centre, self.params.centre_params, centres)

    def run(self, ctx: StageContext) -> ObjectStateTable:
        track = self._track(ctx)
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
        mode = self.params.gravity_by_dataset.get(ctx.clip.dataset, self.params.gravity)
        gravity, gravity_sigma = self._gravity(camera, mode)
        frame = ground_frame(gravity)
        errors = _lift_errors(ctx, camera)
        rng = np.random.default_rng(int(stable_hash(ctx.clip.clip_id)[:SEED_HEX_DIGITS], 16))
        centres = self.centre.image_centres(track.masks, primitives)
        summaries = self._summaries(track, z, valid, camera.K, frame, primitives, centres)
        for t_index, row in enumerate(summaries):
            for n_index, (summary, primitive) in enumerate(zip(row, primitives, strict=True)):
                cell = (t_index, n_index)
                self._fill(fields, cell, summary, primitive, camera, errors, rng, frame)
        posed = np.isfinite(fields["quaternion"]).all(axis=-1)
        fields["orientation_sigma"][posed] = orientation_sigma(
            errors.orientation_rad, gravity_sigma
        )
        fields["confidence"] = np.clip(track.scores.astype(np.float64), 0.0, 1.0)
        if self.params.gap_visibility == "interpolate":
            gaps = gap_visibility(track.masks, fields["visible"], self.params.gap_occluded_at)
            fields["in_frame"], fields["occlusion"] = gaps.in_frame, gaps.occlusion
        cut = border_cut(track.masks, self.params.border_margin_px)
        cover = nearer_cover(track.masks, fields["position"][..., 2], fields["visible"])
        return replace(
            table,
            **fields,
            gravity_dir=gravity,
            gravity_dir_sigma=gravity_sigma,
            border_cut=cut,
            nearer_cover=cover,
            centre_method=None if centres is None else centres.method,
            causal_centre=_causal_centre(centres, camera.K),
        )

    def _track(self, ctx: StageContext) -> MaskTrack:
        """The tracked masks this lift places."""
        return ctx.require(Stage.TRACK, MaskTrack)

    def _summaries(
        self,
        track: MaskTrack,
        z: FloatArray,
        valid: NDArray[np.bool_],
        intrinsics: FloatArray,
        frame: GroundFrame | None,
        primitives: list[PrimitiveKind],
        centres: ImageCentres | None = None,
    ) -> list[list[ObjectSummary | None]]:
        """Rows (T) of per-object summaries at the strategy's ``centres`` where it set one; with
        ``"median"`` every object's visible runs are stabilised before positions and sigma are
        computed from them."""
        n_t, n_obj = track.masks.shape[:2]
        rows = [
            [
                self._summary(track, z, valid, intrinsics, frame, kind, centres, (t, n))
                for n, kind in enumerate(primitives)
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

    def _summary(
        self,
        track: MaskTrack,
        z: FloatArray,
        valid: NDArray[np.bool_],
        intrinsics: FloatArray,
        frame: GroundFrame | None,
        kind: PrimitiveKind,
        centres: ImageCentres | None,
        cell: tuple[int, int],
    ) -> ObjectSummary | None:
        t, n = cell
        method = CentreMethod.SURFACE_MEDIAN if centres is None else centres.method[cell]
        if method == CentreMethod.ABSENT:
            return None
        uv = None
        if centres is not None and method != CentreMethod.SURFACE_MEDIAN:
            uv = (float(centres.uv[t, n, 0]), float(centres.uv[t, n, 1]))
        mask = track.masks[t, n]
        return object_summary(mask, z[t], valid[t], intrinsics, self._options, frame, kind, uv)

    @property
    def _options(self) -> SummaryOptions:
        return summary_options(self.params)

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
        frame: GroundFrame | None,
    ) -> None:
        if summary is None:
            fields["visible"][cell] = False
            fields["in_frame"][cell] = False
            return
        placed = place_object(summary, primitive, self.params.center_offset_frac[primitive], frame)
        position = object_position(summary.centre, placed.offset)
        if self.params.sigma_mode == "monte_carlo":
            pos_sigma, size_sigma = monte_carlo_sigma(
                summary, placed.size, placed.offset, camera.K, errors, rng, self.params.mc_draws
            )
        else:
            pos_sigma, size_sigma = closed_form_sigma(
                position, placed.size, summary.r_eq_px, camera.K, errors
            )
        fields["position"][cell] = position
        fields["position_sigma"][cell] = pos_sigma
        fields["size"][cell] = placed.size
        fields["size_sigma"][cell] = size_sigma
        if placed.quaternion is not None:
            fields["quaternion"][cell] = placed.quaternion

    def _gravity(self, camera: CameraModel, mode: GravityMode) -> tuple[FloatArray, float]:
        """Gravity direction in camera coordinates and its angular sigma in radians (4.3)."""
        if mode == "auto":
            mode = "camera_pose" if camera.world_from_camera is not None else "level_camera"
        if mode == "level_camera":
            return LEVEL_CAMERA_GRAVITY.copy(), math.radians(self.params.level_sigma_deg)
        if mode == "none" or camera.world_from_camera is None:
            return np.full(3, np.nan), math.nan
        return gravity_from_pose(camera.world_from_camera), self._pose_sigma_rad(camera)

    def _pose_sigma_rad(self, camera: CameraModel) -> float:
        """0 for the truth, the camera's own estimate when it gives one, else pose_sigma_deg."""
        if camera.source is IntrinsicsSource.GROUND_TRUTH:
            return 0.0
        if math.isfinite(camera.gravity_sigma_deg):
            return math.radians(camera.gravity_sigma_deg)
        return math.radians(self.params.pose_sigma_deg)


def _causal_centre(centres: ImageCentres | None, intrinsics: FloatArray) -> FloatArray | None:
    """The strategy's causal pixel centres as K^-1 [u, v, 1] without its third coordinate."""
    if centres is None or centres.causal_uv is None:
        return None
    uv = centres.causal_uv.reshape(-1, 2)
    xy = ray_points(uv[:, 0], uv[:, 1], np.ones(len(uv)), intrinsics)[:, :2]
    out: FloatArray = xy.reshape(centres.causal_uv.shape)
    return out


def _mutable_fields(table: ObjectStateTable) -> dict[str, Any]:
    names = (
        "position",
        "position_sigma",
        "size",
        "size_sigma",
        "quaternion",
        "orientation_sigma",
        "visible",
        "in_frame",
        "occlusion",
    )
    return {name: np.array(getattr(table, name), copy=True) for name in names}


def _lift_errors(ctx: StageContext, camera: CameraModel) -> LiftErrors:
    return LiftErrors(
        depth_rel=ctx.error_entry(Stage.DEPTH, DEPTH_REL),
        centroid_px_rel=ctx.error_entry(Stage.TRACK, CENTROID_PX_REL),
        centroid_rel=ctx.error_entry(Stage.LIFT, CENTROID_REL),
        size_rel=ctx.error_entry(Stage.LIFT, SIZE_REL),
        phi=camera_focal_phi(camera),
        orientation_rad=ctx.error_entry(Stage.LIFT, ORIENTATION_RAD),
    )
