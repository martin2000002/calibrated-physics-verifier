"""One valid instance of every schema record, with NaN wherever the schema allows it."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import (
    ClipManifest,
    FrameSampling,
    ObjectQuery,
    SamplingSpec,
    Split,
    TimeSource,
    resolve_sampling,
)
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import (
    ContactKind,
    ContactRecord,
    ContactTable,
    ManifoldKind,
    RawCollisions,
)
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.io import Record
from physverify.schema.matching import IdMap, MatchMethod
from physverify.schema.perception import (
    CameraModel,
    DepthKind,
    DepthMap,
    Detections,
    IntrinsicsSource,
    MaskTrack,
)
from physverify.schema.render import RenderLayers
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.scene import (
    AssetSource,
    CameraSpec,
    MagnitudeKind,
    ObjectSpec,
    PerturbationRecord,
    SceneSpec,
)
from physverify.schema.stages import Stage, StageArtifactMeta
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable, PrimitiveKind

NAN = math.nan


def manifest() -> ClipManifest:
    return ClipManifest(
        dataset="kubric_synthetic",
        clip_id="f1_0001_gravity_l2",
        split=Split.CALIB,
        video_path="synthetic/v1/render/f1_0001/frames",
        n_frames=24,
        width=128,
        height=96,
        container_fps=24.0,
        seconds_per_frame=1.0 / 24.0,
        time_source=TimeSource.GROUND_TRUTH,
        object_queries=(ObjectQuery("red cube", 1), ObjectQuery("ball", None)),
        scene_id="f1_0001",
        perturbation=perturbation(),
        gt_stages=(Stage.DETECT, Stage.TRACK),
        prompt=None,
        license="CC-BY-4.0",
    )


def sampling() -> FrameSampling:
    return resolve_sampling(manifest(), SamplingSpec(target_fps=12.0, max_side=64))


def states() -> ObjectStateTable:
    table = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.RELATIVE,
        source_key="lift-key",
        t=np.array([0.0, 0.5, 1.0]),
        frame_index=np.array([0, 12, 24]),
        object_ids=("a", "b"),
        labels=("cube", "ball"),
        primitive=(PrimitiveKind.BOX.value, PrimitiveKind.UNKNOWN.value),
    )
    position = table.position.copy()
    position[0, 0] = [1.0, 2.0, 3.0]
    return dataclasses.replace(table, position=position)


def contact() -> ContactRecord:
    return ContactRecord(
        object_a="a",
        object_b=ENV_ID,
        t_start=3.5,
        t_end=9.0,
        kind=ContactKind.RESTING,
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]]),
        manifold=ManifoldKind.PRIMITIVE,
        point_normals=np.tile([0.0, 0.0, 1.0], (4, 1)),
        point_distances=np.array([0.0, -1e-4, NAN, 2e-4]),
        orientation_assumed=True,
        max_penetration=1e-4,
        peak_force=None,
        confidence=NAN,
        source="proximity",
    )


def contacts() -> ContactTable:
    return ContactTable(
        clip_id="clip",
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key="gt:clip",
        records=(contact(),),
    )


def scale() -> ScaleEstimate:
    return ScaleEstimate(
        meters_per_unit=0.25,
        sigma=NAN,
        source=ScaleSource.SIZE_PRIOR,
        lift_key="lift-key",
        detail="median over 2 objects",
    )


def error_profile() -> ErrorProfile:
    entries = (
        ErrorEntry(
            stage=Stage.DEPTH,
            provider="vda_small",
            provider_version="1",
            quantity="abs_rel",
            unit="",
            bias=0.01,
            sigma=0.05,
            n=40,
            residuals=(0.1, -0.02, 0.0),
            conditions={"dataset": "kubric_synthetic"},
        ),
        ErrorEntry.exact(Stage.TRACK, "ground_truth", "mask_iou"),
    )
    return ErrorProfile(
        created_utc="2026-09-24T00:00:00+00:00", run_id="run-1", commit="abc123", entries=entries
    )


def id_map() -> IdMap:
    return IdMap(
        pairs=(("p0", "a", 0.9), ("p1", "b", NAN)),
        unmatched_pred=("p2",),
        unmatched_gt=(),
        method=MatchMethod.MASK_IOU,
        frames_used=(0, 4),
        min_score=0.5,
    )


def perturbation() -> PerturbationRecord:
    return PerturbationRecord(
        name="gravity_scale",
        target_checks=("R5",),
        magnitude=1.5,
        magnitude_unit="k",
        magnitude_kind=MagnitudeKind.INJECTED_PARAMETER,
        level=2,
        target_objects=("a",),
        frame_span=(0, 23),
        seed=7,
        params={"k": 1.5, "note": None},
    )


def scene() -> SceneSpec:
    cube = ObjectSpec(
        object_id="a",
        label="cube",
        asset_source=AssetSource.KUBASIC,
        asset_id="cube",
        primitive=PrimitiveKind.BOX,
        extents=(0.4, 0.4, 0.4),
        mass=1.0,
        friction=0.5,
        restitution=0.3,
        static=False,
        position=(0.0, 0.0, 0.2),
    )
    return SceneSpec(
        scene_id="f1_0001",
        family="f1_free_fall",
        seed=1,
        n_frames=24,
        settle_frames=0,
        resolution=(128, 96),
        camera=CameraSpec(position=(0.0, -5.0, 1.0), look_at=(0.0, 0.0, 0.5)),
        objects=(cube,),
    )


def check_result() -> CheckResult:
    return CheckResult(
        check_id="R5a",
        clip_id="clip",
        subject=("a",),
        frame_span=(2, 10),
        magnitude=None,
        sigma=NAN,
        unit="residual_ratio",
        verdict=Verdict.ABSTAIN,
        input_level=InputLevel.PERCEIVED,
        abstain_reason=AbstainReason.UNKNOWN_UNCERTAINTY,
        threshold=None,
        provenance={"lift": "backproject"},
        details={"n": 3, "x": NAN},
    )


def detections() -> Detections:
    grid = sampling()
    return Detections(
        sampling=grid,
        frame_index=int(grid.frame_indices[0]),
        boxes_xyxy=np.array([[1.0, 2.0, 10.5, 20.0]]),
        scores=np.array([NAN]),
        labels=np.array(["cube"]),
        object_ids=np.array(["p0"]),
    )


def mask_track() -> MaskTrack:
    grid = sampling()
    masks = np.zeros((grid.n_frames, 2, grid.height, grid.width), dtype=bool)
    masks[:, 0, 5:9, 3:7] = True
    scores = np.full((grid.n_frames, 2), NAN)
    return MaskTrack(
        sampling=grid,
        object_ids=np.array(["p0", "p1"]),
        labels=np.array(["cube", "ball"]),
        masks=masks,
        scores=scores,
    )


def depth_map() -> DepthMap:
    grid = sampling()
    depth = np.full((grid.n_frames, grid.height, grid.width), 2.5, dtype=np.float32)
    depth[0, 0, 0] = np.nan
    valid = np.isfinite(depth)
    return DepthMap(sampling=grid, depth=depth, kind=DepthKind.RELATIVE_DEPTH, valid=valid)


def camera_model() -> CameraModel:
    grid = sampling()
    k = np.array([[50.0, 0.0, grid.width / 2], [0.0, 50.0, grid.height / 2], [0.0, 0.0, 1.0]])
    return CameraModel(
        sampling=grid,
        K=k,
        source=IntrinsicsSource.ASSUMED_FOV,
        fov_deg=65.0,
        fov_sigma_deg=NAN,
        world_from_camera=np.eye(4),
    )


def stage_meta() -> StageArtifactMeta:
    return StageArtifactMeta(
        stage=Stage.LIFT,
        provider="backproject",
        provider_version="1",
        params_hash="p" * 64,
        sampling_hash="s" * 64,
        upstream={Stage.TRACK: "t" * 64, Stage.DEPTH: "d" * 64, Stage.CAMERA: "c" * 64},
        error_profile_hash="e" * 64,
        gt_version=None,
        device=None,
        cache_key="k" * 64,
        clip_id="clip",
        is_ground_truth=False,
        wall_s=1.25,
        peak_rss_mb=NAN,
        created_utc="2026-09-24T00:00:00+00:00",
    )


def render_layers() -> RenderLayers:
    t, h, w = 2, 4, 5
    z = np.full((t, h, w), 3.0, dtype=np.float32)
    z[0, 0, 0] = np.inf
    return RenderLayers(
        clip_id="clip",
        frame_indices=np.arange(t, dtype=np.int64),
        rgba=np.zeros((t, h, w, 4), dtype=np.uint8),
        segmentation=np.ones((t, h, w), dtype=np.uint32),
        z=z,
        object_ids=np.array(["a"]),
    )


def raw_collisions() -> RawCollisions:
    return RawCollisions(
        scene_id="f1_0001",
        a=np.array(["a", "a"]),
        b=np.array(["floor", "b"]),
        frame=np.array([3.25, 7.5]),
        position=np.zeros((2, 3)),
        normal=np.tile([0.0, 0.0, 1.0], (2, 1)),
        force=np.array([9.81, NAN]),
    )


SAMPLES: dict[str, Callable[[], Record]] = {
    "ObjectStateTable": states,
    "ContactRecord": contact,
    "ContactTable": contacts,
    "ScaleEstimate": scale,
    "ErrorProfile": error_profile,
    "IdMap": id_map,
    "FrameSampling": sampling,
    "ClipManifest": manifest,
    "SceneSpec": scene,
    "PerturbationRecord": perturbation,
    "CheckResult": check_result,
    "Detections": detections,
    "MaskTrack": mask_track,
    "DepthMap": depth_map,
    "CameraModel": camera_model,
    "StageArtifactMeta": stage_meta,
    "RenderLayers": render_layers,
    "RawCollisions": raw_collisions,
}
"""Record type name -> builder of one valid instance."""
