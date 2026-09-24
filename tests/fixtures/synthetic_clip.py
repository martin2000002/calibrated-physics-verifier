"""Analytic synthetic clip: a sphere in free fall next to a box resting on the ground.

Every layer is computed in closed form (exact ray casting, no simulator), so tests can compare
pipeline pieces against exact values. World frame z-up in meters; the camera follows the
OpenCV convention (x right, y down, z forward) and looks along world +y.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from physverify.config.models import RuntimeSettings
from physverify.core.resample import resize_frame
from physverify.core.video import check_frame_budget
from physverify.datasets._base import ArrayGroundTruthStore, NativeGroundTruth
from physverify.physics.geometry import (
    Plane,
    Pose,
    Primitive,
    Solid,
    contact_manifold,
    quaternion_from_matrix,
)
from physverify.schema.clip import ClipManifest, ObjectQuery, Split, TimeSource, scaled_size
from physverify.schema.constants import ENV_ID, STANDARD_GRAVITY
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.stages import STAGE_ORDER
from physverify.schema.states import (
    CoordFrame,
    LengthUnit,
    ObjectStateTable,
    PrimitiveKind,
    ground_truth_source_key,
)

FloatArray = NDArray[np.float64]

DATASET = "synthetic_fixture"
CLIP_ID = "fixture_fall"
GT_VERSION = "analytic-1"
WIDTH, HEIGHT, N_FRAMES, FPS = 96, 72, 12, 24.0
FOCAL_PX = 80.0
CAMERA_POSITION = np.array([0.0, -4.0, 0.5])
CAMERA_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
"""world_from_camera rotation: columns are the camera x, y, z axes in world coordinates."""
SPHERE_RADIUS = 0.15
SPHERE_START = np.array([-0.4, 0.0, 1.2])
BOX_EXTENTS = np.array([0.5, 0.4, 0.3])
BOX_CENTER = np.array([0.4, 0.0, 0.15])
OBJECT_IDS = ("ball", "box")
PRIMITIVES = (PrimitiveKind.SPHERE.value, PrimitiveKind.BOX.value)
GRAVITY_DIR = np.array([0.0, 0.0, -1.0])
PALETTE = np.array([[120, 120, 120], [220, 40, 40], [40, 70, 220], [180, 210, 240]], dtype=np.uint8)


def times() -> FloatArray:
    return np.arange(N_FRAMES, dtype=np.float64) / FPS


def intrinsics() -> FloatArray:
    return np.array([[FOCAL_PX, 0.0, WIDTH / 2], [0.0, FOCAL_PX, HEIGHT / 2], [0.0, 0.0, 1.0]])


def world_from_camera() -> FloatArray:
    pose = np.eye(4)
    pose[:3, :3] = CAMERA_ROTATION
    pose[:3, 3] = CAMERA_POSITION
    return pose


def sphere_centres(t: FloatArray) -> FloatArray:
    centres = np.repeat(SPHERE_START[None, :], t.shape[0], axis=0)
    centres[:, 2] -= 0.5 * STANDARD_GRAVITY * t**2
    return centres


def _ray_directions() -> FloatArray:
    """World directions of the pixel-centre rays, scaled so that camera z = 1 (s = depth)."""
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    k = intrinsics()
    camera = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
    directions: FloatArray = camera @ CAMERA_ROTATION.T
    return directions


def _hit_sphere(d: FloatArray, centre: FloatArray) -> FloatArray:
    oc = CAMERA_POSITION - centre
    a = np.sum(d * d, axis=-1)
    b = 2.0 * (d @ oc)
    disc = b**2 - 4.0 * a * (oc @ oc - SPHERE_RADIUS**2)
    with np.errstate(invalid="ignore"):
        s = (-b - np.sqrt(disc)) / (2.0 * a)
    return np.where((disc >= 0) & (s > 0), s, np.inf)


def _hit_box(d: FloatArray) -> FloatArray:
    lo = (BOX_CENTER - BOX_EXTENTS / 2 - CAMERA_POSITION) / d
    hi = (BOX_CENTER + BOX_EXTENTS / 2 - CAMERA_POSITION) / d
    near = np.max(np.minimum(lo, hi), axis=-1)
    far = np.min(np.maximum(lo, hi), axis=-1)
    return np.where((far >= np.maximum(near, 0.0)) & (near > 0), near, np.inf)


def _hit_ground(d: FloatArray) -> FloatArray:
    s = -CAMERA_POSITION[2] / d[..., 2]
    return np.where(s > 0, s, np.inf)


def render() -> tuple[NDArray[np.uint8], NDArray[np.uint32], NDArray[np.float32]]:
    """RGB frames, segmentation (k = object k-1, 0 = none) and metric z, all (T, H, W)."""
    d = _ray_directions()
    rgb, seg, depth = [], [], []
    for centre in sphere_centres(times()):
        hits = np.stack([_hit_ground(d), _hit_sphere(d, centre), _hit_box(d)])
        nearest = np.argmin(hits, axis=0)
        s = np.min(hits, axis=0)
        labels = np.where(np.isfinite(s), nearest, 0).astype(np.uint32)
        colour = np.where(np.isfinite(s), nearest, 3)
        rgb.append(PALETTE[colour])
        seg.append(labels)
        depth.append(np.where(np.isfinite(s), s, 0.0).astype(np.float32))
    return np.stack(rgb), np.stack(seg), np.stack(depth)


def _world_states(visible: NDArray[np.bool_]) -> ObjectStateTable:
    t = times()
    n_t = t.shape[0]
    blank = ObjectStateTable.blank(
        clip_id=CLIP_ID,
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key=ground_truth_source_key(CLIP_ID),
        t=t,
        frame_index=np.arange(n_t, dtype=np.int64),
        object_ids=OBJECT_IDS,
        labels=OBJECT_IDS,
        primitive=PRIMITIVES,
    )
    zeros3 = np.zeros((n_t, 2, 3))
    velocity = zeros3.copy()
    velocity[:, 0, 2] = -STANDARD_GRAVITY * t
    acceleration = zeros3.copy()
    acceleration[:, 0, 2] = -STANDARD_GRAVITY
    size = np.stack([np.full(3, 2 * SPHERE_RADIUS), BOX_EXTENTS])
    return dataclasses.replace(
        blank,
        position=np.stack([sphere_centres(t), np.repeat(BOX_CENTER[None], n_t, 0)], axis=1),
        position_sigma=zeros3,
        quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (n_t, 2, 1)),
        orientation_sigma=np.zeros((n_t, 2)),
        velocity=velocity,
        velocity_sigma=zeros3,
        acceleration=acceleration,
        acceleration_sigma=zeros3,
        angular_velocity=zeros3,
        size=np.repeat(size[None], n_t, axis=0),
        size_sigma=zeros3,
        visible=visible,
        occlusion=np.zeros((n_t, 2)),
        confidence=np.ones((n_t, 2)),
        gravity_dir=GRAVITY_DIR.copy(),
        gravity_dir_sigma=0.0,
    )


def _to_camera(states: ObjectStateTable) -> ObjectStateTable:
    rotation = CAMERA_ROTATION.T
    q = quaternion_from_matrix(rotation)
    return dataclasses.replace(
        states,
        frame=CoordFrame.CAMERA_OPENCV,
        position=(states.position - CAMERA_POSITION) @ rotation.T,
        velocity=states.velocity @ rotation.T,
        acceleration=states.acceleration @ rotation.T,
        quaternion=np.tile(q, states.quaternion.shape[:2] + (1,)),
        gravity_dir=rotation @ states.gravity_dir,
    )


def _box_contact() -> ContactRecord:
    box = Solid(
        Primitive(PrimitiveKind.BOX, BOX_EXTENTS.copy()), Pose(BOX_CENTER.copy(), np.eye(3))
    )
    manifold = contact_manifold(box, Plane.ground(), tol=1e-6)
    assert manifold is not None
    return ContactRecord(
        object_a="box",
        object_b=ENV_ID,
        t_start=0.0,
        t_end=float(N_FRAMES - 1),
        kind=ContactKind.RESTING,
        normal=np.array([0.0, 0.0, 1.0]),
        points=manifold.points,
        manifold=ManifoldKind.PRIMITIVE,
        point_distances=manifold.distances,
        confidence=1.0,
        source="analytic",
    )


def _contact_tables(record: ContactRecord) -> dict[CoordFrame, ContactTable]:
    rotation = CAMERA_ROTATION.T
    camera_record = dataclasses.replace(
        record,
        normal=rotation @ record.normal,
        points=(record.points - CAMERA_POSITION) @ rotation.T,
    )
    key = ground_truth_source_key(CLIP_ID)
    return {
        frame: ContactTable(
            clip_id=CLIP_ID, frame=frame, unit=LengthUnit.METER, source_key=key, records=(rec,)
        )
        for frame, rec in (
            (CoordFrame.WORLD_ZUP, record),
            (CoordFrame.CAMERA_OPENCV, camera_record),
        )
    }


def native_ground_truth(
    segmentation: NDArray[np.uint32], z: NDArray[np.float32]
) -> NativeGroundTruth:
    visible = np.stack([(segmentation == k + 1).any(axis=(1, 2)) for k in range(2)], axis=1)
    world = _world_states(visible)
    return NativeGroundTruth(
        version=GT_VERSION,
        frame_indices=np.arange(N_FRAMES, dtype=np.int64),
        segmentation=segmentation,
        z=z,
        object_ids=np.array(OBJECT_IDS),
        labels=np.array(OBJECT_IDS),
        K=intrinsics(),
        world_from_camera=world_from_camera(),
        states={CoordFrame.WORLD_ZUP: world, CoordFrame.CAMERA_OPENCV: _to_camera(world)},
        contacts=_contact_tables(_box_contact()),
    )


def manifest() -> ClipManifest:
    return ClipManifest(
        dataset=DATASET,
        clip_id=CLIP_ID,
        split=Split.EVAL,
        video_path=f"{DATASET}/{CLIP_ID}",
        n_frames=N_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        container_fps=FPS,
        seconds_per_frame=1.0 / FPS,
        time_source=TimeSource.GROUND_TRUTH,
        object_queries=(ObjectQuery("ball", 1), ObjectQuery("box", 1)),
        gt_stages=STAGE_ORDER,
    )


class ArrayVideoSource:
    """VideoSource over frames already in memory."""

    def __init__(self, frames: NDArray[np.uint8], real_fps: float, max_frame_bytes: int) -> None:
        self._frames = frames
        self._real_fps = real_fps
        self._max_frame_bytes = max_frame_bytes

    @property
    def real_fps(self) -> float:
        return self._real_fps

    @property
    def n_frames(self) -> int:
        return int(self._frames.shape[0])

    @property
    def native_size(self) -> tuple[int, int]:
        return int(self._frames.shape[2]), int(self._frames.shape[1])

    def frames(
        self, start: int, stop: int, step: int = 1, max_side: int | None = None
    ) -> NDArray[np.uint8]:
        size = scaled_size(*self.native_size, max_side)
        rows = range(start, min(stop, self.n_frames), step)
        check_frame_budget(len(rows), size, self._max_frame_bytes)
        return np.stack([resize_frame(self._frames[i], size) for i in rows])


@dataclasses.dataclass(frozen=True)
class SyntheticClip:
    manifest: ClipManifest
    video: ArrayVideoSource
    ground_truth: ArrayGroundTruthStore
    native: NativeGroundTruth


def build_clip(runtime: RuntimeSettings) -> SyntheticClip:
    rgb, segmentation, z = render()
    native = native_ground_truth(segmentation, z)
    video = ArrayVideoSource(rgb, FPS, runtime.max_frame_bytes)
    return SyntheticClip(manifest(), video, ArrayGroundTruthStore(native), native)


class SyntheticFixtureDataset:
    """DatasetSource with the single analytic clip."""

    name: ClassVar[str] = DATASET

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._clip = build_clip(settings.runtime if settings is not None else RuntimeSettings())

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return [self._clip.manifest]

    def video(self, clip: ClipManifest) -> ArrayVideoSource:
        return self._clip.video

    def ground_truth(self, clip: ClipManifest) -> ArrayGroundTruthStore | None:
        return self._clip.ground_truth

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None
