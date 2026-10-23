"""A Kubric scene from a ``SceneSpec`` and the job file handling shared by the container tasks
(container side: Python 3.9, Kubric 2022.4).

Frame convention: the simulation runs Kubric frames 0..settle + T - 1 and keeps the entries
settle..settle + T - 1 as native frames 0..T - 1; the renderer draws native frame i as Kubric
frame i + 1. Segmentation value k is ``SceneSpec.objects[k - 1]`` and 0 is the floor or background.
A job's marker file (``sim_meta.json``, ``kubric_meta.json``, ...) is written last, so the host
never mistakes a half-written job for a finished one.
"""

from __future__ import annotations

import contextlib
import functools
import os
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema.constants import ENV_ID
from physverify.schema.io import Record, dumps
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from physverify.schema.scene import ObjectSpec

KUBASIC_MANIFEST = "gs://kubric-public/assets/KuBasic/KuBasic.json"
HDRI_MANIFEST = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
PRIMITIVE_BOUNDS = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
HIDDEN_Z = -50.0
FLOOR_HALF_EXTENTS = (100.0, 100.0, 0.5)
FLOOR_COLOR = (0.45, 0.45, 0.45)
BACKGROUND_SEGMENTATION = 0
DEFAULT_PHYSICS_CLIENT = 0
SUN_OFFSET = (-1.0, -0.5, 3.0)
SUN_INTENSITY = 1.5
AMBIENT = (0.05, 0.05, 0.05)
PROBE_OFFSETS = tuple(
    (dx, dy, dz) for dx in (-0.3, 0.3) for dy in (-0.3, 0.3) for dz in (0.0, 0.1, 0.2, 0.4)
)


@dataclass
class BuiltScene:
    """``objects`` follow ``SceneSpec.objects``; ``simulator``/``renderer`` are None when unused."""

    scene: Any
    objects: list[Any]
    bounds: list[tuple[tuple[float, float, float], tuple[float, float, float]]]
    floor: Any
    simulator: Any = None
    renderer: Any = None


def kubric_scale(
    extents: Sequence[float], bounds_min: Sequence[float], bounds_max: Sequence[float]
) -> tuple[float, float, float]:
    """Kubric ``scale`` giving full ``extents`` to an asset spanning [bounds_min, bounds_max]."""
    span = np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64)
    scale = np.asarray(extents, dtype=np.float64) / span
    return (float(scale[0]), float(scale[1]), float(scale[2]))


def hidden_position(position: Sequence[float]) -> tuple[float, float, float]:
    """A pose under the floor slab: the object stays in the scene but draws no pixel."""
    x, y = (float(v) if np.isfinite(v) else 0.0 for v in position[:2])
    return (x, y, HIDDEN_Z)


def sim_frame_end(settle_frames: int, n_frames: int) -> int:
    return settle_frames + n_frames - 1


def native_entries(settle_frames: int, n_frames: int) -> slice:
    """Rows of Kubric's ``animation`` lists that are native frames 0..T - 1."""
    return slice(settle_frames, settle_frames + n_frames)


def kubric_frame(native_index: int) -> int:
    return native_index + 1


def probe_points(look_at: Sequence[float]) -> np.ndarray:
    """16 world points around the look-at target used to check K against ``project_point``."""
    return np.asarray(look_at, dtype=np.float64) + np.asarray(PROBE_OFFSETS, dtype=np.float64)


def build(
    spec: SceneSpec,
    floor: Mapping[str, float],
    *,
    scratch_dir: str,
    simulate: bool = False,
    render: bool = False,
    hdri_ids: Sequence[str] = (),
) -> BuiltScene:
    """Kubric scene with the requested views; both views observe every asset added afterwards."""
    import kubric as kb

    scene = kb.Scene(
        resolution=tuple(spec.resolution),
        frame_start=1,
        frame_end=spec.n_frames,
        frame_rate=spec.frame_rate,
        step_rate=spec.step_rate,
        gravity=tuple(spec.gravity),
    )
    built = BuiltScene(scene=scene, objects=[], bounds=[], floor=None)
    if simulate:
        built.simulator = _owned_pybullet()(scene, scratch_dir=f"{scratch_dir}/sim")
        if built.simulator.physics_client != DEFAULT_PHYSICS_CLIENT:
            raise RuntimeError(
                f"PyBullet client {DEFAULT_PHYSICS_CLIENT} is held by an earlier job; Kubric "
                "only drives the default client"
            )
    if render:
        from kubric.renderer.blender import Blender

        built.renderer = Blender(
            scene,
            scratch_dir=f"{scratch_dir}/render",
            samples_per_pixel=spec.spp,
            use_denoising=True,
        )
    for index, obj in enumerate(spec.objects, start=1):
        asset, bounds = _object(kb, obj, index)
        built.objects.append(asset)
        built.bounds.append(bounds)
    built.floor = _floor(kb, floor)
    scene += built.objects
    if render and spec.background == "hdri":
        _hdri_dome(kb, built, hdri_for(spec.seed, hdri_ids))
    else:
        scene += built.floor
    _lights_and_camera(kb, scene, spec)
    if built.simulator is not None:
        linear_damping_off(built.simulator, [*built.objects, built.floor])
    return built


@contextlib.contextmanager
def open_scene(
    spec: SceneSpec,
    floor: Mapping[str, float],
    *,
    simulate: bool = False,
    render: bool = False,
    hdri_ids: Sequence[str] = (),
) -> Iterator[BuiltScene]:
    """``build`` in a scratch directory; the PyBullet client is closed when the block ends."""
    with tempfile.TemporaryDirectory() as scratch:
        built = build(
            spec, floor, scratch_dir=scratch, simulate=simulate, render=render, hdri_ids=hdri_ids
        )
        try:
            yield built
        finally:
            release(built)


def release(built: BuiltScene) -> None:
    """Disconnects the job's PyBullet client so the next job in this process gets client 0."""
    if built.simulator is None:
        return
    import pybullet as pb

    client = built.simulator.physics_client
    if pb.isConnected(physicsClientId=client):
        pb.disconnect(physicsClientId=client)
    built.simulator = None


@functools.cache
def _owned_pybullet() -> type:
    """Kubric's PyBullet issues its calls to PyBullet's default client (id 0) and its
    ``__del__`` disconnects that default client whenever the garbage collector frees it. With
    several jobs in one container process a collected simulator would cut the next job's client,
    so ``release`` closes the client instead."""
    from kubric.simulator.pybullet import PyBullet

    class OwnedPyBullet(PyBullet):  # type: ignore[misc]
        def __del__(self) -> None:
            pass

    return OwnedPyBullet


def _object(kb: Any, obj: ObjectSpec, segmentation_id: int) -> tuple[Any, Any]:
    common: dict[str, Any] = {
        "name": obj.object_id,
        "position": tuple(obj.position),
        "quaternion": tuple(obj.quaternion),
        "velocity": tuple(obj.velocity),
        "angular_velocity": tuple(obj.angular_velocity),
        "mass": obj.mass,
        "friction": obj.friction,
        "restitution": obj.restitution,
        "static": obj.static,
        "material": kb.PrincipledBSDFMaterial(color=kb.Color(*obj.color)),
        "segmentation_id": segmentation_id,
    }
    if obj.primitive in (PrimitiveKind.BOX, PrimitiveKind.SPHERE):
        cls = kb.Cube if obj.primitive is PrimitiveKind.BOX else kb.Sphere
        return cls(scale=kubric_scale(obj.extents, *PRIMITIVE_BOUNDS), **common), PRIMITIVE_BOUNDS
    asset = kubasic_source().create(asset_id=obj.asset_id, **common)
    bounds = (tuple(asset.bounds[0]), tuple(asset.bounds[1]))
    asset.scale = kubric_scale(obj.extents, *bounds)
    return asset, bounds


@functools.lru_cache(maxsize=1)
def kubasic_source() -> Any:
    """The KuBasic manifest, read once per container process (about 3 s over ``gs://``)."""
    import kubric as kb

    return kb.AssetSource.from_manifest(KUBASIC_MANIFEST)


def _floor(kb: Any, floor: Mapping[str, float]) -> Any:
    """Static slab whose top face is the plane z = 0."""
    return kb.Cube(
        name=ENV_ID,
        scale=FLOOR_HALF_EXTENTS,
        position=(0.0, 0.0, -FLOOR_HALF_EXTENTS[2]),
        static=True,
        friction=float(floor["friction"]),
        restitution=float(floor["restitution"]),
        material=kb.PrincipledBSDFMaterial(color=kb.Color(*FLOOR_COLOR)),
        segmentation_id=BACKGROUND_SEGMENTATION,
    )


def hdri_for(seed: int, hdri_ids: Sequence[str]) -> str:
    """The scene's background: ``hdri_ids`` (the host's ``[dataset]`` list) indexed by seed."""
    if not hdri_ids:
        raise ValueError("background 'hdri' needs the job's hdri_ids")
    return str(hdri_ids[seed % len(hdri_ids)])


def _hdri_dome(kb: Any, built: BuiltScene, hdri_id: str) -> None:
    """HDRI lighting on the KuBasic dome (movi_def_worker recipe); the dome replaces the floor."""
    import bpy

    background = kb.AssetSource.from_manifest(HDRI_MANIFEST).create(asset_id=hdri_id)
    # Kubric has no public API for the ambient HDRI.
    built.renderer._set_ambient_light_hdri(background.filename)
    dome = kubasic_source().create(
        asset_id="dome",
        name="dome",
        static=True,
        background=True,
        segmentation_id=BACKGROUND_SEGMENTATION,
    )
    built.scene += dome
    texture = dome.linked_objects[built.renderer].data.materials[0].node_tree.nodes["Image Texture"]
    texture.image = bpy.data.images.load(background.filename)


def _lights_and_camera(kb: Any, scene: Any, spec: SceneSpec) -> None:
    target = tuple(spec.camera.look_at)
    sun = tuple(t + o for t, o in zip(target, SUN_OFFSET))
    scene += kb.DirectionalLight(name="sun", position=sun, look_at=target, intensity=SUN_INTENSITY)
    scene.ambient_illumination = kb.Color(*AMBIENT)
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        focal_length=spec.camera.focal_length_mm,
        sensor_width=spec.camera.sensor_width_mm,
        position=tuple(spec.camera.position),
        look_at=target,
    )


def linear_damping_off(simulator: Any, assets: Sequence[Any]) -> None:
    """PyBullet damps every body by 0.04 per second by default, which would bias measured g."""
    import pybullet as pb

    for asset in assets:
        body = asset.linked_objects.get(simulator)
        if body is not None:
            pb.changeDynamics(
                body,
                -1,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=simulator.physics_client,
            )


def body_ids(built: BuiltScene) -> tuple[list[int], int]:
    """PyBullet body ids of the objects (SceneSpec order) and of the floor."""
    sim = built.simulator
    return [asset.linked_objects[sim] for asset in built.objects], built.floor.linked_objects[sim]


def clip_inputs(job: Mapping[str, Any]) -> tuple[SceneSpec, ObjectStateTable]:
    """Scene spec and variant states of a clip job, checked to list the same objects in order."""
    spec = SceneSpec.load_json(job["spec_path"])
    states = ObjectStateTable.load_npz(job["states_path"])
    ids = [o.object_id for o in spec.objects]
    if [str(i) for i in states.object_ids] != ids:
        raise ValueError(f"state objects {list(states.object_ids)} differ from the spec {ids}")
    return spec, states


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_json(path: Path, payload: Any) -> None:
    tmp = _temporary(path)
    tmp.write_text(dumps(payload), encoding="utf-8")
    tmp.replace(path)


def save_record(record: Record, path: Path) -> None:
    tmp = _temporary(path)
    if path.suffix == ".npz":
        record.save_npz(tmp)
    else:
        record.save_json(tmp)
    tmp.replace(path)


def save_arrays(path: Path, arrays: dict[str, Any]) -> None:
    tmp = _temporary(path)
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def guarded(job: dict[str, Any], work: Callable[[dict[str, Any]], None]) -> bool:
    """Runs one job; an error lands in ``failed_path`` so the remaining jobs still run."""
    try:
        work(job)
    # Every failure is reported per job.
    except Exception as exc:
        write_json(
            Path(job["failed_path"]),
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        return False
    return True
