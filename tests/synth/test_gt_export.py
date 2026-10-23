"""Ground-truth export: Kubric camera conversion, visibility and the round trip through the
``kubric_synthetic`` dataset, on a real simulation with a rasterized stand-in for the render."""

from __future__ import annotations

import dataclasses
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.core.video import Mp4VideoSource
from physverify.datasets.kubric_synthetic import KubricSynthetic, load_native
from physverify.schema.contacts import ContactTable
from physverify.schema.render import RenderLayers
from physverify.schema.scene import CameraSpec, SceneSpec
from physverify.schema.states import CoordFrame, ObjectStateTable
from physverify.synth.generate import plan_subset, save_plan
from physverify.synth.gt_camera import (
    OPENGL_TO_OPENCV,
    opencv_intrinsics,
    opencv_world_from_camera,
    visibility,
)
from physverify.synth.gt_export import content_sha256, export_all, export_clip
from physverify.synth.perturb import (
    apply_variants,
    closest_points,
    ground_truth_contacts,
    instantiate,
)
from physverify.synth.perturbations._frame_windows import (
    projected_silhouette,
    silhouette_mask,
    solid_at,
)
from physverify.synth.placement import project, spec_intrinsics, spec_world_from_camera
from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import DatasetSettings, dataset_root, load_dataset_settings
from physverify.synth.store import (
    CAMERA,
    CLOSEST,
    CONTACTS_WORLD,
    GT_META,
    KUBRIC_META,
    LAYERS,
    RECORD,
    VARIANT_STATES,
    VIDEO,
    DatasetStore,
    Step,
    atomic_save,
    atomic_write_json,
    read_json,
)

DATA = Path(__file__).parent / "data"
SIM_ID = "f2_bounce-s000"
FAR_Z = 1e5


def kubric_k_norm(camera: CameraSpec, width: int, height: int) -> np.ndarray:
    """``PerspectiveCamera.intrinsics`` of Kubric 2022.4 (normalized image, OpenGL axes)."""
    f_x = camera.focal_length_mm / camera.sensor_width_mm
    f_y = camera.focal_length_mm / (camera.sensor_width_mm / width * height)
    return np.array([[f_x, 0.0, -0.5], [0.0, -f_y, -0.5], [0.0, 0.0, -1.0]])


def kubric_project(k_norm: np.ndarray, matrix_world: np.ndarray, point: np.ndarray) -> np.ndarray:
    """``Camera.project_point`` of Kubric 2022.4 in float64: normalized (u, v) in [0, 1]."""
    projected = k_norm @ (np.linalg.inv(matrix_world) @ np.append(point, 1.0))[:3]
    return projected[:2] / projected[2]


def scene_camera() -> tuple[SceneSpec, np.ndarray, np.ndarray]:
    scene = SceneSpec.load_json(DATA / "f2_bounce" / "spec.json")
    width, height = scene.resolution
    matrix_world = spec_world_from_camera(scene.camera) @ OPENGL_TO_OPENCV
    return scene, kubric_k_norm(scene.camera, width, height), matrix_world


def test_opencv_camera_matches_kubric_projection() -> None:
    scene, k_norm, matrix_world = scene_camera()
    width, height = scene.resolution
    K = opencv_intrinsics(k_norm, width, height)  # noqa: N806
    world_from_camera = opencv_world_from_camera(matrix_world)
    assert pytest.approx(spec_intrinsics(scene.camera, (width, height)), abs=1e-9) == K
    points = np.asarray(scene.camera.look_at) + np.random.default_rng(0).uniform(-0.3, 0.3, (50, 3))
    uv, z = project(points, K, world_from_camera)
    expected = np.array([kubric_project(k_norm, matrix_world, p) for p in points])
    assert np.all(z > 0.0)
    assert np.max(np.abs(uv - expected * [width, height])) < 1e-9 * width


def raster(scene: SceneSpec, states: ObjectStateTable) -> tuple[np.ndarray, np.ndarray]:
    """Segmentation and z of the objects' convex silhouettes, each drawn at its centre depth."""
    width, height = scene.resolution
    K = spec_intrinsics(scene.camera, (width, height))  # noqa: N806
    pose = spec_world_from_camera(scene.camera)
    seg = np.zeros((states.n_frames, height, width), dtype=np.uint32)
    z = np.full(seg.shape, FAR_Z, dtype=np.float32)
    for t in range(states.n_frames):
        for k in range(states.n_objects):
            mask = silhouette_mask(
                projected_silhouette(K, pose, solid_at(states, k, t)), (width, height)
            )
            depth = float((states.position[t, k] - pose[:3, 3]) @ pose[:3, 2])
            drawn = mask & (z[t] > depth)
            seg[t][drawn] = k + 1
            z[t][drawn] = depth
    return seg, z


def fake_render(store: DatasetStore, clip_id: str, scene: SceneSpec) -> None:
    clip_dir = store.clip_dir(clip_id)
    states = ObjectStateTable.load_npz(clip_dir / VARIANT_STATES)
    seg, z = raster(scene, states)
    width, height = scene.resolution
    layers = RenderLayers(
        clip_id=clip_id,
        frame_indices=np.arange(states.n_frames, dtype=np.int64),
        rgba=np.zeros((states.n_frames, height, width, 4), dtype=np.uint8),
        segmentation=seg,
        z=z,
        object_ids=np.asarray([o.object_id for o in scene.objects], dtype=str),
    )
    _, k_norm, matrix_world = scene_camera()
    atomic_save(layers, clip_dir / LAYERS)
    atomic_write_json(
        clip_dir / KUBRIC_META,
        {
            "clip_id": clip_id,
            "frame_seconds": [0.0] * states.n_frames,
            "postprocess_seconds": 0.0,
            "k_norm": k_norm.tolist(),
            "matrix_world": matrix_world.tolist(),
        },
    )


def test_occluder_in_front_hides_and_behind_does_not() -> None:
    scene, k_norm, matrix_world = scene_camera()
    states = ObjectStateTable.load_npz(DATA / "f2_bounce" / "states.npz")
    width, height = scene.resolution
    K = opencv_intrinsics(k_norm, width, height)  # noqa: N806
    world_from_camera = opencv_world_from_camera(matrix_world)
    seg, z = raster(scene, states)
    layers = RenderLayers(
        clip_id="x",
        frame_indices=np.arange(states.n_frames, dtype=np.int64),
        rgba=np.zeros((*seg.shape, 4), dtype=np.uint8),
        segmentation=seg,
        z=z,
        object_ids=np.asarray(["ball_0", "cube_0", "wall"], dtype=str),
    )
    clear = visibility(layers, states, K, world_from_camera)
    assert clear.visible.all() and clear.in_frame.all()
    assert np.nanmax(clear.occlusion) < 0.5
    wall = np.full(seg.shape, 3, dtype=np.uint32)
    front = visibility(
        dataclasses.replace(layers, segmentation=wall, z=np.full(z.shape, 1e-3, np.float32)),
        states,
        K,
        world_from_camera,
    )
    assert not front.visible.any()
    assert front.occlusion == pytest.approx(np.ones_like(front.occlusion))
    behind = visibility(
        dataclasses.replace(layers, segmentation=wall, z=np.full(z.shape, FAR_Z, np.float32)),
        states,
        K,
        world_from_camera,
    )
    assert behind.occlusion == pytest.approx(np.zeros_like(behind.occlusion))


class CountingRunner:
    """Stands in for the container: counts the jobs of each task and writes nothing."""

    def __init__(self) -> None:
        self.jobs: dict[str, int] = {}

    def container_path(self, host_path: Path) -> str:
        return str(host_path)

    def run(self, task: str, jobs: Path) -> Any:
        self.jobs[task] = self.jobs.get(task, 0) + len(jobs.read_text().splitlines())
        return SimpleNamespace(returncode=0)


@dataclasses.dataclass
class Smoke:
    plan: DatasetPlan
    store: DatasetStore
    plugins: dict[str, Any]
    ds: DatasetSettings

    def clip(self, perturbation: str) -> str:
        return next(v.clip_id for v in self.plan.variants if v.perturbation == perturbation)


def applied_smoke(data_settings: Settings, families: Any, perturbations: Any) -> Smoke:
    """The smoke subset on the recorded f2 simulation, after APPLY."""
    ds = load_dataset_settings(data_settings)
    store = DatasetStore(dataset_root(data_settings, ds))
    plan = plan_subset(ds, families, perturbations, "smoke")
    save_plan(store, plan)
    store.scene_dir(SIM_ID).mkdir(parents=True)
    for path in (DATA / "f2_bounce").iterdir():
        shutil.copy(path, store.scene_dir(SIM_ID) / path.name)
    plugins = instantiate(perturbations, ds)
    apply_variants(plan, store, plugins, ds)
    return Smoke(plan, store, plugins, ds)


def exported_smoke(data_settings: Settings, families: Any, perturbations: Any) -> Smoke:
    """The smoke subset exported, with empty closest-point samples and a rasterized render."""
    smoke = applied_smoke(data_settings, families, perturbations)
    plan, store, ds = smoke.plan, smoke.store, smoke.ds
    for variant in plan.variants:
        closest = store.clip_dir(variant.clip_id) / CLOSEST
        if not (store.clip_dir(variant.clip_id) / CONTACTS_WORLD).is_file():
            np.savez(
                closest,
                frame=np.zeros(0, np.int64),
                a=np.zeros(0, str),
                b=np.zeros(0, str),
                point_a=np.zeros((0, 3)),
                point_b=np.zeros((0, 3)),
                normal_on_b=np.zeros((0, 3)),
                distance=np.zeros(0),
            )
    ground_truth_contacts(plan, store, ds)
    scene = SceneSpec.load_json(store.scene_dir(SIM_ID) / "spec.json")
    for variant in plan.variants:
        fake_render(store, variant.clip_id, scene)
    truth_measures = PLUGIN_KINDS["truth_measures"]()
    exported = export_all(
        plan, store, smoke.plugins, truth_measures, ds, data_settings.paths.data_dir
    )
    assert sorted(exported) == sorted(v.clip_id for v in plan.variants)
    return smoke


def test_exported_clips_round_trip(
    data_settings: Settings, families: Any, perturbations: Any
) -> None:
    smoke = exported_smoke(data_settings, families, perturbations)
    plan, store, ds = smoke.plan, smoke.store, smoke.ds
    exported = [v.clip_id for v in plan.variants]
    scene = SceneSpec.load_json(store.scene_dir(SIM_ID) / "spec.json")

    dataset = KubricSynthetic(data_settings)
    clips = dataset.clips("smoke")
    assert [c.clip_id for c in clips] == sorted(exported)
    for clip in clips:
        clip_dir = store.clip_dir(clip.clip_id)
        meta = read_json(clip_dir / GT_META)
        camera = read_json(clip_dir / CAMERA)
        gt = dataset.ground_truth(clip)
        assert gt.version == meta["gt_version"]
        assert dataset.video(clip).n_frames == scene.n_frames
        video = Mp4VideoSource(clip_dir / VIDEO, data_settings.runtime.max_frame_bytes)
        assert video.n_frames == scene.n_frames
        assert video.native_size == tuple(scene.resolution)
        assert np.asarray(camera["K"]) == pytest.approx(
            spec_intrinsics(scene.camera, tuple(scene.resolution)), abs=1e-9
        )
        world = gt.states(CoordFrame.WORLD_ZUP)
        variant = ObjectStateTable.load_npz(clip_dir / VARIANT_STATES)
        assert np.array_equal(world.position, variant.position, equal_nan=True)
        pose = np.asarray(camera["world_from_camera"])
        in_camera = gt.states(CoordFrame.CAMERA_OPENCV).position
        assert in_camera == pytest.approx((world.position - pose[:3, 3]) @ pose[:3, :3])
        assert all(e <= 1.0 for e in meta["e_clean"].values())
        target = next(v.target_magnitude for v in plan.variants if v.clip_id == clip.clip_id)
        assert clip.perturbation is not None
        assert clip.perturbation.magnitude == pytest.approx(target, rel=1e-3)
    native = load_native(store, clips[0].clip_id, ds.render.far_z_m)
    assert float(native.z.max()) < ds.render.far_z_m
    assert bool((native.z == 0.0).any())


def test_npz_content_hash_ignores_the_zip_container(tmp_path: Path) -> None:
    payload = np.arange(6.0).tobytes()
    containers = (
        ("a.npz", (1980, 1, 1, 0, 0, 0), zipfile.ZIP_STORED),
        ("b.npz", (2026, 9, 24, 12, 0, 0), zipfile.ZIP_DEFLATED),
    )
    for name, stamp, compression in containers:
        with zipfile.ZipFile(tmp_path / name, "w", compression) as archive:
            archive.writestr(zipfile.ZipInfo("x.npy", date_time=stamp), payload, compression)
    assert (tmp_path / "a.npz").read_bytes() != (tmp_path / "b.npz").read_bytes()
    assert content_sha256(tmp_path / "a.npz") == content_sha256(tmp_path / "b.npz")
    with zipfile.ZipFile(tmp_path / "c.npz", "w") as archive:
        archive.writestr("x.npy", payload[:-1] + b"\x01")
    assert content_sha256(tmp_path / "c.npz") != content_sha256(tmp_path / "a.npz")


def test_gt_version_follows_the_exported_contacts(
    data_settings: Settings, families: Any, perturbations: Any, truth_measures: Any
) -> None:
    smoke = exported_smoke(data_settings, families, perturbations)
    store, ds = smoke.store, smoke.ds
    clip_id = smoke.clip("none")
    variant = smoke.plan.variant(clip_id)
    clip_dir = store.clip_dir(clip_id)
    plugin = smoke.plugins[variant.perturbation]

    def export() -> tuple[str, dict[str, Any], str]:
        export_clip(variant, store, plugin, truth_measures, ds, data_settings.paths.data_dir)
        meta = read_json(clip_dir / GT_META)
        return meta["gt_version"], meta["record"], content_sha256(clip_dir / LAYERS)

    first = export()
    assert export() == first
    contacts = ContactTable.load_json(clip_dir / CONTACTS_WORLD)
    assert len(contacts.records) > 1
    fewer = dataclasses.replace(contacts, records=contacts.records[:-1])
    atomic_save(fewer, clip_dir / CONTACTS_WORLD)
    version, record, layers = export()
    assert (record, layers) == first[1:]
    assert version != first[0]
    dataset = KubricSynthetic(data_settings)
    clip = next(c for c in dataset.clips("smoke") if c.clip_id == clip_id)
    gt = dataset.ground_truth(clip)
    assert gt.version == version
    assert len(gt.contacts(CoordFrame.WORLD_ZUP).records) == len(fewer.records)


def test_apply_reruns_when_its_contact_table_is_gone(
    data_settings: Settings, families: Any, perturbations: Any
) -> None:
    """A clip whose contact table APPLY wrote (poses not edited) loses it: APPLY writes it again,
    keeps the render (same states) and drops the export, and no closest-point job is launched for
    it. A re-applied clip whose states changed also loses its render."""
    smoke = applied_smoke(data_settings, families, perturbations)
    plan, store, ds = smoke.plan, smoke.store, smoke.ds
    unedited, edited = smoke.clip("none"), smoke.clip("r3_restitution_gain")
    for clip_id in (unedited, edited):
        for name in (KUBRIC_META, LAYERS, GT_META):
            atomic_write_json(store.clip_dir(clip_id) / name, {"fake": True})
    table = store.clip_dir(unedited) / CONTACTS_WORLD
    before = table.read_bytes()
    table.unlink()
    rendered = store.clip_dir(edited) / VARIANT_STATES
    old = ObjectStateTable.load_npz(rendered)
    atomic_save(dataclasses.replace(old, position=old.position + 0.01), rendered)
    (store.clip_dir(edited) / RECORD).unlink()

    report = apply_variants(plan, store, smoke.plugins, ds)
    assert report.applied == tuple(
        v.clip_id for v in plan.variants if v.clip_id in (unedited, edited)
    )
    assert report.discarded_renders == (edited,)
    assert table.read_bytes() == before
    assert store.is_done(Step.RENDER, unedited) and not store.is_done(Step.EXPORT, unedited)
    assert not store.is_done(Step.RENDER, edited) and not (store.clip_dir(edited) / LAYERS).exists()
    runner = CountingRunner()
    assert closest_points(plan, store, runner, ds) == 1
    assert runner.jobs == {"contacts": 1}
    assert not store.is_done(Step.PERTURB, edited)
