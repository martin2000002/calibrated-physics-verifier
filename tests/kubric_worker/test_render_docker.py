"""The simulate, contacts and render tasks in the Kubric image on a small real
scene. Run with ``pytest -m docker`` under the heavy lock."""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.loader import load_settings
from physverify.runtime.docker import DockerRunner
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable, ManifoldKind, RawCollisions
from physverify.schema.render import RenderLayers
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.gt_camera import opencv_intrinsics, opencv_world_from_camera
from physverify.synth.gt_contacts import ClosestSamples, from_raw_events
from physverify.synth.placement import project
from physverify.synth.settings import DatasetSettings, load_dataset_settings
from physverify.synth.store import DatasetStore, read_json

pytestmark = pytest.mark.docker

SPEC = Path(__file__).parents[1] / "synth" / "data" / "clean" / "f2_bounce" / "spec.json"
WIDTH, HEIGHT = 96, 64
HIDDEN = (3, 4, 5)
SCALED = (6, 7, 8)
GROWTH = 0.3
UP = np.array([0.0, 0.0, 1.0])


@pytest.fixture
def container(
    tmp_path: Path, repo_root: Path, config_dir: Path, environ: dict[str, str]
) -> tuple[DockerRunner, DatasetStore, DatasetSettings]:
    settings = load_settings(config_dir, environ=environ)
    image = settings.runtime.docker_image
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is not installed")
    if (
        subprocess.run(
            [docker, "image", "inspect", image], capture_output=True, check=False
        ).returncode
        != 0
    ):
        pytest.skip(f"image {image} is not pulled")
    data_dir = tmp_path / "data"
    runner = DockerRunner(
        image=image,
        src_dir=repo_root / "src",
        data_dir=data_dir,
        threads=2,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    return runner, DatasetStore(data_dir / "synthetic"), load_dataset_settings(settings)


def small_scene(root: Path) -> tuple[SceneSpec, Path]:
    scene = dataclasses.replace(
        SceneSpec.load_json(SPEC), n_frames=10, resolution=(WIDTH, HEIGHT), spp=4
    )
    path = root / "spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.save_json(path)
    return scene, path


def run_task(
    runner: DockerRunner, store: DatasetStore, task: str, jobs: list[dict[str, Any]]
) -> None:
    assert runner.run(task, store.write_jobs(task, jobs)).returncode == 0


def test_tasks_in_one_container_process(container: Any) -> None:
    runner, store, ds = container
    floor = ds.floor.model_dump()
    scene, spec_path = small_scene(store.root / "scene")
    sims = [store.root / name for name in ("sim_a", "sim_b")]
    run_task(
        runner,
        store,
        "simulate",
        [
            {
                "sim_id": d.name,
                "spec_path": runner.container_path(spec_path),
                "states_path": runner.container_path(d / "states.npz"),
                "raw_path": runner.container_path(d / "raw_collisions.npz"),
                "meta_path": runner.container_path(d / "sim_meta.json"),
                "failed_path": runner.container_path(d / "failed.json"),
                "floor": floor,
            }
            for d in sims
        ],
    )
    for d in sims:
        assert not (d / "failed.json").exists(), read_json(d / "failed.json")
    states = ObjectStateTable.load_npz(sims[0] / "states.npz")
    assert np.array_equal(
        states.position, ObjectStateTable.load_npz(sims[1] / "states.npz").position
    )
    raw = RawCollisions.load_npz(sims[0] / "raw_collisions.npz")
    again = RawCollisions.load_npz(sims[1] / "raw_collisions.npz")
    assert raw.frame.size > 0
    assert np.array_equal(raw.frame, again.frame)
    table = from_raw_events(raw, scene, states, ds.contacts)
    assert_impact_timing(scene, states, table)
    assert_cube_rests_on_the_floor(scene, states, table)

    clip, edited = store.root / "clip", store.root / "clip_edit"
    edited.mkdir(parents=True)
    edit_ball(states, ball_index(scene)).save_npz(edited / "states_variant.npz")

    def job(directory: Path, states_path: Path, **extra: Any) -> dict[str, Any]:
        paths = {"states_path": states_path, "failed_path": directory / "failed.json", **extra}
        return {
            "clip_id": directory.name,
            "spec_path": runner.container_path(spec_path),
            "floor": floor,
            **{k: runner.container_path(v) if isinstance(v, Path) else v for k, v in paths.items()},
        }

    run_task(
        runner,
        store,
        "contacts",
        [job(clip, sims[0] / "states.npz", out_path=clip / "closest.npz", tol=1e-3)],
    )
    # The edited clip renders first so that anything it leaves behind shows up in the plain one.
    run_task(
        runner,
        store,
        "render",
        [
            job(d, s, layers_path=d / "layers.npz", meta_path=d / "kubric_meta.json")
            for d, s in ((edited, edited / "states_variant.npz"), (clip, sims[0] / "states.npz"))
        ],
    )
    for d in (clip, edited):
        assert not (d / "failed.json").exists(), read_json(d / "failed.json")

    closest = ClosestSamples.load(clip / "closest.npz")
    cube_rows = (closest.a == "cube_0") & (closest.distance <= 1e-3)
    assert set(closest.frame[cube_rows].tolist()) == set(range(scene.n_frames))

    layers = RenderLayers.load_npz(clip / "layers.npz")
    assert layers.rgba.shape == (scene.n_frames, HEIGHT, WIDTH, 4)
    assert set(np.unique(layers.segmentation).tolist()) <= {0, 1, 2}
    assert 1 in layers.segmentation and 2 in layers.segmentation
    meta = read_json(clip / "kubric_meta.json")
    assert meta["render_threads"] == runner.threads
    K = opencv_intrinsics(meta["k_norm"], WIDTH, HEIGHT)  # noqa: N806
    world_from_camera = opencv_world_from_camera(meta["matrix_world"])
    uv, _ = project(meta["probe_points"], K, world_from_camera)
    kubric = np.asarray(meta["probe_projections"])[:, :2] * [WIDTH, HEIGHT]
    assert np.max(np.abs(uv - kubric)) < 1e-3

    ball = ball_index(scene)
    centres, _ = project(states.position[:, ball], K, world_from_camera)
    assert_mask_centred_on_the_projection(ball_masks(layers, ball), centres)
    assert_edits_render(
        ball_masks(RenderLayers.load_npz(edited / "layers.npz"), ball), ball_masks(layers, ball)
    )


def ball_index(scene: SceneSpec) -> int:
    return [o.object_id for o in scene.objects].index("ball_0")


def edit_ball(states: ObjectStateTable, ball: int) -> ObjectStateTable:
    """The ball hidden in ``HIDDEN`` and grown by ``1 + GROWTH`` in ``SCALED``."""
    visible = states.visible.copy()
    visible[list(HIDDEN), ball] = False
    size = states.size.copy()
    size[list(SCALED), ball] *= 1.0 + GROWTH
    return dataclasses.replace(states, visible=visible, size=size)


def ball_masks(layers: RenderLayers, ball: int) -> Any:
    return layers.segmentation == ball + 1


def inside(mask: Any) -> bool:
    """Some pixels, none on the image border (a clipped mask has a biased centroid and area)."""
    rows, cols = np.nonzero(mask)
    return bool(
        rows.size >= 5
        and rows.min() > 0
        and cols.min() > 0
        and rows.max() < mask.shape[0] - 1
        and cols.max() < mask.shape[1] - 1
    )


def assert_mask_centred_on_the_projection(masks: Any, centres: Any) -> None:
    """(2) The ball's mask centroid (pixel centres at +0.5) is within 1 px of its projected
    centre."""
    errors = []
    for mask, centre in zip(masks, centres, strict=True):
        if inside(mask):
            rows, cols = np.nonzero(mask)
            errors.append(
                float(np.hypot(cols.mean() + 0.5 - centre[0], rows.mean() + 0.5 - centre[1]))
            )
    assert errors and max(errors) < 1.0, errors


def assert_edits_render(edited: Any, plain: Any) -> None:
    """(4) The hidden pose leaves no pixel; (5) the grown ball covers (1 + m)^2 of the plain area
    within 15 %; (6) the plain clip, rendered after it in the same process, shows neither edit."""
    for t in HIDDEN:
        assert edited[t].sum() == 0
        assert plain[t].sum() > 0
    ratios = [
        edited[t].sum() / plain[t].sum() for t in SCALED if inside(edited[t]) and inside(plain[t])
    ]
    assert ratios and all(abs(r / (1.0 + GROWTH) ** 2 - 1.0) <= 0.15 for r in ratios), ratios
    untouched = [t for t in range(len(plain)) if t not in HIDDEN + SCALED]
    assert all(np.array_equal(edited[t], plain[t]) for t in untouched)


def assert_cube_rests_on_the_floor(
    scene: SceneSpec, states: ObjectStateTable, table: ContactTable
) -> None:
    """(7) The dynamic cube stays at z = h/2 within 2 mm; (8) its floor contact is one resting
    record with a normal within 5 degrees of +z, measured with >= 3 points or a primitive
    fallback that says why."""
    cube = [o.object_id for o in scene.objects].index("cube_0")
    half = scene.objects[cube].extents[2] / 2.0
    assert np.max(np.abs(states.position[:, cube, 2] - half)) < 2e-3
    (record,) = [r for r in table.records if (r.object_a, r.object_b) == ("cube_0", ENV_ID)]
    assert record.kind is ContactKind.RESTING
    assert math.degrees(math.acos(min(1.0, float(record.normal @ UP)))) < 5.0
    if record.manifold is ManifoldKind.MEASURED:
        assert record.points.shape[0] >= 3
    else:
        assert record.manifold is ManifoldKind.PRIMITIVE
        assert record.source.startswith("primitive:sparse_")


def assert_impact_timing(scene: SceneSpec, states: ObjectStateTable, table: ContactTable) -> None:
    """The first floor impact of the ball, read from the events, lands within 0.1 frame of the
    ballistic extrapolation of the simulated flight."""
    impacts = [r.t_start for r in table.records if r.kind is ContactKind.IMPACT]
    g, radius = -scene.gravity[2], scene.objects[0].extents[2] / 2.0
    z, vz = states.position[:, 0, 2], states.velocity[:, 0, 2]
    k = int(np.flatnonzero((vz[:-1] < 0.0) & (vz[1:] > 0.0))[0])
    tau = (vz[k] + math.sqrt(vz[k] ** 2 + 2.0 * g * (z[k] - radius))) / g
    assert impacts[0] == pytest.approx(k + tau * scene.frame_rate, abs=0.1)
