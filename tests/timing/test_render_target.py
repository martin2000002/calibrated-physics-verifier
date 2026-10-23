"""The Kubric render target: one clip per container in a scratch store, no Docker in the tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from records import perturbation, scene

from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError, TimingError, UsageError
from physverify.schema.clip import ClipManifest, Split
from physverify.synth.generate import save_plan
from physverify.synth.gt.export import DATASET_NAME
from physverify.synth.plan import DatasetPlan, ScenePlan, VariantPlan
from physverify.synth.settings import dataset_root, load_dataset_settings
from physverify.synth.store import (
    KUBRIC_META,
    RECORD,
    SPEC,
    VARIANT_STATES,
    DatasetStore,
)
from physverify.timing._base import TimingContext
from physverify.timing.records import RENDER_PER_FRAME_S, RENDER_WALL_S, check_measures
from physverify.timing.report import SCRATCH_DIR, timing_dir
from physverify.timing.target_services import TimingServices
from physverify.timing.targets.kubric_render import (
    FRAMES_S,
    OVERHEAD_S,
    POSTPROCESS_S,
    KubricRender,
    render_measures,
)

SCENE = "s0"
CLIP_IDS = ("v0", "v1", "v2", "v3", "v4", "v5")
SKIPPED = "v1"
NOT_APPLIED = "v2"


@dataclass
class FakeRunner:
    """Plays the render worker: writes the meta (or a failure) where each job says."""

    clock: Any
    frames: tuple[float, ...] = (0.5, 0.25, 0.25)
    postprocess: float = 0.125
    startup: float = 2.0
    returncode: int = 0
    failure: str | None = None
    write_meta: bool = True
    outside: Path | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)
    inputs_present: list[bool] = field(default_factory=list)

    def container_path(self, host_path: Path) -> str:
        if self.outside is not None and host_path.is_relative_to(self.outside):
            raise UsageError(f"{host_path} is outside the data directory")
        return str(host_path)

    def run(self, task: str, jobs: Path) -> Any:
        for line in jobs.read_text().splitlines():
            job = json.loads(line)
            self.jobs.append(job)
            self.inputs_present.append(
                Path(job["spec_path"]).is_file() and Path(job["states_path"]).is_file()
            )
            self._answer(job)
        self.clock.advance(self.startup + sum(self.frames) + self.postprocess)
        return SimpleNamespace(returncode=self.returncode)

    def _answer(self, job: dict[str, Any]) -> None:
        if self.failure is not None:
            Path(job["failed_path"]).write_text(json.dumps({"error": self.failure}))
        elif self.write_meta:
            meta = {
                "clip_id": job["clip_id"],
                "frame_seconds": list(self.frames),
                "postprocess_seconds": self.postprocess,
            }
            Path(job["meta_path"]).write_text(json.dumps(meta))


def variant(clip_id: str, index: int) -> VariantPlan:
    return VariantPlan(
        clip_id=clip_id,
        scene_id=SCENE,
        sim_id=SCENE,
        family="f1_free_fall",
        index=index,
        perturbation="gravity_scale",
        level=2,
        target_magnitude=1.5,
        split=Split.EVAL,
    )


def write_dataset(settings: Settings, *, applied: bool = True) -> DatasetStore:
    ds = load_dataset_settings(settings)
    store = DatasetStore(dataset_root(settings, ds))
    variants = tuple(variant(clip_id, index) for index, clip_id in enumerate(CLIP_IDS))
    scenes = (ScenePlan(SCENE, "f1_free_fall", 0, Split.EVAL),)
    save_plan(store, DatasetPlan(ds.version, settings.config_hash(), scenes, variants, {}))
    store.scene_dir(SCENE).mkdir(parents=True)
    scene().save_json(store.scene_dir(SCENE) / SPEC)
    for clip_id in CLIP_IDS:
        clip_dir = store.clip_dir(clip_id)
        clip_dir.mkdir(parents=True)
        if applied and clip_id != NOT_APPLIED:
            perturbation().save_json(clip_dir / RECORD)
            (clip_dir / VARIANT_STATES).write_bytes(b"states")
    store.mark_skipped(SKIPPED, "no contact")
    return store


@dataclass
class Setup:
    settings: Settings
    runner: FakeRunner
    services: TimingServices
    ctx: TimingContext
    dockers: list[str] = field(default_factory=list)

    def target(self, **changes: Any) -> KubricRender:
        options: dict[str, Any] = {
            "which": lambda name: "/usr/bin/docker",
            "image_check": lambda image: True,
        }
        return KubricRender(self.services, **{**options, **changes})


@pytest.fixture
def setup(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    fake_clock: Callable[..., Any],
    tmp_path: Path,
) -> Setup:
    settings = make_settings()
    runner = FakeRunner(fake_clock())
    dockers: list[str] = []

    def docker() -> FakeRunner:
        dockers.append("docker")
        return runner

    services = make_services(clock=runner.clock, docker=docker)
    ctx = TimingContext(settings, "cpu", 1, settings.paths.data_dir / "runs" / "r0")
    return Setup(settings, runner, services, ctx, dockers)


def refuse(name: str) -> bool:
    raise AssertionError("checked out of order")


def test_unavailable_reasons_come_in_order(setup: Setup) -> None:
    missing_docker = setup.target(which=lambda name: None, image_check=refuse)
    assert missing_docker.unavailable(setup.ctx) == "docker is not on PATH"
    missing_image = setup.target(image_check=lambda image: False)
    assert "physverify synth pull-image" in str(missing_image.unavailable(setup.ctx))
    assert "physverify synth generate" in str(setup.target().unavailable(setup.ctx))
    write_dataset(setup.settings, applied=False)
    assert "physverify synth perturb" in str(setup.target().unavailable(setup.ctx))


def test_unavailable_when_the_run_dir_cannot_be_mounted(setup: Setup) -> None:
    write_dataset(setup.settings)
    assert setup.target().unavailable(setup.ctx) is None
    setup.runner.outside = setup.ctx.run_dir
    reason = setup.target().unavailable(setup.ctx)
    assert reason is not None and reason.startswith("the run directory cannot be mounted")


def test_own_clips_are_applied_variants_in_plan_order_up_to_max_clips(
    setup: Setup, make_settings: Callable[..., Settings]
) -> None:
    write_dataset(setup.settings)
    clips = setup.target().own_clips(setup.ctx)
    assert [clip.clip_id for clip in clips] == ["v0", "v3", "v4"]
    assert {clip.dataset for clip in clips} == {DATASET_NAME}
    assert clips[0].n_frames == scene().n_frames
    assert clips[0].perturbation == perturbation()
    limited = make_settings("timing.targets.kubric_render.max_clips=1")
    ctx = TimingContext(limited, "cpu", 1, setup.ctx.run_dir)
    assert [clip.clip_id for clip in setup.target().own_clips(ctx)] == ["v0"]


def test_run_renders_one_clip_in_its_own_container(
    setup: Setup, clip_factory: Callable[..., ClipManifest]
) -> None:
    store = write_dataset(setup.settings)
    target = setup.target()
    measures = target.run(clip_factory("v3"), setup.ctx)
    check_measures(measures)
    assert measures == {
        RENDER_WALL_S: 3.125,
        FRAMES_S: 1.0,
        RENDER_PER_FRAME_S: 1.0 / 3,
        POSTPROCESS_S: 0.125,
        OVERHEAD_S: 2.0,
    }
    [job] = setup.runner.jobs
    assert job["clip_id"] == "v3"
    assert Path(job["meta_path"]).is_relative_to(
        timing_dir(setup.ctx.run_dir) / SCRATCH_DIR / "render-0"
    )
    assert job["floor"] == load_dataset_settings(setup.settings).floor.model_dump()
    assert setup.runner.inputs_present == [True]
    assert not (timing_dir(setup.ctx.run_dir) / SCRATCH_DIR / "render-0").exists()
    assert not (store.clip_dir("v3") / KUBRIC_META).exists()
    target.run(clip_factory("v3"), setup.ctx)
    assert Path(setup.runner.jobs[1]["meta_path"]).is_relative_to(
        timing_dir(setup.ctx.run_dir) / SCRATCH_DIR / "render-1"
    )
    assert setup.dockers == ["docker"]
    target.close()
    target.run(clip_factory("v3"), setup.ctx)
    assert setup.dockers == ["docker", "docker"]


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"failure": "blender crashed"}, TimingError, "v0: render failed: .*blender crashed"),
        ({"write_meta": False}, TimingError, f"v0: the render left no {KUBRIC_META}"),
        ({"returncode": 3}, PhysverifyError, "exited with 3"),
    ],
)
def test_failed_renders_raise_and_remove_the_scratch(
    setup: Setup,
    clip_factory: Callable[..., ClipManifest],
    change: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    write_dataset(setup.settings)
    vars(setup.runner).update(change)
    with pytest.raises(error, match=message):
        setup.target().run(clip_factory("v0"), setup.ctx)
    assert not (timing_dir(setup.ctx.run_dir) / SCRATCH_DIR).exists()


def test_describe_names_the_image(setup: Setup) -> None:
    assert setup.target().describe(setup.ctx) == {
        "image": setup.settings.runtime.docker_image,
        "jobs_per_container": "1",
    }


def test_render_measures_need_frame_times() -> None:
    with pytest.raises(TimingError, match="no frame times"):
        render_measures({"clip_id": "v0", "frame_seconds": [], "postprocess_seconds": 0.0}, 1.0)
