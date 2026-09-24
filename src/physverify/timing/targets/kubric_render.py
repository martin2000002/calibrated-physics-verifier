"""Per-clip cost of A4's Kubric render, one clip per container in a scratch dataset store.

A4 batches several clips per container; here each call renders a single clip so the container
start-up is charged to that clip (``render.overhead_s``). The frame and post-processing times are
the worker's own (``kubric_meta.json``); ``render.wall_s`` is measured on the host.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from itertools import count
from pathlib import Path
from statistics import fmean
from typing import Any, ClassVar, Final

from physverify.cli.commands.dataset import image_present
from physverify.core.errors import PhysverifyError
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import PerturbationRecord, SceneSpec
from physverify.synth.generate import load_plan
from physverify.synth.gt_export import clip_manifest, video_path
from physverify.synth.plan import DatasetPlan
from physverify.synth.render import render_job
from physverify.synth.settings import DatasetSettings, dataset_root, load_dataset_settings
from physverify.synth.store import (
    FRAMES_DIR,
    KUBRIC_META,
    RECORD,
    SPEC,
    VARIANT_STATES,
    ContainerRunner,
    DatasetStore,
    Step,
    read_json,
    run_container,
)
from physverify.timing._base import TimingContext
from physverify.timing.hooks import TargetHooks
from physverify.timing.records import RENDER_PER_FRAME_S, RENDER_WALL_S, TimingError
from physverify.timing.report import discard_scratch, scratch_dir
from physverify.timing.services import TimingServices
from physverify.timing.settings import load_timing, run_settings

TASK: Final = "render"
JOBS_PER_CONTAINER: Final = 1  # isolates the per-clip cost, container start-up included
FRAMES_S: Final = "render.frames_s"
POSTPROCESS_S: Final = "render.postprocess_s"
OVERHEAD_S: Final = "render.overhead_s"


def applied_clip_ids(plan: DatasetPlan, store: DatasetStore) -> list[str]:
    """Clips whose variant states exist (apply step done) and were not skipped, in plan order."""
    return [
        v.clip_id
        for v in plan.variants
        if store.is_done(Step.APPLY, v.clip_id) and not store.is_skipped(v.clip_id)
    ]


def scratch_store(store: DatasetStore, plan: DatasetPlan, clip_id: str, root: Path) -> DatasetStore:
    """A store holding only what the render worker reads for ``clip_id``."""
    scratch = DatasetStore(root)
    sim_id = plan.variant(clip_id).sim_id
    for source, target in (
        (store.scene_dir(sim_id) / SPEC, scratch.scene_dir(sim_id) / SPEC),
        (store.clip_dir(clip_id) / VARIANT_STATES, scratch.clip_dir(clip_id) / VARIANT_STATES),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return scratch


def render_measures(meta: Mapping[str, Any], wall_s: float) -> dict[str, float]:
    frames = [float(seconds) for seconds in meta["frame_seconds"]]
    if not frames:
        raise TimingError(f"{meta.get('clip_id')}: kubric_meta.json has no frame times")
    postprocess = float(meta["postprocess_seconds"])
    return {
        RENDER_WALL_S: wall_s,
        FRAMES_S: sum(frames),
        RENDER_PER_FRAME_S: fmean(frames),
        POSTPROCESS_S: postprocess,
        OVERHEAD_S: wall_s - sum(frames) - postprocess,
    }


class KubricRender(TargetHooks):
    name: ClassVar[str] = "kubric_render"
    requires_gpu: ClassVar[bool] = False

    def __init__(
        self,
        services: TimingServices,
        *,
        which: Callable[[str], str | None] = shutil.which,
        image_check: Callable[[str], bool] = image_present,
    ) -> None:
        self._services = services
        self._which = which
        self._image_check = image_check
        self._runner: ContainerRunner | None = None
        self._calls = count()

    def unavailable(self, ctx: TimingContext) -> str | None:
        if self._which("docker") is None:
            return "docker is not on PATH"
        image = ctx.settings.runtime.docker_image
        if not self._image_check(image):
            return f"docker image {image!r} not present: run `physverify dataset pull-image`"
        _, store, plan = self._dataset(ctx)
        if plan is None:
            return f"no Kubric dataset plan in {store.root}: run `physverify dataset generate`"
        if not applied_clip_ids(plan, store):
            return "no Kubric clip has its variant states: run `physverify dataset perturb`"
        return self._mount_problem(ctx)

    def own_clips(self, ctx: TimingContext) -> Sequence[ClipManifest]:
        _, store, plan = self._dataset(ctx)
        if plan is None:
            return []
        limit = run_settings(load_timing(ctx.settings), self.name).max_clips
        clip_ids = applied_clip_ids(plan, store)[:limit]
        return [self._manifest(store, plan, clip_id, ctx) for clip_id in clip_ids]

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        return {
            "image": ctx.settings.runtime.docker_image,
            "jobs_per_container": str(JOBS_PER_CONTAINER),
        }

    def close(self) -> None:
        self._runner = None

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]:
        ds, store, plan = self._dataset(ctx)
        if plan is None:
            raise TimingError(f"no Kubric dataset plan in {store.root}")
        root = scratch_dir(ctx.run_dir, TASK, next(self._calls))
        try:
            scratch = scratch_store(store, plan, clip.clip_id, root)
            return self._render(clip.clip_id, plan, scratch, ds)
        finally:
            discard_scratch(root)

    def _render(
        self, clip_id: str, plan: DatasetPlan, scratch: DatasetStore, ds: DatasetSettings
    ) -> dict[str, float]:
        runner = self._container_runner()
        jobs = scratch.write_jobs(TASK, [render_job(clip_id, plan, scratch, runner, ds)])
        clock = self._services.clock
        started = clock()
        run_container(runner, TASK, jobs)
        wall_s = clock() - started
        failure = scratch.failure(Step.RENDER, clip_id)
        if failure is not None:
            raise TimingError(f"{clip_id}: render failed: {failure}")
        meta_path = scratch.clip_dir(clip_id) / KUBRIC_META
        if not meta_path.is_file():
            raise TimingError(f"{clip_id}: the render left no {KUBRIC_META}")
        return render_measures(read_json(meta_path), wall_s)

    def _container_runner(self) -> ContainerRunner:
        if self._runner is None:
            self._runner = self._services.docker()
        return self._runner

    def _mount_problem(self, ctx: TimingContext) -> str | None:
        try:
            self._container_runner().container_path(ctx.run_dir)
        except PhysverifyError as exc:
            return f"the run directory cannot be mounted in the container: {exc}"
        return None

    @staticmethod
    def _dataset(ctx: TimingContext) -> tuple[DatasetSettings, DatasetStore, DatasetPlan | None]:
        ds = load_dataset_settings(ctx.settings)
        store = DatasetStore(dataset_root(ctx.settings, ds))
        return ds, store, load_plan(store)

    @staticmethod
    def _manifest(
        store: DatasetStore, plan: DatasetPlan, clip_id: str, ctx: TimingContext
    ) -> ClipManifest:
        variant = plan.variant(clip_id)
        clip_dir = store.clip_dir(clip_id)
        return clip_manifest(
            variant,
            SceneSpec.load_json(store.scene_dir(variant.sim_id) / SPEC),
            PerturbationRecord.load_json(clip_dir / RECORD),
            video_path(clip_dir / FRAMES_DIR, ctx.settings.paths.data_dir),
        )


PLUGINS = (KubricRender,)
