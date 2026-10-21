"""The ``render`` step: batches of ``jobs_per_process`` clips per container, ``workers``
containers in parallel. The worker keyframes the variant states (position, orientation, scale
from ``size``, hidden below the floor where ``visible`` is false) and writes ``layers.npz`` and
``kubric_meta.json``."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from physverify.core.errors import UsageError
from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import DatasetSettings
from physverify.synth.simulate import StepReport
from physverify.synth.store import (
    FAILURE_FILE,
    KUBRIC_META,
    LAYERS,
    SPEC,
    VARIANT_STATES,
    ContainerRunner,
    DatasetStore,
    Step,
    read_json,
    run_container,
)


def pending_renders(
    plan: DatasetPlan,
    store: DatasetStore,
    *,
    limit: int | None = None,
    clip_ids: Sequence[str] | None = None,
) -> list[str]:
    """Clips ready to render (contacts done, not skipped, not rendered), in plan order, restricted
    to ``clip_ids`` and cut to ``limit``; ids that the plan does not know are a usage error."""
    wanted = set(clip_ids) if clip_ids else None
    unknown = sorted((wanted or set()) - {v.clip_id for v in plan.variants})
    if unknown:
        raise UsageError(f"--clips names clips that are not in the plan: {unknown}")
    pending = [
        v.clip_id
        for v in plan.variants
        if (wanted is None or v.clip_id in wanted)
        and store.is_done(Step.PERTURB, v.clip_id)
        and not store.is_skipped(v.clip_id)
        and not store.is_done(Step.RENDER, v.clip_id)
    ]
    return pending if limit is None else pending[:limit]


def render_job(
    clip_id: str,
    plan: DatasetPlan,
    store: DatasetStore,
    runner: ContainerRunner,
    ds: DatasetSettings,
) -> dict[str, Any]:
    clip_dir = store.clip_dir(clip_id)
    return {
        "clip_id": clip_id,
        "spec_path": runner.container_path(store.scene_dir(plan.variant(clip_id).sim_id) / SPEC),
        "states_path": runner.container_path(clip_dir / VARIANT_STATES),
        "layers_path": runner.container_path(clip_dir / LAYERS),
        "meta_path": runner.container_path(clip_dir / KUBRIC_META),
        "failed_path": runner.container_path(clip_dir / FAILURE_FILE),
        "floor": ds.floor.model_dump(),
        "hdri_ids": list(ds.effective_hdri_ids()),
    }


def batches(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[k : k + size]) for k in range(0, len(items), size)]


def render(
    plan: DatasetPlan,
    store: DatasetStore,
    runner: ContainerRunner,
    ds: DatasetSettings,
    *,
    limit: int | None = None,
    clip_ids: Sequence[str] | None = None,
) -> StepReport:
    pending = pending_renders(plan, store, limit=limit, clip_ids=clip_ids)
    for clip_id in pending:
        store.clear_failure(Step.RENDER, clip_id)
    job_files = [
        store.write_jobs("render", [render_job(c, plan, store, runner, ds) for c in batch])
        for batch in batches(pending, ds.render.jobs_per_process)
    ]
    with ThreadPoolExecutor(max_workers=ds.render.workers) as pool:
        list(pool.map(lambda path: run_container(runner, "render", path), job_files))
    failed = tuple(c for c in pending if store.failure(Step.RENDER, c) is not None)
    done = sum(store.is_done(Step.RENDER, c) for c in pending)
    return StepReport(launched=len(pending), done=done, failed=failed)


def seconds_per_frame(store: DatasetStore, clip_ids: Sequence[str]) -> float | None:
    """Mean wall time per rendered frame over the clips' ``kubric_meta.json`` (None if none)."""
    totals = [
        read_json(store.clip_dir(c) / KUBRIC_META)
        for c in clip_ids
        if store.is_done(Step.RENDER, c)
    ]
    frames = sum(len(meta["frame_seconds"]) for meta in totals)
    return sum(sum(meta["frame_seconds"]) for meta in totals) / frames if frames else None
