"""Resumable on-disk layout: markers, failures, skips and content-named job files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from physverify.core.errors import PhysverifyError, UsageError
from physverify.synth.generate import plan_subset, write_specs
from physverify.synth.perturb import apply_variants, closest_points, ground_truth_contacts
from physverify.synth.render import pending_renders, render
from physverify.synth.settings import DatasetSettings
from physverify.synth.simulate import simulate
from physverify.synth.store import (
    MARKERS,
    PROVISIONAL_CONTACTS,
    DatasetStore,
    Step,
    atomic_write_json,
    read_json,
    run_container,
)


def test_markers_live_in_scene_or_clip_directories(store: DatasetStore) -> None:
    for step, marker in MARKERS.items():
        item = "scene-1" if step in (Step.GENERATE, Step.SIMULATE) else "clip-1"
        assert not store.is_done(step, item)
        atomic_write_json(store.item_dir(step, item) / marker, {})
        assert store.is_done(step, item)
    assert store.item_dir(Step.SIMULATE, "x") == store.scene_dir("x")
    assert store.item_dir(Step.RENDER, "x") == store.clip_dir("x")


def test_failures_and_skips(store: DatasetStore) -> None:
    atomic_write_json(store.clip_dir("c") / "failed.json", {"error": "boom"})
    assert store.failure(Step.RENDER, "c") == {"error": "boom"}
    store.clear_failure(Step.RENDER, "c")
    assert store.failure(Step.RENDER, "c") is None
    store.mark_skipped("c", "no ball in flight")
    assert store.is_skipped("c") and store.skip_reason("c") == "no ball in flight"


def test_job_files_are_named_by_content(store: DatasetStore) -> None:
    first = store.write_jobs("render", [{"clip_id": "a"}])
    assert first == store.write_jobs("render", [{"clip_id": "a"}])
    assert first != store.write_jobs("render", [{"clip_id": "b"}])
    assert read_json(first) == {"clip_id": "a"}


def test_atomic_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "a" / "b.json", {"x": 1})
    assert [p.name for p in (tmp_path / "a").iterdir()] == ["b.json"]


def test_a_failed_container_is_an_error(tmp_path: Path) -> None:
    ok = SimpleNamespace(run=lambda task, jobs: SimpleNamespace(returncode=0))
    run_container(ok, "render", tmp_path)  # type: ignore[arg-type]
    broken = SimpleNamespace(run=lambda task, jobs: SimpleNamespace(returncode=125))
    with pytest.raises(PhysverifyError, match="125"):
        run_container(broken, "render", tmp_path)  # type: ignore[arg-type]


OUTPUT_OF_TASK = {"simulate": "meta_path", "contacts": "out_path", "render": "meta_path"}


class FakeRunner:
    """Stands in for the container: writes each job's completion file and counts the jobs."""

    def __init__(self) -> None:
        self.jobs: dict[str, int] = {}

    def container_path(self, host_path: Path) -> str:
        return str(host_path)

    def run(self, task: str, jobs: Path) -> Any:
        lines = [json.loads(line) for line in jobs.read_text().splitlines()]
        for job in lines:
            atomic_write_json(Path(job[OUTPUT_OF_TASK[task]]), {"fake": True})
        self.jobs[task] = self.jobs.get(task, 0) + len(lines)
        return SimpleNamespace(returncode=0)


def mark(store: DatasetStore, step: Step, clip_ids: list[str]) -> None:
    for clip_id in clip_ids:
        atomic_write_json(store.item_dir(step, clip_id) / MARKERS[step], {"fake": True})


def test_a_second_pass_of_every_step_launches_no_job(
    ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    plugins: dict[str, Any],
    priors: dict[str, Any],
    store: DatasetStore,
) -> None:
    plan = plan_subset(ds, families, perturbations, "smoke")
    clips = [v.clip_id for v in plan.variants]
    runner = FakeRunner()
    assert write_specs(plan, store, families, plugins, ds, priors)
    assert write_specs(plan, store, families, plugins, ds, priors) == []

    assert simulate(plan, store, runner, ds).launched == len(plan.sim_ids())
    assert simulate(plan, store, runner, ds).launched == 0
    mark(store, Step.APPLY, clips)
    for clip_id in clips:
        atomic_write_json(store.clip_dir(clip_id) / PROVISIONAL_CONTACTS, {"fake": True})
    assert apply_variants(plan, store, plugins, ds).applied == ()
    assert closest_points(plan, store, runner, ds) == len(clips)
    assert closest_points(plan, store, runner, ds) == 0
    mark(store, Step.PERTURB, clips)
    assert ground_truth_contacts(plan, store, ds) == []
    assert render(plan, store, runner, ds).launched == len(clips)
    assert render(plan, store, runner, ds).launched == 0

    assert runner.jobs == {
        "simulate": len(plan.sim_ids()),
        "contacts": len(clips),
        "render": len(clips),
    }


def test_pending_renders_honours_limit_and_clip_ids(
    ds: DatasetSettings, families: Any, perturbations: Any, store: DatasetStore
) -> None:
    plan = plan_subset(ds, families, perturbations, "local")
    clips = [v.clip_id for v in plan.variants]
    mark(store, Step.PERTURB, clips)
    mark(store, Step.RENDER, clips[:2])
    store.mark_skipped(clips[2], "refused")
    ready = clips[3:]
    assert pending_renders(plan, store) == ready
    assert pending_renders(plan, store, limit=2) == ready[:2]
    assert pending_renders(plan, store, clip_ids=[clips[0], ready[-1], ready[0]]) == [
        ready[0],
        ready[-1],
    ]
    assert pending_renders(plan, store, limit=1, clip_ids=[ready[-1], ready[0]]) == [ready[0]]
    with pytest.raises(UsageError, match="no-such-clip") as info:
        pending_renders(plan, store, clip_ids=[ready[0], "no-such-clip"])
    assert info.value.exit_code == 2
