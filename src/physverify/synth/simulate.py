"""The ``simulate`` step: one container run of the ``simulate`` task over every pending sim_id."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import DatasetSettings
from physverify.synth.store import (
    FAILURE_FILE,
    RAW_COLLISIONS,
    SIM_META,
    SIM_STATES,
    SPEC,
    ContainerRunner,
    DatasetStore,
    Step,
    run_container,
)


@dataclass(frozen=True)
class StepReport:
    launched: int
    done: int
    failed: tuple[str, ...]


def pending_sims(plan: DatasetPlan, store: DatasetStore) -> list[str]:
    """sim_ids with a spec and without ``sim_meta.json`` (refused scene edits have no spec)."""
    return [
        sim_id
        for sim_id in plan.sim_ids()
        if store.is_done(Step.GENERATE, sim_id) and not store.is_done(Step.SIMULATE, sim_id)
    ]


def simulate_job(
    sim_id: str, store: DatasetStore, runner: ContainerRunner, ds: DatasetSettings
) -> dict[str, Any]:
    sim_dir = store.scene_dir(sim_id)
    return {
        "sim_id": sim_id,
        "spec_path": runner.container_path(sim_dir / SPEC),
        "states_path": runner.container_path(sim_dir / SIM_STATES),
        "raw_path": runner.container_path(sim_dir / RAW_COLLISIONS),
        "meta_path": runner.container_path(sim_dir / SIM_META),
        "failed_path": runner.container_path(sim_dir / FAILURE_FILE),
        "floor": ds.floor.model_dump(),
    }


def simulate(
    plan: DatasetPlan, store: DatasetStore, runner: ContainerRunner, ds: DatasetSettings
) -> StepReport:
    pending = pending_sims(plan, store)
    for sim_id in pending:
        store.clear_failure(Step.SIMULATE, sim_id)
    if pending:
        jobs = [simulate_job(sim_id, store, runner, ds) for sim_id in pending]
        run_container(runner, "simulate", store.write_jobs("simulate", jobs))
    failed = tuple(s for s in pending if store.failure(Step.SIMULATE, s) is not None)
    done = sum(store.is_done(Step.SIMULATE, s) for s in pending)
    return StepReport(launched=len(pending), done=done, failed=failed)
