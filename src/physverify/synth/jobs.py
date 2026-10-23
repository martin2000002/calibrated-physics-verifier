"""The steps of ``physverify synth`` on one subset, with the context and output injected."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from physverify.config.plugin_context import PluginContext, registry_of
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.core.output import Echo
from physverify.runtime.container import ContainerEngine, missing_reason
from physverify.runtime.docker import Runner
from physverify.synth.families._base import FAMILIES
from physverify.synth.generate import (
    load_plan,
    plan_subset,
    save_plan,
    subset_settings,
    write_specs,
)
from physverify.synth.gt_export import export_all
from physverify.synth.perturb import (
    apply_variants,
    closest_points,
    ground_truth_contacts,
    instantiate,
)
from physverify.synth.perturbations._base import PERTURBATIONS, Perturbation
from physverify.synth.plan import DatasetPlan
from physverify.synth.render import render as render_clips
from physverify.synth.render import seconds_per_frame
from physverify.synth.settings import (
    DatasetSettings,
    dataset_root,
    load_dataset_settings,
    load_size_priors,
)
from physverify.synth.simulate import StepReport
from physverify.synth.simulate import simulate as simulate_scenes
from physverify.synth.store import ContainerRunner, DatasetStore, Step
from physverify.synth.truth_measures._base import TRUTH_MEASURES


class JobContext(PluginContext, Protocol):
    """What a dataset job reads from the application context."""

    def docker(self) -> ContainerEngine: ...


@dataclass
class DatasetJob:
    """What every dataset step needs, resolved from the injected job context."""

    context: JobContext
    subset: str | None
    echo: Echo

    @cached_property
    def ds(self) -> DatasetSettings:
        ds = load_dataset_settings(self.context.settings)
        if self.subset is not None:
            subset_settings(ds, self.subset)
        return ds

    @cached_property
    def store(self) -> DatasetStore:
        return DatasetStore(dataset_root(self.context.settings, self.ds))

    @cached_property
    def plugins(self) -> dict[str, Perturbation]:
        return instantiate(registry_of(self.context, PERTURBATIONS), self.ds)

    def plan(self, current: bool = True) -> DatasetPlan:
        """The stored plan of the subset; ``current`` refuses one built before this checkout's
        ground-truth revision (it must be migrated first)."""
        plan = load_plan(self.store)
        if plan is None:
            raise MissingInputError(
                f"nothing planned in {self.store.root}: run `physverify synth generate "
                f"--subset {self.subset or 'local'}`"
            )
        if current:
            plan.check_gt_revision()
        return plan.subset(self.subset)

    def subset_name(self) -> str:
        if self.subset is None:
            raise UsageError("this step needs --subset")
        return self.subset

    def runner(self, run: Runner = subprocess.run) -> ContainerRunner:
        runner = self.context.docker()
        reason = missing_reason(self.context.settings, runner, run)
        if reason is not None:
            raise MissingInputError(f"the Kubric container is not available: {reason}")
        return runner


def _report(echo: Echo, step: str, report: StepReport) -> list[str]:
    echo(f"{step}: {report.launched} launched, {report.done} done, {len(report.failed)} failed")
    return [f"{step} {item}" for item in report.failed]


def fail_if(failures: Sequence[str]) -> None:
    if failures:
        raise PhysverifyError(
            f"{len(failures)} failed (see failed.json in each item): {', '.join(failures[:10])}"
        )


def run_generate(job: DatasetJob) -> DatasetPlan:
    registry = registry_of(job.context, FAMILIES)
    perturbations = registry_of(job.context, PERTURBATIONS)
    plan = plan_subset(job.ds, registry, perturbations, job.subset_name())
    save_plan(job.store, plan)
    written = write_specs(
        plan, job.store, registry, job.plugins, job.ds, load_size_priors(job.context.settings)
    )
    skipped = sum(job.store.is_skipped(v.clip_id) for v in plan.variants)
    job.echo(
        f"generate: {len(plan.scenes)} scenes, {len(plan.variants)} clips, "
        f"{len(plan.sim_ids())} simulations; {len(written)} specs written, {skipped} skipped"
    )
    return plan


def run_simulate(job: DatasetJob, runner: ContainerRunner) -> list[str]:
    return _report(job.echo, "simulate", simulate_scenes(job.plan(), job.store, runner, job.ds))


def run_perturb(job: DatasetJob, runner: ContainerRunner) -> list[str]:
    plan = job.plan()
    applied = apply_variants(plan, job.store, job.plugins, job.ds).applied
    launched = closest_points(plan, job.store, runner, job.ds)
    contacts = ground_truth_contacts(plan, job.store, job.ds)
    failed = [
        v.clip_id for v in plan.variants if job.store.failure(Step.PERTURB, v.clip_id) is not None
    ]
    job.echo(
        f"perturb: {len(applied)} applied, {launched} closest-point jobs, "
        f"{len(contacts)} contact tables, {len(failed)} failed"
    )
    return [f"perturb {c}" for c in failed]


def run_render(
    job: DatasetJob, runner: ContainerRunner, limit: int | None, clips: Sequence[str] | None
) -> list[str]:
    report = render_clips(job.plan(), job.store, runner, job.ds, limit=limit, clip_ids=clips)
    failures = _report(job.echo, "render", report)
    rendered = [v.clip_id for v in job.plan().variants if job.store.is_done(Step.RENDER, v.clip_id)]
    mean = seconds_per_frame(job.store, rendered)
    if mean is not None:
        job.echo(f"render: {mean:.2f} s per frame over {len(rendered)} clips")
    return failures


def run_export(job: DatasetJob) -> None:
    exported = export_all(
        job.plan(),
        job.store,
        job.plugins,
        registry_of(job.context, TRUTH_MEASURES),
        job.ds,
        job.context.settings.paths.data_dir,
    )
    job.echo(f"export-gt: {len(exported)} clips exported; index {job.store.manifest_path}")


def status_lines(plan: DatasetPlan, store: DatasetStore) -> list[str]:
    """Per step: done, failed, skipped and pending items of the planned subset."""
    lines = []
    sims = plan.sim_ids()
    for step in (Step.GENERATE, Step.SIMULATE):
        done = sum(store.is_done(step, s) for s in sims)
        failed = sum(store.failure(step, s) is not None for s in sims)
        lines.append(f"{step.value:9} {done:4} done {failed:4} failed {len(sims) - done:4} pending")
    clips = [v.clip_id for v in plan.variants]
    skipped = [c for c in clips if store.is_skipped(c)]
    for step in (Step.APPLY, Step.PERTURB, Step.RENDER, Step.EXPORT):
        done = sum(store.is_done(step, c) for c in clips)
        failed = sum(store.failure(step, c) is not None for c in clips)
        pending = len(clips) - done - len(skipped)
        lines.append(
            f"{step.value:9} {done:4} done {failed:4} failed {pending:4} pending "
            f"{len(skipped):4} skipped"
        )
    lines += [f"skipped {c}: {store.skip_reason(c)}" for c in skipped]
    mean = seconds_per_frame(store, [c for c in clips if store.is_done(Step.RENDER, c)])
    if mean is not None:
        lines.append(f"render: {mean:.2f} s per frame (measured)")
    return lines
