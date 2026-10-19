"""``physverify dataset``: the Kubric synthetic dataset, step by step or end to end (ADR 0004,
section 3.10). Container steps take the heavy lock; ``build`` holds it for the whole run."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any

import typer

from physverify.cli.context import AppContext, app_context, heavy_command
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.runtime.container import container_settings, missing_reason
from physverify.runtime.runs import current_commit
from physverify.synth.generate import (
    load_plan,
    plan_subset,
    save_plan,
    subset_settings,
    write_specs,
)
from physverify.synth.gt_export import export_all
from physverify.synth.migrate import migrate_velocity
from physverify.synth.perturb import (
    apply_variants,
    closest_points,
    ground_truth_contacts,
    instantiate,
)
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

NAME = "dataset"
OWNER = "A4"
IMPLEMENTED = True

Run = Callable[..., "subprocess.CompletedProcess[bytes]"]

SubsetOption = Annotated[
    str,
    typer.Option(
        "--subset",
        help="smoke (2 clips), local (Deliverable 1 pipeline proof) or full (H200 dataset).",
    ),
]

app = typer.Typer(
    help="Kubric synthetic dataset: image, scenes, simulation, perturbations, rendering, GT.",
    no_args_is_help=True,
)


@dataclass
class DatasetJob:
    """What every subcommand needs, resolved from the CLI context."""

    context: AppContext
    subset: str | None

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
    def plugins(self) -> dict[str, Any]:
        return instantiate(self.context.registry("perturbations"), self.ds)

    def plan(self) -> DatasetPlan:
        plan = load_plan(self.store)
        if plan is None:
            raise MissingInputError(
                f"nothing planned in {self.store.root}: run `physverify dataset generate "
                f"--subset {self.subset or 'local'}`"
            )
        return plan.subset(self.subset)

    def subset_name(self) -> str:
        if self.subset is None:
            raise UsageError("this step needs --subset")
        return self.subset

    def runner(self, run: Run = subprocess.run) -> ContainerRunner:
        runner = self.context.docker()
        reason = missing_reason(self.context.settings, runner, run)
        if reason is not None:
            raise MissingInputError(f"the Kubric container is not available: {reason}")
        return runner


def image_present(image: str, run: Run = subprocess.run) -> bool:
    return (
        run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode == 0
    )


def _job(subset: str | None) -> DatasetJob:
    job = DatasetJob(app_context(), subset)
    _ = job.ds
    return job


def _report(step: str, report: StepReport) -> list[str]:
    typer.echo(
        f"{step}: {report.launched} launched, {report.done} done, {len(report.failed)} failed"
    )
    return [f"{step} {item}" for item in report.failed]


def _fail_if(failures: Sequence[str]) -> None:
    if failures:
        raise PhysverifyError(
            f"{len(failures)} failed (see failed.json in each item): {', '.join(failures[:10])}"
        )


def run_generate(job: DatasetJob) -> DatasetPlan:
    registry = job.context.registry("families")
    perturbations = job.context.registry("perturbations")
    plan = plan_subset(job.ds, registry, perturbations, job.subset_name())
    save_plan(job.store, plan)
    written = write_specs(
        plan, job.store, registry, job.plugins, job.ds, load_size_priors(job.context.settings)
    )
    skipped = sum(job.store.is_skipped(v.clip_id) for v in plan.variants)
    typer.echo(
        f"generate: {len(plan.scenes)} scenes, {len(plan.variants)} clips, "
        f"{len(plan.sim_ids())} simulations; {len(written)} specs written, {skipped} skipped"
    )
    return plan


def run_simulate(job: DatasetJob, runner: ContainerRunner) -> list[str]:
    return _report("simulate", simulate_scenes(job.plan(), job.store, runner, job.ds))


def run_perturb(job: DatasetJob, runner: ContainerRunner) -> list[str]:
    plan = job.plan()
    applied = apply_variants(plan, job.store, job.plugins, job.ds).applied
    launched = closest_points(plan, job.store, runner, job.ds)
    contacts = ground_truth_contacts(plan, job.store, job.ds)
    failed = [
        v.clip_id for v in plan.variants if job.store.failure(Step.PERTURB, v.clip_id) is not None
    ]
    typer.echo(
        f"perturb: {len(applied)} applied, {launched} closest-point jobs, "
        f"{len(contacts)} contact tables, {len(failed)} failed"
    )
    return [f"perturb {c}" for c in failed]


def run_render(
    job: DatasetJob, runner: ContainerRunner, limit: int | None, clips: Sequence[str] | None
) -> list[str]:
    report = render_clips(job.plan(), job.store, runner, job.ds, limit=limit, clip_ids=clips)
    failures = _report("render", report)
    rendered = [v.clip_id for v in job.plan().variants if job.store.is_done(Step.RENDER, v.clip_id)]
    mean = seconds_per_frame(job.store, rendered)
    if mean is not None:
        typer.echo(f"render: {mean:.2f} s per frame over {len(rendered)} clips")
    return failures


def run_export(job: DatasetJob) -> None:
    exported = export_all(
        job.plan(), job.store, job.plugins, job.ds, job.context.settings.paths.data_dir
    )
    typer.echo(f"export-gt: {len(exported)} clips exported; index {job.store.manifest_path}")


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


@app.command("pull-image", help="Pull the pinned Kubric image (heavy).")
@heavy_command
def pull_image() -> None:
    settings = app_context().settings
    engine = container_settings(settings).engine
    if engine != "docker":
        raise UsageError(
            f"pull-image only drives docker; container.engine is {engine!r} "
            "(on the H200, scripts/h200/run_kubric.sh pulls and creates the container)"
        )
    image = settings.runtime.docker_image
    if subprocess.run(["docker", "pull", image], check=False).returncode != 0:
        raise PhysverifyError(f"docker pull {image} failed")


@app.command("generate", help="Plan the subset and write the scene specs.")
def generate(subset: SubsetOption = "local") -> None:
    run_generate(_job(subset))


@app.command("simulate", help="Simulate the pending scenes in the container (heavy).")
@heavy_command
def simulate(subset: SubsetOption = "local") -> None:
    job = _job(subset)
    _fail_if(run_simulate(job, job.runner()))


@app.command("perturb", help="Apply perturbations and compute contact truth (heavy).")
@heavy_command
def perturb(subset: SubsetOption = "local") -> None:
    job = _job(subset)
    _fail_if(run_perturb(job, job.runner()))


@app.command("render", help="Render the pending clips in the container (heavy, resumable).")
@heavy_command
def render(
    subset: SubsetOption = "local",
    limit: Annotated[int | None, typer.Option(min=1, help="Render at most N clips.")] = None,
    clips: Annotated[str | None, typer.Option(help="Comma-separated clip ids.")] = None,
) -> None:
    job = _job(subset)
    wanted = [c.strip() for c in clips.split(",") if c.strip()] if clips else None
    _fail_if(run_render(job, job.runner(), limit, wanted))


@app.command("export-gt", help="Export frames, camera, states, contacts and final records.")
def export_gt(subset: SubsetOption = "local") -> None:
    run_export(_job(subset))


@app.command("list", help="Progress of every step for the subset, skipped clips and s/frame.")
def list_command(subset: SubsetOption = "local") -> None:
    job = _job(subset)
    for line in status_lines(job.plan(), job.store):
        typer.echo(line)


@app.command(
    "migrate-velocity",
    help="Re-run APPLY on every simulated clip under the configured velocity_convention, keeping "
    "the version and every render whose inputs did not change; writes migrations/<name>.json.",
)
@heavy_command
def migrate_velocity_command(
    commit: Annotated[
        str | None, typer.Option(help="Commit to log (default: HEAD of the repository).")
    ] = None,
) -> None:
    job = _job(None)
    settings = job.context.settings
    report = migrate_velocity(
        job.store, job.plugins, job.ds, commit or current_commit(settings.repo_root)
    )
    typer.echo(
        f"migrate-velocity: {report.applied} applied, {len(report.rerender)} renders discarded, "
        f"{len(report.status_changes)} status changes; manifest {report.manifest}"
    )
    typer.echo("next: `physverify dataset build --subset <name>` for every built subset")


@app.command("build", help="generate, simulate, perturb, render and export-gt under one lock.")
@heavy_command
def build(subset: SubsetOption = "local") -> None:
    job = _job(subset)
    runner = job.runner()
    run_generate(job)
    failures = run_simulate(job, runner)
    failures += run_perturb(job, runner)
    failures += run_render(job, runner, None, None)
    run_export(job)
    _fail_if(failures)
