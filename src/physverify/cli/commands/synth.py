"""``physverify synth``: the Kubric synthetic dataset, step by step or end to end.

Container steps take the heavy lock; ``build`` holds it for the whole run."""

from __future__ import annotations

import subprocess
from typing import Annotated

import typer

from physverify.cli.context import app_context, heavy_command
from physverify.core.errors import PhysverifyError, UsageError
from physverify.runtime.container import container_settings
from physverify.synth.jobs import (
    DatasetJob,
    fail_if,
    run_export,
    run_generate,
    run_perturb,
    run_render,
    run_simulate,
    status_lines,
)
from physverify.synth.migrate import run_migrate, velocity_migration
from physverify.synth.plan import GT_MIGRATIONS

NAME = "synth"

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


def _job(subset: str | None) -> DatasetJob:
    job = DatasetJob(app_context(), subset, typer.echo)
    _ = job.ds
    return job


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
    fail_if(run_simulate(job, job.runner()))


@app.command("perturb", help="Apply perturbations and compute contact truth (heavy).")
@heavy_command
def perturb(subset: SubsetOption = "local") -> None:
    job = _job(subset)
    fail_if(run_perturb(job, job.runner()))


@app.command("render", help="Render the pending clips in the container (heavy, resumable).")
@heavy_command
def render(
    subset: SubsetOption = "local",
    limit: Annotated[int | None, typer.Option(min=1, help="Render at most N clips.")] = None,
    clips: Annotated[str | None, typer.Option(help="Comma-separated clip ids.")] = None,
) -> None:
    job = _job(subset)
    wanted = [c.strip() for c in clips.split(",") if c.strip()] if clips else None
    fail_if(run_render(job, job.runner(), limit, wanted))


@app.command("export-gt", help="Export frames, camera, states, contacts and final records.")
def export_gt(subset: SubsetOption = "local") -> None:
    run_export(_job(subset))


@app.command("list", help="Progress of every step for the subset, skipped clips and s/frame.")
def list_command(subset: SubsetOption = "local") -> None:
    job = _job(subset)
    for line in status_lines(job.plan(current=False), job.store):
        typer.echo(line)


CommitOption = Annotated[
    str | None, typer.Option(help="Commit to log (default: HEAD of the repository).")
]
NameOption = Annotated[
    str | None,
    typer.Option(
        "--name",
        help="velocity-<convention> (WP-G 1.C, the default) or the ground-truth fix the plan "
        f"needs next, in order: {', '.join(GT_MIGRATIONS)}.",
    ),
]


@app.command(
    "migrate",
    help="Re-run APPLY on every simulated clip with this checkout (a named migration: "
    "velocity-<convention> or the next ground-truth fix), keeping the version and every render "
    "whose inputs did not change; writes migrations/<name>.json.",
)
@heavy_command
def migrate_command(name: NameOption = None, commit: CommitOption = None) -> None:
    job = _job(None)
    run_migrate(job, name or velocity_migration(job.ds), commit)


@app.command("migrate-velocity", help="Alias of `migrate --name velocity-<convention>` (WP-G 1.C).")
@heavy_command
def velocity_alias(commit: CommitOption = None) -> None:
    job = _job(None)
    run_migrate(job, velocity_migration(job.ds), commit)


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
    fail_if(failures)
